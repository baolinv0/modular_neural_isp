import tempfile
import unittest
from pathlib import Path

import torch

from cross_camera_tm.contracts import FailureType, GateStatus, canonical_tensor_sha256
from cross_camera_tm.policy import (
    InternVLAnonymousABAdapter,
    LocalModelSpec,
    OpenSourceLocalPolicy,
    OvisLocalInspector,
    Qwen3VLDiagnosisAdapter,
    QwenImageEditLocalAdapter,
)
from cross_camera_tm.projection import PreProjectionSafetyChecker, TMSpaceProjector
from cross_camera_tm.residuals import ResidualComponent, ResidualEstimate
from cross_camera_tm.teachers import L1GlobalTeacher, L2ROITeacher, L3LocalTeacher
from cross_camera_tm.tm_residual import TMResidualAdapter


def estimate():
    return ResidualEstimate(
        available=True,
        reason="ok",
        global_residual=ResidualComponent(0.10, 0.06, 0.16, 0.02),
        face_residual=ResidualComponent(0.18, 0.12, 0.23, 0.03),
        luma_low=0.08,
        luma_high=0.72,
        support_distance=0.1,
        neighbor_count=5,
    )


class TeacherTests(unittest.TestCase):
    def test_l1_is_global_and_l2_is_roi_distinct_with_soft_boundary(self):
        image = torch.full((1, 3, 32, 32), 0.25)
        face = torch.zeros(1, 1, 32, 32)
        face[:, :, 8:24, 8:24] = 1.0
        l1 = L1GlobalTeacher().generate(image, estimate())
        l2 = L2ROITeacher().generate(image, estimate(), face)
        l1_ratio = l1.image / image
        self.assertLess(float(l1_ratio.max() - l1_ratio.min()), 1e-5)
        center = float(l2.image[:, :, 16, 16].mean())
        corner = float(l2.image[:, :, 0, 0].mean())
        transition = float(l2.image[:, :, 7, 16].mean())
        self.assertGreater(center, corner)
        self.assertGreater(transition, corner)
        self.assertLess(transition, center)
        self.assertEqual(l1.level, "L1")
        self.assertEqual(l2.level, "L2")

    def test_l3_is_raw_proposal_only(self):
        image = torch.full((1, 3, 8, 8), 0.2)
        editor = QwenImageEditLocalAdapter.deterministic_mock(lambda value, prompt: value + 0.05)
        candidate = L3LocalTeacher(editor).propose(image, "lift face midtones")
        self.assertTrue(candidate.raw_generated)
        self.assertFalse(candidate.pixel_target_eligible)
        self.assertEqual(candidate.level, "L3")
        self.assertNotEqual(candidate.image_sha256, canonical_tensor_sha256(image))


class ProjectionAndResidualTests(unittest.TestCase):
    def test_preprojection_safety_fails_closed(self):
        checker = PreProjectionSafetyChecker()
        reference = torch.full((1, 3, 8, 8), 0.2)
        malformed = torch.full((1, 3, 7, 8), 0.3)
        invalid = reference.clone()
        invalid[0, 0, 0, 0] = float("nan")
        self.assertIs(checker.check(reference, malformed).status, GateStatus.FAIL)
        self.assertIs(checker.check(reference, invalid).status, GateStatus.FAIL)

    def test_projection_discards_color_and_detail_and_creates_lineage_hash(self):
        reference = torch.linspace(0.1, 0.7, 16 * 16).view(1, 1, 16, 16).repeat(1, 3, 1, 1)
        proposal = reference.clone()
        proposal[:, 0] = (proposal[:, 0] * 1.3 + 0.04).clamp(0, 1)
        proposal[:, :, ::2, ::2] = (proposal[:, :, ::2, ::2] + 0.04).clamp(0, 1)
        projected = TMSpaceProjector().project(reference, proposal)
        self.assertNotEqual(projected.raw_proposal_sha256, projected.projected_sha256)
        self.assertFalse(projected.raw_generated_eligible)
        self.assertTrue(projected.requires_full_recertification)
        reference_chroma = reference / reference.sum(dim=1, keepdim=True).clamp_min(1e-6)
        projected_chroma = projected.image / projected.image.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.assertTrue(torch.allclose(reference_chroma, projected_chroma, atol=2e-5))
        self.assertTrue(torch.all(projected.curve_y[:, 1:] >= projected.curve_y[:, :-1]))

    def test_tm_residual_identity_full_image_and_bounded_smooth_gain(self):
        adapter = TMResidualAdapter(curve_points=6, grid_size=8)
        image = torch.rand(1, 3, 32, 32) * 0.7
        identity = adapter(image)
        self.assertTrue(torch.allclose(identity.image, image, atol=1e-6))
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 69)
        with torch.no_grad():
            adapter.log_gain_grid[0, 0, 4, 4] = 3.0
            adapter.curve_logits.copy_(torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
        changed = adapter(image)
        self.assertEqual(changed.image.shape, image.shape)
        self.assertTrue(torch.all(changed.curve_y[:, 1:] >= changed.curve_y[:, :-1]))
        self.assertLessEqual(float(changed.log_gain_map.abs().max()), adapter.max_log_gain + 1e-6)
        horizontal_jump = (changed.log_gain_map[:, :, :, 1:] - changed.log_gain_map[:, :, :, :-1]).abs().max()
        self.assertLess(float(horizontal_jump), 0.08)


class LocalPolicyTests(unittest.TestCase):
    def test_open_source_local_policy_rejects_remote_and_closed_models(self):
        policy = OpenSourceLocalPolicy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.bin"
            checkpoint.write_bytes(b"local-test")
            accepted = policy.validate(
                LocalModelSpec("Qwen3-VL", "Apache-2.0", str(checkpoint), remote_api=False)
            )
            remote = policy.validate(
                LocalModelSpec("Qwen3-VL", "Apache-2.0", str(checkpoint), remote_api=True)
            )
            closed = policy.validate(
                LocalModelSpec("closed", "proprietary", str(checkpoint), remote_api=False)
            )
        self.assertIs(accepted.status, GateStatus.PASS)
        self.assertIs(remote.status, GateStatus.FAIL)
        self.assertIs(closed.status, GateStatus.FAIL)

    def test_diagnosis_mock_is_explicit_and_cannot_set_strength(self):
        adapter = Qwen3VLDiagnosisAdapter.deterministic_mock(
            lambda image: {"failure_type": "face_underexposure", "roi_hint": [1, 1, 4, 4]}
        )
        result = adapter.diagnose(torch.zeros(1, 3, 8, 8))
        self.assertTrue(result.synthetic_mock)
        self.assertIs(result.failure_type, FailureType.FACE_UNDEREXPOSURE)
        self.assertFalse(hasattr(result, "ev") or hasattr(result, "strength"))

    def test_internvl_ab_ba_and_optional_ovis_are_explicit_non_acceptance_evidence(self):
        arbiter = InternVLAnonymousABAdapter.deterministic_mock(
            lambda first, second: "A" if first.mean() > second.mean() else "B"
        )
        projected = torch.full((1, 3, 4, 4), 0.3)
        baseline = torch.full((1, 3, 4, 4), 0.2)
        result = arbiter.arbitrate(projected, baseline)
        inspection = OvisLocalInspector.deterministic_mock(
            lambda image: {"available": True, "hard_defect": False, "reason": "mechanical"}
        ).inspect(projected)
        self.assertTrue(result.available)
        self.assertTrue(result.order_consistent)
        self.assertEqual(result.preference, "PROJECTED")
        self.assertTrue(result.synthetic_mock)
        self.assertTrue(inspection.available)
        self.assertFalse(inspection.hard_defect)
        self.assertTrue(inspection.synthetic_mock)


if __name__ == "__main__":
    unittest.main()
