import unittest

import torch

from cross_camera_tm.contracts import ConfidenceSummary, FailureType, GateStatus
from cross_camera_tm.residuals import (
    ActivationSample,
    DirectionAlignmentGate,
    DynamicROIBuilder,
    PsiFeatureExtractor,
    ResidualComponent,
    ResidualEstimate,
    SourceResidualEstimator,
    SourceResidualProfile,
    assess_phase2_activation,
)


def confidence():
    return ConfidenceSummary(1.0, 0.8, 0.9, 0.7, 0.5, 1.0, 0.82)


class PsiAndProfileTests(unittest.TestCase):
    def test_fixed_feature_schema_and_order(self):
        baseline = torch.linspace(0.02, 0.95, 3 * 8 * 8).view(1, 3, 8, 8)
        reliable = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        highlight = torch.ones_like(reliable)
        face = torch.zeros_like(reliable)
        face[:, :, 2:6, 2:6] = True
        extracted = PsiFeatureExtractor().extract(
            baseline,
            baseline * 0.9,
            reliable,
            highlight,
            confidence(),
            phase1_bound_margin=0.4,
            calibration_support_distance=0.2,
            failure_type=FailureType.FACE_UNDEREXPOSURE,
            scene_code=3,
            face_mask=face,
        )
        self.assertEqual(extracted.names, PsiFeatureExtractor.FEATURE_NAMES)
        self.assertEqual(extracted.values.shape, (1, len(PsiFeatureExtractor.FEATURE_NAMES)))
        self.assertTrue(torch.isfinite(extracted.values).all())
        self.assertEqual(extracted.as_mapping()["failure_code"], 1.0)

    def test_calibrated_nearest_neighbor_support_and_ood_unavailable(self):
        count = len(PsiFeatureExtractor.FEATURE_NAMES)
        features = torch.zeros(8, count)
        features[:, 0] = torch.linspace(0.0, 0.7, 8)
        residuals = torch.stack((torch.linspace(0.05, 0.12, 8), torch.linspace(0.08, 0.15, 8)), dim=1)
        luma_intervals = torch.tensor([[0.08, 0.72]]).repeat(8, 1)
        profile = SourceResidualProfile.calibrate(
            features,
            residuals,
            luma_intervals,
            feature_names=PsiFeatureExtractor.FEATURE_NAMES,
            neighbors=3,
        )
        estimator = SourceResidualEstimator(profile)
        inside = estimator.estimate(features[3:4], PsiFeatureExtractor.FEATURE_NAMES)
        outside = estimator.estimate(torch.full((1, count), 100.0), PsiFeatureExtractor.FEATURE_NAMES)
        self.assertTrue(inside.available)
        self.assertGreater(inside.global_residual.center, 0.0)
        self.assertFalse(outside.available)
        self.assertEqual(outside.reason, "outside_source_support")


class DynamicROIAndDirectionTests(unittest.TestCase):
    def estimate(self, *, available=True):
        return ResidualEstimate(
            available=available,
            reason="ok" if available else "missing_support",
            global_residual=ResidualComponent(0.10, 0.05, 0.16, 0.02),
            face_residual=ResidualComponent(0.15, 0.08, 0.22, 0.03),
            luma_low=0.10,
            luma_high=0.70,
            support_distance=0.2,
            neighbor_count=5,
        )

    def test_global_roi_uses_profile_range_and_reliability(self):
        image = torch.full((1, 3, 8, 8), 0.3)
        image[:, :, :2] = 0.9
        reliable = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        reliable[:, :, :, :1] = False
        highlight = torch.ones_like(reliable)
        highlight[:, :, :2] = False
        result = DynamicROIBuilder(min_coverage=0.05).build(
            image,
            self.estimate(),
            reliable,
            highlight,
            FailureType.GLOBAL_UNDEREXPOSURE,
        )
        self.assertIs(result.status, GateStatus.PASS)
        self.assertEqual(result.mask[:, :, :2].sum().item(), 0)
        self.assertEqual(result.mask[:, :, :, :1].sum().item(), 0)
        self.assertGreater(result.coverage, 0.5)

    def test_face_roi_is_feathered_and_rejects_missing_face(self):
        image = torch.full((1, 3, 16, 16), 0.3)
        valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
        face = torch.zeros_like(valid)
        face[:, :, 4:12, 4:12] = True
        builder = DynamicROIBuilder(min_coverage=0.01)
        result = builder.build(
            image,
            self.estimate(),
            valid,
            valid,
            FailureType.FACE_UNDEREXPOSURE,
            face_mask=face,
        )
        unavailable = builder.build(
            image,
            self.estimate(),
            valid,
            valid,
            FailureType.FACE_UNDEREXPOSURE,
        )
        self.assertIs(result.status, GateStatus.PASS)
        self.assertTrue(((result.mask > 0) & (result.mask < 1)).any())
        self.assertIs(unavailable.status, GateStatus.UNAVAILABLE)

    def test_direction_alignment_requires_interval_sign_distance_ratio_and_non_target(self):
        gate = DirectionAlignmentGate(epsilon_direction=0.1, ratio_bounds=(0.5, 2.0), non_target_max=0.03)
        passed = gate.evaluate(
            self.estimate(), {"global": 0.09, "face": 0.14}, non_target_correction=0.01
        )
        wrong_sign = gate.evaluate(
            self.estimate(), {"global": -0.06, "face": 0.14}, non_target_correction=0.01
        )
        bad_ratio = gate.evaluate(
            self.estimate(), {"global": 0.05, "face": 0.22}, non_target_correction=0.01
        )
        unavailable = gate.evaluate(
            self.estimate(available=False), {"global": 0.09}, non_target_correction=0.0
        )
        self.assertIs(passed.status, GateStatus.PASS)
        self.assertLess(passed.candidate_distance, passed.baseline_distance)
        self.assertIs(wrong_sign.status, GateStatus.FAIL)
        self.assertIn("sign", wrong_sign.reasons)
        self.assertIs(bad_ratio.status, GateStatus.FAIL)
        self.assertIn("ratio", bad_ratio.reasons)
        self.assertIs(unavailable.status, GateStatus.UNAVAILABLE)


class ActivationTests(unittest.TestCase):
    def stable_samples(self):
        return [
            ActivationSample(
                eligible=True,
                severity=0.12 + (index % 3) * 0.01,
                source_p75=0.05,
                scene_group=f"scene-{index % 5}",
                source_supported=True,
                phase1_bound_margin=0.2,
                source_replay_regressed=False,
                failure_type=FailureType.GLOBAL_UNDEREXPOSURE,
            )
            for index in range(60)
        ]

    def test_activation_requires_stable_supported_population(self):
        stable = assess_phase2_activation(self.stable_samples(), bootstrap_samples=300, seed=7)
        insufficient = assess_phase2_activation(self.stable_samples()[:49], bootstrap_samples=100, seed=7)
        regressed = self.stable_samples()
        regressed[0] = ActivationSample(**{**regressed[0].__dict__, "source_replay_regressed": True})
        source_failure = assess_phase2_activation(regressed, bootstrap_samples=100, seed=7)
        self.assertTrue(stable.activated)
        self.assertGreater(stable.bootstrap_lower_bound, 0.0)
        self.assertFalse(insufficient.activated)
        self.assertIn("eligible_count", insufficient.reasons)
        self.assertFalse(source_failure.activated)
        self.assertIn("source_replay", source_failure.reasons)


if __name__ == "__main__":
    unittest.main()
