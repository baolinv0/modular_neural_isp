import unittest

import torch
from torch import nn

import cross_camera_tm.phase1_training as legacy_training
from cross_camera_tm.adapters import TargetCameraAdapter
from cross_camera_tm.canonicalization import CanonicalizationConfig, DeviceCanonicalizer
from cross_camera_tm.config import PipelineConfig
from cross_camera_tm.contracts import AlignmentQuality, LinearMetadata
from cross_camera_tm.phase1 import (
    FrozenSamsungTM,
    TeacherMetricProfile,
    TeacherMetricThreshold,
)
from cross_camera_tm.phase1_data import AlignmentEvidence
from cross_camera_tm.phase1_protocol import Phase1Artifact
from cross_camera_tm.phase1_remediation import (
    DEFAULT_ALIGNMENT_POLICY,
    HardenedPhase1Artifact,
    run_hardened_phase1_inference,
)


class IdentityTone(nn.Module):
    def forward(self, image):
        return {"output": image}


def _metadata() -> LinearMetadata:
    return LinearMetadata.from_mapping(
        {
            "sample_id": "iphone-real",
            "device": "iPhone",
            "white_level": 1.0,
            "is_normalized": True,
            "black_level_corrected": True,
            "white_balanced": True,
            "awb_gains_applied": [1.0, 1.0, 1.0],
            "reference_awb_gains": [1.0, 1.0, 1.0],
            "awb_gains_comparable": True,
            "ccm_to_common": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "exposure_time_s": 0.01,
            "iso": 100.0,
            "aperture": 1.8,
            "reference_exposure_product": 0.01 * 100.0 / (1.8**2),
            "hdr_confidence": 1.0,
            "metadata_complete": True,
        }
    )


def _profile() -> TeacherMetricProfile:
    keys = ("clipping_delta", "contrast_error", "highlight_error", "log_luma_mae")
    return TeacherMetricProfile(
        thresholds={key: TeacherMetricThreshold(0.1, 0.2) for key in keys},
        source_count=10,
    )


def _base_artifact(
    data_mode: str,
    *,
    adapter: TargetCameraAdapter | None = None,
    support_min: float = -10.0,
    support_max: float = 10.0,
) -> Phase1Artifact:
    return Phase1Artifact(
        adapter=adapter or TargetCameraAdapter(14, 4),
        feature_mean=torch.zeros(1, 14),
        feature_std=torch.ones(1, 14),
        support_min=torch.full((1, 14), support_min),
        support_max=torch.full((1, 14), support_max),
        samsung_model_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        calibration_manifest_sha256="c" * 64,
        phase1_passed=True,
        validation_report={"passed": True},
        teacher_profile=_profile(),
        data_mode=data_mode,
    )


def _artifact(
    data_mode: str,
    *,
    adapter: TargetCameraAdapter | None = None,
    support_min: float = -10.0,
    support_max: float = 10.0,
    max_support_distance: float = 100.0,
    minimum_parameter_bound_margin: float = 0.0,
    canonicalization: CanonicalizationConfig | None = None,
) -> HardenedPhase1Artifact:
    canonicalization = canonicalization or CanonicalizationConfig()
    return HardenedPhase1Artifact(
        base=_base_artifact(
            data_mode,
            adapter=adapter,
            support_min=support_min,
            support_max=support_max,
        ),
        canonicalization_config=canonicalization,
        canonicalization_sha256=canonicalization.sha256,
        alignment_policy=DEFAULT_ALIGNMENT_POLICY,
        alignment_policy_sha256=DEFAULT_ALIGNMENT_POLICY.sha256,
        max_support_distance=max_support_distance,
        minimum_parameter_bound_margin=minimum_parameter_bound_margin,
        real_phase1_calibration_accepted=data_mode == "real",
        real_source_replay_verified=False,
        real_target_effectiveness_verified=False,
    )


def _real_config(*, phase2_enabled: bool = False, pixel_enabled: bool = False):
    return {
        "schema_version": 2,
        "mode": "real",
        "seed": 17,
        "phase2": {"enabled": phase2_enabled, "minimum_eligible_samples": 50},
        "routing": {"pixel_route_enabled": pixel_enabled},
        "models": {
            "samsung_checkpoint": "samsung.pth",
            "qwen3_vl_checkpoint": None,
            "qwen_image_edit_checkpoint": None,
            "internvl_checkpoint": None,
            "ovis_checkpoint": None,
            "require_real_model": True,
        },
        "canonicalization": {
            "exposure_scale_min": 0.5,
            "exposure_scale_max": 2.0,
            "reliable_dark_threshold": 0.01,
            "highlight_threshold": 0.98,
        },
    }


class Phase1FailClosedRemediationTests(unittest.TestCase):
    def test_real_mode_rejects_phase2_and_pixel_routing(self):
        with self.assertRaisesRegex(ValueError, "PHASE2_NOT_IMPLEMENTED"):
            PipelineConfig.from_mapping(_real_config(phase2_enabled=True))
        with self.assertRaisesRegex(ValueError, "PIXEL_ROUTING_NOT_IMPLEMENTED"):
            PipelineConfig.from_mapping(_real_config(pixel_enabled=True))

    def test_alignment_claim_is_downgraded_when_numeric_evidence_is_weak(self):
        evidence = AlignmentEvidence.from_mapping(
            {
                "quality": "low_frequency",
                "overlap": 0.0,
                "forward_backward_consistency": 0.0,
                "valid_roi_fraction": 0.0,
                "residual_displacement_px": 100.0,
            }
        )
        effective = DEFAULT_ALIGNMENT_POLICY.effective_quality(evidence)
        self.assertIs(effective, AlignmentQuality.SCENE_ONLY)

    def test_synthetic_artifact_cannot_run_real_inference(self):
        image = torch.full((1, 3, 8, 8), 0.3)
        with self.assertRaisesRegex(ValueError, "real Phase 1 artifact"):
            run_hardened_phase1_inference(
                image=image,
                metadata=_metadata(),
                frozen_tm=FrozenSamsungTM(IdentityTone()),
                artifact=_artifact("synthetic"),
                canonicalizer=DeviceCanonicalizer(CanonicalizationConfig()),
            )

    def test_real_calibration_acceptance_does_not_claim_target_effectiveness(self):
        image = torch.full((1, 3, 8, 8), 0.3)
        _, manifest = run_hardened_phase1_inference(
            image=image,
            metadata=_metadata(),
            frozen_tm=FrozenSamsungTM(IdentityTone()),
            artifact=_artifact("real"),
            canonicalizer=DeviceCanonicalizer(CanonicalizationConfig()),
        )
        self.assertTrue(manifest["real_phase1_calibration_accepted"])
        self.assertFalse(manifest["real_source_replay_verified"])
        self.assertFalse(manifest["real_target_effectiveness_verified"])
        self.assertNotIn("real_data_effectiveness_verified", manifest)

    def test_canonicalization_hash_mismatch_is_rejected(self):
        image = torch.full((1, 3, 8, 8), 0.3)
        different = CanonicalizationConfig(highlight_threshold=0.90)
        with self.assertRaisesRegex(ValueError, "canonicalization configuration"):
            run_hardened_phase1_inference(
                image=image,
                metadata=_metadata(),
                frozen_tm=FrozenSamsungTM(IdentityTone()),
                artifact=_artifact("real"),
                canonicalizer=DeviceCanonicalizer(different),
            )

    def test_calibration_support_threshold_is_not_runtime_overridable(self):
        image = torch.full((1, 3, 8, 8), 0.3)
        with self.assertRaisesRegex(ValueError, "BLOCKED_OUTSIDE_CALIBRATION_SUPPORT"):
            run_hardened_phase1_inference(
                image=image,
                metadata=_metadata(),
                frozen_tm=FrozenSamsungTM(IdentityTone()),
                artifact=_artifact(
                    "real",
                    support_min=0.0,
                    support_max=0.0,
                    max_support_distance=0.0,
                ),
                canonicalizer=DeviceCanonicalizer(CanonicalizationConfig()),
            )

    def test_adapter_boundary_saturation_blocks_output(self):
        adapter = TargetCameraAdapter(14, 4)
        with torch.no_grad():
            adapter.head.bias[:12].fill_(100.0)
        image = torch.full((1, 3, 8, 8), 0.3)
        with self.assertRaisesRegex(ValueError, "BLOCKED_OUTSIDE_ADAPTER_SUPPORT"):
            run_hardened_phase1_inference(
                image=image,
                metadata=_metadata(),
                frozen_tm=FrozenSamsungTM(IdentityTone()),
                artifact=_artifact("real", adapter=adapter),
                canonicalizer=DeviceCanonicalizer(CanonicalizationConfig()),
            )

    def test_only_phase1_protocol_exports_train_phase1(self):
        self.assertFalse(hasattr(legacy_training, "train_phase1"))


if __name__ == "__main__":
    unittest.main()
