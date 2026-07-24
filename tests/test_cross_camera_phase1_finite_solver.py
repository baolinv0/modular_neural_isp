import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import cross_camera_tm.phase1_protocol as protocol
from cross_camera_tm import phase1_training as core
from cross_camera_tm.adapters import PairTransformParameters, TargetCameraAdapter
from cross_camera_tm.canonicalization import DeviceCanonicalizer
from cross_camera_tm.contracts import AlignmentQuality, LinearMetadata
from cross_camera_tm.phase1 import FrozenSamsungTM
from cross_camera_tm.phase1_data import (
    AlignmentEvidence,
    Phase1CalibrationExample,
    Phase1SourceExample,
)


class TinyTone(nn.Module):
    def forward(self, image):
        return {"output": (image * 0.9).clamp(0.0, 1.0)}


def _metadata(sample_id: str, device: str) -> LinearMetadata:
    return LinearMetadata.from_mapping(
        {
            "sample_id": sample_id,
            "device": device,
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


def _make_protocol_data():
    torch.manual_seed(19)
    sources = []
    for index in range(10):
        image = torch.rand(1, 3, 4, 4) * 0.35 + 0.1
        sources.append(
            Phase1SourceExample(
                sample_id=f"source-{index}",
                scene_group=f"source-scene-{index}",
                samsung_image=image,
                samsung_gt=image * 0.9,
                metadata=_metadata(f"source-{index}", "Samsung S24"),
            )
        )

    alignment = AlignmentEvidence.from_mapping(
        {
            "quality": "scene_only",
            "overlap": 0.5,
            "forward_backward_consistency": 0.5,
            "valid_roi_fraction": 0.5,
            "residual_displacement_px": 4.0,
        }
    )
    pairs = []
    for index in range(50):
        samsung = torch.rand(1, 3, 4, 4) * 0.3 + 0.1
        iphone = samsung / 1.2
        split = "development" if index < 40 else "locked"
        scene_group = (
            f"dev-scene-{index // 4}"
            if split == "development"
            else f"locked-scene-{(index - 40) // 5}"
        )
        pairs.append(
            Phase1CalibrationExample(
                pair_id=f"pair-{index}",
                scene_group=scene_group,
                split=split,
                iphone_image=iphone,
                samsung_image=samsung,
                samsung_gt=samsung * 0.9,
                iphone_metadata=_metadata(f"iphone-{index}", "iPhone"),
                samsung_metadata=_metadata(f"samsung-{index}", "Samsung S24"),
                alignment=alignment,
            )
        )
    return sources, pairs


class ObservablePairSolverTests(unittest.TestCase):
    def test_production_solver_is_finite_bounded_monotonic_and_reduces_error(self):
        torch.manual_seed(5)
        iphone = torch.rand(1, 3, 8, 8) * 0.35 + 0.1
        samsung = (iphone * 1.25).clamp(0.0, 1.0)
        reliable = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        frozen_tm = FrozenSamsungTM(TinyTone())
        teacher = frozen_tm(samsung)
        solver = core.ObservablePairSolver(curve_points=6)

        initialized = solver.initialize(iphone, samsung, reliable)
        initial = PairTransformParameters.identity(
            batch=1,
            device=iphone.device,
            dtype=iphone.dtype,
            curve_points=6,
        )
        initial_student = frozen_tm(
            TargetCameraAdapter.apply_explicit(
                iphone,
                initial,
                confidence=torch.ones(1),
            )
        )
        initial_error = core.observable_pair_error(
            initial_student,
            teacher,
            alignment=AlignmentQuality.SCENE_ONLY,
            roi_mask=None,
            alignment_mask=None,
        ).item()

        refined = solver.refine(
            initial=initial,
            iphone=iphone,
            teacher=teacher,
            frozen_tm=frozen_tm,
            alignment=AlignmentQuality.SCENE_ONLY,
            roi_mask=None,
            alignment_mask=None,
            teacher_weight=1.0,
            steps=30,
            learning_rate=0.05,
        )
        refined_student = frozen_tm(
            TargetCameraAdapter.apply_explicit(
                iphone,
                refined,
                confidence=torch.ones(1),
            )
        )
        refined_error = core.observable_pair_error(
            refined_student,
            teacher,
            alignment=AlignmentQuality.SCENE_ONLY,
            roi_mask=None,
            alignment_mask=None,
        ).item()

        for parameters in (initialized, refined):
            self.assertTrue(torch.isfinite(parameters.gains).all())
            self.assertTrue(torch.isfinite(parameters.matrix).all())
            self.assertTrue(torch.isfinite(parameters.curve_y).all())
            self.assertTrue(torch.all(parameters.curve_y[:, 1:] >= parameters.curve_y[:, :-1]))
        self.assertTrue(torch.all(refined.gains >= 0.70))
        self.assertTrue(torch.all(refined.gains <= 1.40))
        identity = torch.eye(3).unsqueeze(0)
        self.assertLessEqual((refined.matrix - identity).abs().max().item(), 0.100001)
        self.assertLess(refined_error, initial_error)


class NonFiniteEvidenceTests(unittest.TestCase):
    def test_numeric_evidence_guards_reject_nan_and_infinity(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "NON_FINITE_PHASE1_EVIDENCE"):
                    protocol._require_finite_values([0.1, invalid], "validation")

    def test_non_finite_pair_target_is_rejected_before_predictor_fit(self):
        sources, pairs = _make_protocol_data()
        bad = PairTransformParameters(
            gains=torch.full((1, 3), float("nan")),
            matrix=torch.eye(3).unsqueeze(0),
            curve_y=torch.linspace(0.0, 1.0, 6).unsqueeze(0),
        )

        def invalid_targets(items, _solver, _frozen_tm, _config):
            return {items[0].example.pair_id: bad}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            protocol.core,
            "_fit_pair_targets",
            side_effect=invalid_targets,
        ), patch.object(protocol, "_fit_ridge_predictor") as fit_predictor:
            with self.assertRaisesRegex(ValueError, "NON_FINITE_PHASE1_EVIDENCE"):
                protocol.train_phase1(
                    source_examples=sources,
                    calibration_examples=pairs,
                    frozen_tm=FrozenSamsungTM(TinyTone()),
                    samsung_model_sha256="a" * 64,
                    source_manifest_sha256="b" * 64,
                    calibration_manifest_sha256="c" * 64,
                    artifact_path=Path(directory) / "phase1.pt",
                    config=protocol.Phase1TrainingConfig(
                        solver_steps=1,
                        bootstrap_samples=50,
                        data_mode="synthetic",
                    ),
                    canonicalizer=DeviceCanonicalizer(),
                )
        fit_predictor.assert_not_called()

    def test_all_finite_helpers_accept_normal_values(self):
        protocol._require_finite_values([0.1, 0.2], "normal")
        parameters = PairTransformParameters(
            gains=torch.ones(1, 3),
            matrix=torch.eye(3).unsqueeze(0),
            curve_y=torch.linspace(0.0, 1.0, 6).unsqueeze(0),
        )
        protocol._require_finite_parameter_targets({"pair": parameters}, "normal")
        self.assertTrue(math.isfinite(float(parameters.gains.mean().item())))


if __name__ == "__main__":
    unittest.main()
