import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import cross_camera_tm.phase1_training as phase1_training
from cross_camera_tm.contracts import AlignmentQuality, LinearMetadata
from cross_camera_tm.phase1 import FrozenSamsungTM
from cross_camera_tm.phase1_data import (
    AlignmentEvidence,
    Phase1CalibrationExample,
    Phase1SourceExample,
    build_group_folds,
)
from cross_camera_tm.phase1_protocol import (
    Phase1TrainingConfig,
    load_phase1_artifact,
    run_phase1_inference,
    train_phase1,
)


class TinyTone(nn.Module):
    def forward(self, image):
        return {"output": (image * 0.9).clamp(0.0, 1.0)}


def metadata(sample_id: str, device: str) -> LinearMetadata:
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


def make_data():
    torch.manual_seed(7)
    sources = []
    for index in range(20):
        image = torch.rand(1, 3, 8, 8) * 0.45 + 0.1
        sources.append(
            Phase1SourceExample(
                sample_id=f"source-{index}",
                scene_group=f"source-scene-{index}",
                samsung_image=image,
                samsung_gt=image * 0.9,
                metadata=metadata(f"source-{index}", "Samsung S24"),
            )
        )

    alignment = AlignmentEvidence.from_mapping(
        {
            "quality": "scene_only",
            "overlap": 0.85,
            "forward_backward_consistency": 0.9,
            "valid_roi_fraction": 0.8,
            "residual_displacement_px": 4.0,
        }
    )
    pairs = []
    for index in range(50):
        samsung = torch.rand(1, 3, 8, 8) * 0.4 + 0.1
        iphone = samsung / 1.25
        split = "development" if index < 40 else "locked"
        scene_group = f"dev-scene-{index // 4}" if split == "development" else f"locked-scene-{(index - 40) // 5}"
        pair_id = f"pair-{index}"
        pairs.append(
            Phase1CalibrationExample(
                pair_id=pair_id,
                scene_group=scene_group,
                split=split,
                iphone_image=iphone,
                samsung_image=samsung,
                samsung_gt=samsung * 0.9,
                iphone_metadata=metadata(f"iphone-{index}", "iPhone"),
                samsung_metadata=metadata(f"samsung-{index}", "Samsung S24"),
                alignment=alignment,
            )
        )
    return sources, pairs


class Phase1RealMVPTests(unittest.TestCase):
    def test_alignment_evidence_controls_maximum_legal_loss(self):
        scene = AlignmentEvidence.from_mapping(
            {
                "quality": "scene_only",
                "overlap": 0.3,
                "forward_backward_consistency": 0.4,
                "valid_roi_fraction": 0.2,
                "residual_displacement_px": 12.0,
            }
        )
        lowfreq = AlignmentEvidence.from_mapping(
            {
                "quality": "low_frequency",
                "overlap": 0.9,
                "forward_backward_consistency": 0.95,
                "valid_roi_fraction": 0.85,
                "residual_displacement_px": 1.2,
            }
        )
        self.assertIs(scene.quality, AlignmentQuality.SCENE_ONLY)
        self.assertEqual(scene.enabled_losses, ("tone",))
        self.assertEqual(lowfreq.enabled_losses, ("tone", "roi", "lowfreq"))
        with self.assertRaises(ValueError):
            AlignmentEvidence.from_mapping(
                {
                    "quality": "low_frequency",
                    "overlap": 1.1,
                    "forward_backward_consistency": 0.9,
                    "valid_roi_fraction": 0.8,
                    "residual_displacement_px": 1.0,
                }
            )

    def test_group_folds_never_leak_scene_groups(self):
        _, pairs = make_data()
        folds = build_group_folds(pairs, folds=5)
        self.assertEqual(len(folds), 5)
        validation_groups = []
        for fold in folds:
            self.assertTrue(set(fold.train_groups).isdisjoint(fold.validation_groups))
            validation_groups.extend(fold.validation_groups)
        self.assertEqual(len(validation_groups), len(set(validation_groups)))
        self.assertTrue(all(not group.startswith("locked-") for group in validation_groups))

    def test_train_artifact_and_inference_use_observable_teacher_output(self):
        sources, pairs = make_data()
        frozen_tm = FrozenSamsungTM(TinyTone())
        config = Phase1TrainingConfig(
            solver_steps=6,
            solver_learning_rate=0.08,
            predictor_steps=60,
            predictor_learning_rate=0.04,
            bootstrap_samples=200,
            seed=9,
            data_mode="synthetic",
        )
        folds = build_group_folds(pairs, folds=5)
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "phase1_adapter.pt"
            with patch.object(
                phase1_training,
                "_fit_pair_targets",
                wraps=phase1_training._fit_pair_targets,
            ) as fit_targets:
                result = train_phase1(
                    source_examples=sources,
                    calibration_examples=pairs,
                    frozen_tm=frozen_tm,
                    samsung_model_sha256="a" * 64,
                    source_manifest_sha256="b" * 64,
                    calibration_manifest_sha256="c" * 64,
                    artifact_path=artifact_path,
                    config=config,
                )
            loaded = load_phase1_artifact(artifact_path, expected_model_sha256="a" * 64)
            output, manifest = run_phase1_inference(
                image=pairs[-1].iphone_image,
                metadata=pairs[-1].iphone_metadata,
                frozen_tm=frozen_tm,
                artifact=loaded,
            )

        self.assertEqual(fit_targets.call_count, 6)
        for call, fold in zip(fit_targets.call_args_list[:5], folds):
            solved_groups = {item.example.scene_group for item in call.args[0]}
            self.assertTrue(solved_groups.issubset(set(fold.train_groups)))
            self.assertTrue(solved_groups.isdisjoint(fold.validation_groups))
        final_groups = {item.example.scene_group for item in fit_targets.call_args_list[-1].args[0]}
        self.assertEqual(final_groups, {pair.scene_group for pair in pairs if pair.split == "development"})

        baseline = frozen_tm(pairs[-1].iphone_image)
        teacher = frozen_tm(pairs[-1].samsung_image)
        baseline_error = (baseline - teacher).abs().mean().item()
        adapted_error = (output - teacher).abs().mean().item()
        self.assertTrue(result.report.passed, result.report.reasons)
        self.assertGreaterEqual(result.report.positive_folds, 4)
        self.assertGreater(result.report.locked_median_improvement, 0.0)
        self.assertLess(adapted_error, baseline_error)
        self.assertEqual(manifest["phase1_status"], "pass")
        self.assertEqual(manifest["samsung_model_sha256"], "a" * 64)
        self.assertFalse(manifest["phase2_executed"])
        self.assertFalse(manifest["real_data_effectiveness_verified"])


if __name__ == "__main__":
    unittest.main()
