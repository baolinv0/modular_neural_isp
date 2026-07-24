import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import cross_camera_tm.phase1_remediation as remediation
import cross_camera_tm.phase1_training as legacy_training
from cross_camera_tm.adapters import TargetCameraAdapter
from cross_camera_tm.canary import DeterministicSamsungTMDouble, _metadata, _profile
from cross_camera_tm.canonicalization import CanonicalizationConfig, DeviceCanonicalizer
from cross_camera_tm.contracts import FailureType, canonical_tensor_sha256
from cross_camera_tm.phase1 import FrozenSamsungTM
from cross_camera_tm.phase1_data import PHASE1_FEATURE_NAMES, load_calibration_manifest, load_source_manifest
from cross_camera_tm.phase1_remediation import (
    DEFAULT_ALIGNMENT_POLICY,
    HardenedPhase1Artifact,
    evaluate_hardened_phase1_artifact,
    load_hardened_phase1_artifact,
)
from cross_camera_tm.pipeline import CrossCameraPipeline
from cross_camera_tm.residuals import ResidualComponent, ResidualEstimate
from cross_camera_tm.routing import SupervisionRouter


def _metadata_payload(sample_id: str, device: str) -> dict:
    return {
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


def _write_tensor(path: Path, value: float) -> torch.Tensor:
    tensor = torch.full((1, 3, 4, 4), value, dtype=torch.float32)
    torch.save(tensor, path)
    return tensor


def _write_metadata(path: Path, sample_id: str, device: str) -> None:
    path.write_text(json.dumps(_metadata_payload(sample_id, device)), encoding="utf-8")


def _hardened_payload(*, data_mode: str = "real", max_support: float = 1.0, minimum_margin: float = 0.0):
    adapter = TargetCameraAdapter(len(PHASE1_FEATURE_NAMES), 4)
    canonicalization = CanonicalizationConfig()
    thresholds = {
        name: {"p75": 0.1, "p90": 0.2}
        for name in ("clipping_delta", "contrast_error", "highlight_error", "log_luma_mae")
    }
    return {
        "schema_version": 2,
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "feature_mean": torch.zeros(1, len(PHASE1_FEATURE_NAMES)),
        "feature_std": torch.ones(1, len(PHASE1_FEATURE_NAMES)),
        "support_min": torch.full((1, len(PHASE1_FEATURE_NAMES)), -1.0),
        "support_max": torch.full((1, len(PHASE1_FEATURE_NAMES)), 1.0),
        "adapter": {
            "feature_dim": len(PHASE1_FEATURE_NAMES),
            "hidden_dim": 4,
            "curve_points": adapter.curve_points,
            "max_log_gain": adapter.max_log_gain,
            "max_matrix_delta": adapter.max_matrix_delta,
            "state_dict": adapter.state_dict(),
        },
        "samsung_model_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "calibration_manifest_sha256": "c" * 64,
        "phase1_passed": True,
        "training_config": {},
        "validation_report": {"passed": True},
        "teacher_profile": {"source_count": 10, "thresholds": thresholds},
        "data_mode": data_mode,
        "canonicalization_config": canonicalization.to_dict(),
        "canonicalization_sha256": canonicalization.sha256,
        "alignment_policy": DEFAULT_ALIGNMENT_POLICY.to_dict(),
        "alignment_policy_sha256": DEFAULT_ALIGNMENT_POLICY.sha256,
        "max_support_distance": max_support,
        "minimum_parameter_bound_margin": minimum_margin,
        "real_phase1_calibration_accepted": data_mode == "real",
        "real_source_replay_verified": False,
        "real_target_effectiveness_verified": False,
    }


class SecondReviewFailClosedTests(unittest.TestCase):
    def test_programmatic_pipeline_forbids_non_synthetic_phase2(self):
        pipeline = CrossCameraPipeline(
            canonicalizer=DeviceCanonicalizer(),
            target_adapter=TargetCameraAdapter(8, 4),
            samsung_tm=FrozenSamsungTM(DeterministicSamsungTMDouble()),
            router=SupervisionRouter(pixel_route_enabled=True),
        )
        residual = ResidualEstimate(
            True,
            "real-call-must-be-blocked",
            ResidualComponent(0.10, 0.05, 0.16, 0.02),
            ResidualComponent(0.16, 0.10, 0.22, 0.03),
            0.08,
            0.72,
            0.1,
            5,
        )
        image = torch.full((1, 3, 8, 8), 0.25)
        with self.assertRaisesRegex(RuntimeError, "PHASE2_NOT_IMPLEMENTED"):
            pipeline.run(
                image,
                _metadata(),
                torch.zeros(1, 8),
                residual,
                _profile(),
                phase2_enabled=True,
                phase2_activated=True,
                failure_type=FailureType.GLOBAL_UNDEREXPOSURE,
                face_mask=None,
                config_sha256="a" * 64,
                model_sha256="b" * 64,
                synthetic=False,
                real_model=True,
            )

    def test_parameter_margin_calibration_uses_development_only(self):
        examples = [SimpleNamespace(split="development") for _ in range(40)] + [
            SimpleNamespace(split="locked") for _ in range(10)
        ]
        prepared = [SimpleNamespace(teacher_weight=1.0) for _ in range(40)]
        artifact = SimpleNamespace(
            teacher_profile=SimpleNamespace(),
            adapter=SimpleNamespace(),
            feature_mean=torch.zeros(1),
            feature_std=torch.ones(1),
        )
        with patch.object(remediation.core, "_prepare_pairs", return_value=prepared) as prepare, patch.object(
            remediation.core,
            "_evaluate_items",
            return_value=([], [0.5] * 40),
        ):
            threshold = remediation._calibrated_margin_threshold(
                examples,
                SimpleNamespace(),
                SimpleNamespace(),
                artifact,
            )
        observed = prepare.call_args.args[0]
        self.assertEqual(len(observed), 40)
        self.assertTrue(all(item.split == "development" for item in observed))
        self.assertGreaterEqual(threshold, 0.0)

    def test_hardened_artifact_rejects_nan_and_infinite_thresholds(self):
        for field in ("max_support_distance", "minimum_parameter_bound_margin"):
            for invalid in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, invalid=invalid), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "artifact.pt"
                    payload = _hardened_payload()
                    payload[field] = invalid
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, "thresholds are invalid"):
                        load_hardened_phase1_artifact(path)

    def test_alignment_policy_rejects_non_finite_displacement(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "displacement threshold"):
                    remediation.AlignmentPolicy(lowfreq_displacement_max_px=invalid)

    def test_evaluate_rejects_synthetic_hardened_artifact(self):
        payload = _hardened_payload(data_mode="synthetic")
        base = remediation._base_artifact_from_payload(payload, expected_model_sha256=None)
        artifact = HardenedPhase1Artifact(
            base=base,
            canonicalization_config=CanonicalizationConfig(),
            canonicalization_sha256=CanonicalizationConfig().sha256,
            alignment_policy=DEFAULT_ALIGNMENT_POLICY,
            alignment_policy_sha256=DEFAULT_ALIGNMENT_POLICY.sha256,
            max_support_distance=1.0,
            minimum_parameter_bound_margin=0.0,
            real_phase1_calibration_accepted=False,
            real_source_replay_verified=False,
            real_target_effectiveness_verified=False,
        )
        with self.assertRaisesRegex(ValueError, "requires a real Phase 1 artifact"):
            evaluate_hardened_phase1_artifact(
                calibration_examples=(),
                frozen_tm=SimpleNamespace(),
                artifact=artifact,
                canonicalizer=DeviceCanonicalizer(),
            )

    def test_reload_does_not_restore_legacy_train_phase1(self):
        reloaded = importlib.reload(legacy_training)
        self.assertFalse(hasattr(reloaded, "train_phase1"))


class StrictManifestContractTests(unittest.TestCase):
    def _source_manifest(self, root: Path, *, wrong_device: bool = False, duplicate_content: bool = False) -> Path:
        rows = []
        for index in range(10):
            sample_id = f"source-{index:02d}"
            tensor_path = root / f"{sample_id}.pt"
            gt_path = root / f"{sample_id}-gt.pt"
            metadata_path = root / f"{sample_id}.json"
            value = 0.1 if duplicate_content and index == 9 else 0.1 + index * 0.01
            image = _write_tensor(tensor_path, value)
            gt = _write_tensor(gt_path, value * 0.9)
            device = "iPhone" if wrong_device and index == 0 else "Samsung S24"
            _write_metadata(metadata_path, sample_id, device)
            rows.append(
                {
                    "sample_id": sample_id,
                    "scene_group": f"source-scene-{index:02d}",
                    "samsung_tensor": tensor_path.name,
                    "samsung_gt_tensor": gt_path.name,
                    "metadata": metadata_path.name,
                    "samsung_sha256": canonical_tensor_sha256(image),
                    "gt_sha256": canonical_tensor_sha256(gt),
                }
            )
        path = root / "source_manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "samples": rows}), encoding="utf-8")
        return path

    def _calibration_manifest(self, root: Path, *, duplicate_across_split: bool = False) -> Path:
        rows = []
        cached_iphone = None
        cached_hash = None
        for index in range(50):
            pair_id = f"pair-{index:02d}"
            split = "development" if index < 40 else "locked"
            iphone_id = f"iphone-{index:02d}"
            samsung_id = f"samsung-{index:02d}"
            iphone_path = root / f"{iphone_id}.pt"
            samsung_path = root / f"{samsung_id}.pt"
            gt_path = root / f"{samsung_id}-gt.pt"
            iphone_meta = root / f"{iphone_id}.json"
            samsung_meta = root / f"{samsung_id}.json"
            if duplicate_across_split and index == 40:
                iphone = cached_iphone.clone()
                torch.save(iphone, iphone_path)
                iphone_hash = cached_hash
            else:
                iphone = _write_tensor(iphone_path, 0.1 + index * 0.002)
                iphone_hash = canonical_tensor_sha256(iphone)
                if index == 0:
                    cached_iphone, cached_hash = iphone, iphone_hash
            samsung = _write_tensor(samsung_path, 0.2 + index * 0.002)
            gt = _write_tensor(gt_path, 0.18 + index * 0.002)
            _write_metadata(iphone_meta, iphone_id, "iPhone")
            _write_metadata(samsung_meta, samsung_id, "Samsung S24")
            rows.append(
                {
                    "pair_id": pair_id,
                    "scene_group": f"dev-scene-{index // 4}" if split == "development" else f"locked-scene-{(index - 40) // 5}",
                    "split": split,
                    "iphone_tensor": iphone_path.name,
                    "samsung_tensor": samsung_path.name,
                    "samsung_gt_tensor": gt_path.name,
                    "iphone_metadata": iphone_meta.name,
                    "samsung_metadata": samsung_meta.name,
                    "alignment": {
                        "quality": "scene_only",
                        "overlap": 0.5,
                        "forward_backward_consistency": 0.5,
                        "valid_roi_fraction": 0.5,
                        "residual_displacement_px": 4.0,
                    },
                    "roi_mask": None,
                    "alignment_mask": None,
                    "iphone_sha256": iphone_hash,
                    "samsung_sha256": canonical_tensor_sha256(samsung),
                    "gt_sha256": canonical_tensor_sha256(gt),
                }
            )
        path = root / "calibration_manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "pairs": rows}), encoding="utf-8")
        return path

    def test_source_manifest_rejects_wrong_device_role(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Samsung"):
                load_source_manifest(self._source_manifest(Path(directory), wrong_device=True))

    def test_source_manifest_rejects_duplicate_content(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_source_manifest(self._source_manifest(Path(directory), duplicate_content=True))

    def test_calibration_manifest_rejects_content_overlap_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "development and locked"):
                load_calibration_manifest(
                    self._calibration_manifest(Path(directory), duplicate_across_split=True)
                )


if __name__ == "__main__":
    unittest.main()
