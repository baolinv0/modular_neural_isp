import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from cross_camera_tm.adapters import TargetCameraAdapter
from cross_camera_tm.canary import DeterministicSamsungTMDouble, _metadata, _profile, run_synthetic_canary
from cross_camera_tm.canonicalization import DeviceCanonicalizer
from cross_camera_tm.cli import main
from cross_camera_tm.config import PipelineConfig
from cross_camera_tm.contracts import FailureType
from cross_camera_tm.phase1 import FrozenSamsungTM
from cross_camera_tm.pipeline import CrossCameraPipeline
from cross_camera_tm.policy import QwenImageEditLocalAdapter
from cross_camera_tm.residuals import ResidualComponent, ResidualEstimate
from cross_camera_tm.routing import Route, SupervisionRouter


def valid_config():
    return {
        "schema_version": 2,
        "mode": "synthetic_canary",
        "seed": 17,
        "phase2": {"enabled": True, "minimum_eligible_samples": 50},
        "routing": {"pixel_route_enabled": False},
        "models": {
            "samsung_checkpoint": None,
            "qwen3_vl_checkpoint": None,
            "qwen_image_edit_checkpoint": None,
            "internvl_checkpoint": None,
            "ovis_checkpoint": None,
            "require_real_model": False,
        },
        "canonicalization": {
            "exposure_scale_min": 0.5,
            "exposure_scale_max": 2.0,
            "reliable_dark_threshold": 0.01,
            "highlight_threshold": 0.98,
        },
    }


class ConfigAndPipelineTests(unittest.TestCase):
    def test_strict_config_rejects_unknown_nested_and_inconsistent_real_mode(self):
        with self.assertRaises(ValueError):
            PipelineConfig.from_mapping({**valid_config(), "unknown": 1})
        nested = valid_config()
        nested["phase2"] = {**nested["phase2"], "surprise": True}
        with self.assertRaises(ValueError):
            PipelineConfig.from_mapping(nested)
        ambiguous = valid_config()
        ambiguous["phase2"] = {"enabled": "false", "minimum_eligible_samples": 50}
        with self.assertRaises(ValueError):
            PipelineConfig.from_mapping(ambiguous)
        real = valid_config()
        real["mode"] = "real"
        with self.assertRaises(ValueError):
            PipelineConfig.from_mapping(real)

    def test_synthetic_pipeline_order_and_phase2_disabled_fail_closed(self):
        config = PipelineConfig.from_mapping(valid_config())
        enabled = run_synthetic_canary(config=config, output_dir=None)
        disabled_mapping = valid_config()
        disabled_mapping["phase2"] = {"enabled": False, "minimum_eligible_samples": 50}
        disabled = run_synthetic_canary(
            config=PipelineConfig.from_mapping(disabled_mapping), output_dir=None
        )
        self.assertEqual(
            enabled["trace"],
            [
                "canonicalization",
                "target_camera_adapter",
                "frozen_samsung_tm",
                "phase2_activation",
                "L1_teacher",
                "full_certification",
                "supervision_routing",
                "manifest",
            ],
        )
        self.assertEqual(enabled["route"], "parameter")
        self.assertEqual(disabled["route"], "diagnostic")
        self.assertEqual(disabled["reason"], "phase2_disabled")

    def test_canary_is_deterministic_and_honestly_labeled(self):
        config = PipelineConfig.from_mapping(valid_config())
        first = run_synthetic_canary(config=config, output_dir=None)
        second = run_synthetic_canary(config=config, output_dir=None)
        self.assertEqual(first, second)
        self.assertTrue(first["synthetic"])
        self.assertFalse(first["real_model"])
        self.assertFalse(first["real_data_effectiveness_verified"])

    def test_l3_fallback_is_projected_and_recertified_before_routing(self):
        base = torch.linspace(0.12, 0.52, 32 * 32).view(1, 1, 32, 32)
        image = torch.cat((base * 0.98, base, base * 1.02), dim=1) * 65535.0
        editor = QwenImageEditLocalAdapter.deterministic_mock(
            lambda value, prompt: (value * torch.exp(torch.tensor(0.18))).clamp(0.0, 1.0)
        )
        pipeline = CrossCameraPipeline(
            canonicalizer=DeviceCanonicalizer(),
            target_adapter=TargetCameraAdapter(8, 4),
            samsung_tm=FrozenSamsungTM(DeterministicSamsungTMDouble()),
            router=SupervisionRouter(pixel_route_enabled=False),
            l3_editor=editor,
        )
        residual = ResidualEstimate(
            True,
            "ok",
            ResidualComponent(0.10, 0.05, 0.22, 0.02),
            ResidualComponent(0.16, 0.10, 0.22, 0.03),
            0.08,
            0.72,
            0.1,
            5,
        )
        result = pipeline.run(
            image,
            _metadata(),
            torch.zeros(1, 8),
            residual,
            replace(_profile(), issue_lift_min=0.15, issue_lift_max=0.22),
            phase2_enabled=True,
            phase2_activated=True,
            failure_type=FailureType.GLOBAL_UNDEREXPOSURE,
            face_mask=None,
            config_sha256="a" * 64,
            model_sha256="b" * 64,
            synthetic=True,
            real_model=False,
        )
        self.assertIs(result.routing.route, Route.PREFERENCE)
        self.assertIn("L3_raw_proposal", result.trace)
        self.assertIn("tm_space_projection", result.trace)
        self.assertEqual(result.trace.count("full_certification"), 2)
        self.assertTrue(result.manifest.projected)
        self.assertFalse(result.manifest.raw_generated)


class CLITests(unittest.TestCase):
    def test_cli_writes_valid_report_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(valid_config()), encoding="utf-8")
            output = root / "run"
            exit_code = main(
                [
                    "synthetic-canary",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output),
                ]
            )
            report = json.loads((output / "canary_report.json").read_text(encoding="utf-8"))
            manifest_lines = (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["route"], "parameter")
        self.assertEqual(len(manifest_lines), 1)
        self.assertTrue(json.loads(manifest_lines[0])["synthetic"])

    def test_real_run_with_unavailable_samsung_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_checkpoint = root / "missing-samsung.pth"
            config_payload = valid_config()
            config_payload["mode"] = "real"
            config_payload["models"] = {
                **config_payload["models"],
                "samsung_checkpoint": str(missing_checkpoint),
                "require_real_model": True,
            }
            config_path = root / "real.yaml"
            config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "real-run",
                        "--config",
                        str(config_path),
                        "--adapter-checkpoint",
                        str(root / "phase1.pt"),
                        "--input",
                        str(root / "input.pt"),
                        "--metadata",
                        str(root / "metadata.json"),
                        "--output-dir",
                        str(root / "output"),
                    ]
                )
        self.assertEqual(exit_code, 2)
        self.assertIn("Samsung checkpoint is unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
