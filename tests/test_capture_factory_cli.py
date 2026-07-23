import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from capture_pipeline.factory import build_capture_pipeline, import_object
from capture_pipeline.types import RawFrame
from main.run_modular_capture import (
    build_raw_frame_from_metadata,
    parse_override_ev,
    save_pipeline_outputs,
    stage_filename,
)


class TinyAE(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
    def forward(self, x):
        return self.value.expand(x.shape[0])


class FactoryTests(unittest.TestCase):
    def test_default_factory_builds_runnable_identity_pipeline(self):
        pipeline = build_capture_pipeline({"clipping_mode": "hard"}, torch.device("cpu"))
        frame = RawFrame(torch.full((1,1,6,6), 0.2), 0.0, 1.0, "RGGB", {
            "illum_color": [1,1,1], "ccm": torch.eye(3).tolist(),
        }, True)
        result = pipeline(frame, override_ev=0.0)
        self.assertEqual(tuple(result.final_srgb.shape), (1,3,6,6))

    def test_factory_imports_class_and_loads_checkpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
            model = TinyAE(0.75)
            torch.save(model.state_dict(), handle.name)
            config = {
                "ae": {
                    "type": "learned",
                    "class_path": "tests.test_capture_factory_cli:TinyAE",
                    "kwargs": {"value": 0.0},
                    "checkpoint": handle.name,
                }
            }
            pipeline = build_capture_pipeline(config, torch.device("cpu"))
            frame = RawFrame(torch.full((1,1,4,4), 0.2), 0.0, 1.0, "RGGB", {
                "illum_color": [1,1,1], "ccm": torch.eye(3).tolist(),
            }, True)
            result = pipeline(frame)
            self.assertAlmostEqual(result.ae.ev.item(), 0.75, places=6)

    def test_import_object_rejects_bad_path(self):
        with self.assertRaisesRegex(ValueError, "module:object"):
            import_object("badpath")


class CLIHelperTests(unittest.TestCase):
    def test_build_raw_frame_from_metadata(self):
        mosaic = torch.full((1,1,4,4), 100.0)
        metadata = {"black_level": [0,0,0,0], "white_level": 1023, "pattern": "BGGR"}
        frame = build_raw_frame_from_metadata(mosaic, metadata, normalized=False)
        self.assertEqual(frame.cfa_pattern, "BGGR")
        self.assertFalse(frame.is_normalized)

    def test_parse_override_ev_and_stage_names(self):
        self.assertIsNone(parse_override_ev(None, -4, 4))
        self.assertEqual(parse_override_ev(1.5, -4, 4), 1.5)
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_override_ev(5, -4, 4)
        self.assertEqual(stage_filename(0, "raw_input", is_final=False), "00-raw-input.png")
        self.assertEqual(stage_filename(9, "final_srgb", is_final=True), "09-final-srgb.jpg")

    def test_save_pipeline_outputs_writes_stages_and_json(self):
        pipeline = build_capture_pipeline({}, torch.device("cpu"))
        frame = RawFrame(torch.full((1,1,4,4), 0.2), 0.0, 1.0, "RGGB", {
            "illum_color": [1,1,1], "ccm": torch.eye(3).tolist(),
        }, True)
        result = pipeline(frame, override_ev=0.0)
        with tempfile.TemporaryDirectory() as directory:
            files = save_pipeline_outputs(result, Path(directory))
            self.assertTrue((Path(directory) / "analysis.json").is_file())
            self.assertTrue(any(path.endswith("final-srgb.jpg") for path in files))
            report = json.loads((Path(directory) / "analysis.json").read_text())
            self.assertIn("exposure", report["diagnostics"])
            self.assertIn("estimated_ev", report)


if __name__ == "__main__":
    unittest.main()
