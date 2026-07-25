import json
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN = os.path.join(ROOT, "main")
if MAIN not in sys.path:
    sys.path.insert(0, MAIN)

from full_pipeline_analysis import (
    FullPipelineAnalyzer,
    compute_luminance_stats,
    summarize_parameter,
    to_jsonable,
)


class FakePipeline:
    def __init__(self, omit_first_key=None):
        self.calls = []
        self.omit_first_key = omit_first_key
        self.pre_ae = torch.full((1, 3, 4, 4), 0.20)
        self.post_ae = torch.full((1, 3, 4, 4), 0.40)
        self.raw = torch.full((1, 3, 4, 4), 0.10)
        self.denoised = torch.full((1, 3, 4, 4), 0.11)

    def __call__(self, raw, **kwargs):
        self.calls.append((raw, kwargs))
        if not kwargs["photofinishing"]:
            output = {
                "raw": self.raw,
                "denoised_raw": self.denoised,
                "lsrgb": self.pre_ae,
                "srgb": self.post_ae,
                "ev": torch.tensor([1.0]),
                "illum": torch.tensor([0.5, 1.0, 0.8]),
                "ccm": torch.eye(3),
                "cct": 5100.0,
                "tint": -2.0,
                "metadata": {"camera": "fake"},
            }
            if self.omit_first_key is not None:
                output.pop(self.omit_first_key)
            return output

        if not torch.equal(kwargs["lsrgb"], self.post_ae):
            raise AssertionError("post-AE image was not passed unchanged into tone rendering")
        return {
            "raw": self.raw,
            "denoised_raw": self.denoised,
            "lsrgb": self.post_ae,
            "lsrgb_gain": torch.full((1, 3, 4, 4), 0.45),
            "lsrgb_gtm": torch.full((1, 3, 4, 4), 0.55),
            "lsrgb_ltm": torch.full((1, 3, 4, 4), 0.60),
            "processed_lsrgb": torch.full((1, 3, 4, 4), 0.62),
            "gamma": torch.full((1, 3, 4, 4), 0.70),
            "srgb": torch.full((1, 3, 4, 4), 0.72),
            "gain_param": torch.tensor([[[[1.25]]]]),
            "gtm_param": torch.tensor([[1.0, 2.0, 3.0]]),
            "ltm_param": torch.ones((1, 4, 4, 4)),
            "chroma_lut_param": torch.zeros((1, 2, 4, 4)),
            "gamma_param": torch.tensor([[[[0.45]]]]),
            "metadata": {"camera": "fake"},
        }


class FullPipelineAnalyzerTests(unittest.TestCase):
    def test_run_uses_capture_then_tone_pass(self):
        fake = FakePipeline()
        analyzer = FullPipelineAnalyzer(fake)
        result = analyzer.run(
            raw=fake.raw,
            metadata={"camera": "fake"},
            illum=None,
            ccm=None,
            auto_exposure=True,
            denoising_strength=1.0,
            chroma_denoising_strength=0.2,
            luma_denoising_strength=0.3,
            enhancement_strength=0.8,
            sharpening_amount=2.0,
        )

        self.assertEqual(len(fake.calls), 2)
        first_kwargs = fake.calls[0][1]
        second_kwargs = fake.calls[1][1]

        self.assertFalse(first_kwargs["photofinishing"])
        self.assertTrue(first_kwargs["auto_exposure"])
        self.assertEqual(first_kwargs["enhancement_strength"], 0.0)
        self.assertEqual(first_kwargs["sharpening_amount"], 0.0)
        self.assertIsNone(first_kwargs["chroma_denoising_strength"])
        self.assertIsNone(first_kwargs["luma_denoising_strength"])

        self.assertTrue(second_kwargs["photofinishing"])
        self.assertFalse(second_kwargs["auto_exposure"])
        self.assertTrue(second_kwargs["return_intermediate"])
        self.assertEqual(second_kwargs["enhancement_strength"], 0.8)
        self.assertEqual(second_kwargs["sharpening_amount"], 2.0)
        self.assertEqual(second_kwargs["chroma_denoising_strength"], 0.2)
        self.assertEqual(second_kwargs["luma_denoising_strength"], 0.3)
        self.assertTrue(torch.equal(second_kwargs["lsrgb"], fake.post_ae))
        self.assertTrue(torch.equal(second_kwargs["illum"], torch.tensor([0.5, 1.0, 0.8])))
        self.assertTrue(torch.equal(second_kwargs["ccm"], torch.eye(3)))

        expected_stages = [
            "raw", "denoised_raw", "linear_awb", "linear_ae", "gain",
            "gtm", "ltm", "chroma", "gamma", "final",
        ]
        self.assertEqual(list(result["stages"].keys()), expected_stages)
        self.assertEqual(result["report"]["processing_order"][0], "raw")
        self.assertEqual(result["report"]["exposure"]["ev"], [1.0])
        self.assertEqual(
            result["report"]["tone_parameters"]["gtm"]["values"],
            [[1.0, 2.0, 3.0]],
        )
        json.dumps(result["report"])

    def test_missing_required_first_pass_key_fails_fast(self):
        fake = FakePipeline(omit_first_key="ev")
        analyzer = FullPipelineAnalyzer(fake)
        with self.assertRaisesRegex(KeyError, "ev"):
            analyzer.run(raw=fake.raw, metadata={"camera": "fake"})

    def test_compute_luminance_stats_for_constant_gray(self):
        image = torch.full((1, 3, 2, 2), 0.25)
        stats = compute_luminance_stats(image)
        self.assertAlmostEqual(stats["mean"], 0.25, places=7)
        self.assertAlmostEqual(stats["std"], 0.0, places=7)
        self.assertAlmostEqual(stats["p01"], 0.25, places=7)
        self.assertAlmostEqual(stats["p50"], 0.25, places=7)
        self.assertAlmostEqual(stats["p99"], 0.25, places=7)
        self.assertAlmostEqual(stats["low_clip_ratio"], 0.0, places=7)
        self.assertAlmostEqual(stats["high_clip_ratio"], 0.0, places=7)
        self.assertAlmostEqual(stats["robust_dynamic_range_stops"], 0.0, places=7)

    def test_to_jsonable_handles_nested_numeric_values(self):
        value = {
            "tensor": torch.tensor([1.0, 2.0]),
            "array": np.array([[3, 4]], dtype=np.int64),
            "scalar": np.float32(5.5),
            "nested": (torch.tensor(6), {"v": np.int32(7)}),
            "none": None,
        }
        converted = to_jsonable(value)
        self.assertEqual(converted["tensor"], [1.0, 2.0])
        self.assertEqual(converted["array"], [[3, 4]])
        self.assertEqual(converted["scalar"], 5.5)
        self.assertEqual(converted["nested"], [6, {"v": 7}])
        self.assertIsNone(converted["none"])
        json.dumps(converted)

    def test_compute_luminance_stats_rejects_non_finite_values(self):
        image = torch.zeros((1, 3, 2, 2))
        image[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_luminance_stats(image)

    def test_summarize_parameter_compacts_large_tensor(self):
        summary = summarize_parameter(torch.arange(64, dtype=torch.float32).reshape(1, 8, 8))
        self.assertEqual(summary["shape"], [1, 8, 8])
        self.assertEqual(summary["numel"], 64)
        self.assertNotIn("values", summary)

    def test_summarize_parameter_inlines_small_tensor(self):
        summary = summarize_parameter(torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(summary["values"], [[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
