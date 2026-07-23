import unittest

import torch

from capture_pipeline.types import RawFrame, AEOutput, AWBOutput, ToneMapOutput, CapturePipelineOutput


class RawFrameTests(unittest.TestCase):
    def test_accepts_supported_cfa_and_normalized_data(self):
        for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
            frame = RawFrame(
                mosaic=torch.full((2, 1, 4, 6), 0.25),
                black_level=torch.zeros(2, 1),
                white_level=torch.ones(2, 1),
                cfa_pattern=pattern,
                metadata={"id": pattern},
                is_normalized=True,
            )
            self.assertEqual(frame.cfa_pattern, pattern)
            self.assertEqual(frame.batch_size, 2)

    def test_normalizes_sensor_domain_per_cfa_channel(self):
        mosaic = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]])
        frame = RawFrame(
            mosaic=mosaic,
            black_level=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            white_level=torch.tensor([[110.0]]),
            cfa_pattern="RGGB",
            metadata={"scene": "x"},
            is_normalized=False,
        )
        normalized = frame.normalized()
        self.assertTrue(torch.allclose(normalized.mosaic, torch.zeros_like(mosaic)))
        self.assertTrue(normalized.is_normalized)
        self.assertEqual(normalized.metadata, {"scene": "x"})
        self.assertIsNot(normalized.metadata, frame.metadata)

    def test_rejects_invalid_shape_pattern_and_range(self):
        with self.assertRaisesRegex(ValueError, r"\[B, 1, H, W\]"):
            RawFrame(torch.zeros(1, 3, 4, 4), torch.zeros(1, 1), torch.ones(1, 1), "RGGB", {}, True)
        with self.assertRaisesRegex(ValueError, "Unsupported CFA"):
            RawFrame(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1), torch.ones(1, 1), "RGBW", {}, True)
        with self.assertRaisesRegex(ValueError, "normalized mosaic"):
            RawFrame(torch.full((1, 1, 4, 4), 1.2), torch.zeros(1, 1), torch.ones(1, 1), "RGGB", {}, True)

    def test_rejects_nonfinite_and_invalid_levels(self):
        bad = torch.zeros(1, 1, 4, 4)
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            RawFrame(bad, torch.zeros(1, 1), torch.ones(1, 1), "RGGB", {}, True)
        with self.assertRaisesRegex(ValueError, "white_level"):
            RawFrame(torch.zeros(1, 1, 4, 4), torch.ones(1, 1), torch.ones(1, 1), "RGGB", {}, False)

    def test_output_dataclasses_hold_expected_values(self):
        ae = AEOutput(torch.zeros(1), torch.ones(1), {})
        awb = AWBOutput(torch.ones(1, 3), torch.eye(3).unsqueeze(0), torch.ones(1), {})
        tone = ToneMapOutput(torch.zeros(1, 3, 2, 2), None, None, None, {})
        out = CapturePipelineOutput(torch.zeros(1, 3, 2, 2), {}, ae, awb, tone, {})
        self.assertEqual(tuple(out.final_srgb.shape), (1, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
