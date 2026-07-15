import unittest

import torch

from capture_pipeline.demosaic import BilinearBayerDemosaicer
from capture_pipeline.types import RawFrame


PATTERN_POSITIONS = {
    "RGGB": (("R", "G"), ("G", "B")),
    "BGGR": (("B", "G"), ("G", "R")),
    "GRBG": (("G", "R"), ("B", "G")),
    "GBRG": (("G", "B"), ("R", "G")),
}
VALUES = {"R": 0.8, "G": 0.4, "B": 0.2}


def constant_color_frame(pattern, requires_grad=False):
    mosaic = torch.empty(1, 1, 8, 8)
    layout = PATTERN_POSITIONS[pattern]
    for row in range(2):
        for col in range(2):
            mosaic[:, :, row::2, col::2] = VALUES[layout[row][col]]
    mosaic.requires_grad_(requires_grad)
    return RawFrame(mosaic, 0.0, 1.0, pattern, {}, True)


class DemosaicTests(unittest.TestCase):
    def test_reconstructs_constant_color_for_all_patterns(self):
        demosaic = BilinearBayerDemosaicer()
        expected = torch.tensor(VALUES["R"]), torch.tensor(VALUES["G"]), torch.tensor(VALUES["B"])
        for pattern in PATTERN_POSITIONS:
            rgb = demosaic(constant_color_frame(pattern))
            self.assertEqual(tuple(rgb.shape), (1, 3, 8, 8))
            self.assertTrue(torch.isfinite(rgb).all())
            for channel, target in enumerate(expected):
                self.assertTrue(torch.allclose(rgb[:, channel], torch.full_like(rgb[:, channel], target), atol=1e-5))

    def test_preserves_known_samples(self):
        pattern = "RGGB"
        frame = constant_color_frame(pattern)
        rgb = BilinearBayerDemosaicer()(frame)
        self.assertTrue(torch.allclose(rgb[:, 0:1, 0::2, 0::2], frame.mosaic[:, :, 0::2, 0::2]))
        self.assertTrue(torch.allclose(rgb[:, 2:3, 1::2, 1::2], frame.mosaic[:, :, 1::2, 1::2]))

    def test_gradient_reaches_every_mosaic_sample(self):
        frame = constant_color_frame("GBRG", requires_grad=True)
        rgb = BilinearBayerDemosaicer()(frame)
        rgb.sum().backward()
        self.assertIsNotNone(frame.mosaic.grad)
        self.assertTrue(torch.isfinite(frame.mosaic.grad).all())
        self.assertTrue((frame.mosaic.grad.abs() > 0).all())


if __name__ == "__main__":
    unittest.main()
