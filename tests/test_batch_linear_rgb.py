import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.batch_linear_rgb_tm import (  # noqa: E402
    build_parser,
    discover_input_files,
    load_linear_rgb16,
    render_linear_rgb,
    save_rendered_image,
)


class _HalfPhotofinishing(torch.nn.Module):
    def forward(self, x, **kwargs):
        return x * 0.5


class _NearestUpsampler(torch.nn.Module):
    def forward(self, high_res_input, low_res_input, low_res_target):
        del low_res_input
        return torch.nn.functional.interpolate(
            low_res_target,
            size=high_res_input.shape[-2:],
            mode="nearest",
        )


class _DoubleEnhancement(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x * 2.0, 0.0, 1.0)


class BatchLinearRgbTmTests(unittest.TestCase):
    def test_parser_defaults_to_both_output_formats(self):
        args = build_parser().parse_args(
            [
                "--input-dir",
                "inputs",
                "--output-dir",
                "outputs",
                "--photofinishing-model-path",
                "tm.pth",
                "--enhancement-model-path",
                "enhance.pth",
            ]
        )

        self.assertEqual(args.output_format, "both")

    def test_parser_requires_enhancement_model(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--input-dir",
                        "inputs",
                        "--output-dir",
                        "outputs",
                        "--photofinishing-model-path",
                        "tm.pth",
                    ]
                )

    def test_load_linear_rgb16_returns_normalized_nchw_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.png"
            rgb = np.zeros((4, 5, 3), dtype=np.uint16)
            rgb[0, 0] = [65535, 32768, 0]
            self.assertTrue(cv2.imwrite(str(path), rgb[..., ::-1]))

            tensor = load_linear_rgb16(path, device=torch.device("cpu"))

            self.assertEqual(tuple(tensor.shape), (1, 3, 4, 5))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0, places=6)
            self.assertAlmostEqual(float(tensor[0, 1, 0, 0]), 32768 / 65535, places=6)
            self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), 0.0, places=6)

    def test_load_linear_rgb16_rejects_uint8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.png"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((4, 5, 3), dtype=np.uint8)))

            with self.assertRaisesRegex(ValueError, "16-bit unsigned"):
                load_linear_rgb16(path, device=torch.device("cpu"))

    def test_discover_input_files_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            for relative in ["b.tiff", "a.png", "nested/c.tif"]:
                (root / relative).write_bytes(b"placeholder")
            (root / "ignore.jpg").write_bytes(b"placeholder")

            files = discover_input_files(root, recursive=True)

            self.assertEqual(
                [p.relative_to(root).as_posix() for p in files],
                ["a.png", "b.tiff", "nested/c.tif"],
            )

    def test_render_linear_rgb_can_bypass_downsampling(self):
        image = torch.ones((1, 3, 8, 10), dtype=torch.float32)

        tm_output, enhanced_output = render_linear_rgb(
            image,
            photofinishing_model=_HalfPhotofinishing(),
            upsampler=_NearestUpsampler(),
            downsample_photofinishing=False,
        )

        self.assertEqual(tuple(tm_output.shape), tuple(image.shape))
        self.assertTrue(torch.allclose(tm_output, image * 0.5))
        self.assertIsNone(enhanced_output)

    def test_render_linear_rgb_upsamples_low_resolution_render(self):
        image = torch.ones((1, 3, 64, 80), dtype=torch.float32)

        tm_output, enhanced_output = render_linear_rgb(
            image,
            photofinishing_model=_HalfPhotofinishing(),
            upsampler=_NearestUpsampler(),
            downsample_photofinishing=True,
        )

        self.assertEqual(tuple(tm_output.shape), tuple(image.shape))
        self.assertTrue(torch.allclose(tm_output, image * 0.5))
        self.assertIsNone(enhanced_output)

    def test_render_linear_rgb_returns_enhanced_output_when_model_is_provided(self):
        image = torch.ones((1, 3, 64, 80), dtype=torch.float32)

        tm_output, enhanced_output = render_linear_rgb(
            image,
            photofinishing_model=_HalfPhotofinishing(),
            upsampler=_NearestUpsampler(),
            enhancement_model=_DoubleEnhancement(),
            enhancement_strength=1.0,
            downsample_photofinishing=True,
        )

        self.assertTrue(torch.allclose(tm_output, image * 0.5))
        self.assertIsNotNone(enhanced_output)
        self.assertTrue(torch.allclose(enhanced_output, image))

    def test_save_rendered_image_writes_uint16_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.png"
            image = torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]], dtype=torch.float32)

            save_rendered_image(image, path, output_format="png16", jpeg_quality=95)

            saved = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.dtype, np.uint16)
            rgb = saved[..., ::-1]
            self.assertEqual(int(rgb[0, 0, 0]), 0)
            self.assertIn(int(rgb[0, 0, 1]), (32767, 32768))
            self.assertEqual(int(rgb[0, 0, 2]), 65535)


if __name__ == "__main__":
    unittest.main()
