import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from analyze_full_pipeline import (
    build_parser,
    find_json_for_png,
    resolve_builtin_path,
    save_analysis_outputs,
    select_awb_inputs,
    validate_raw_image,
)


class AnalyzeFullPipelineCliTests(unittest.TestCase):
    def test_find_json_for_png_checks_supported_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image = image_dir / "scene.png"
            image.write_bytes(b"png")
            metadata_dir = root / "data"
            metadata_dir.mkdir()
            expected = metadata_dir / "scene.json"
            expected.write_text("{}", encoding="utf-8")
            self.assertEqual(find_json_for_png(str(image)), str(expected))

    def test_select_awb_inputs_uses_camera_metadata_or_recomputes(self):
        metadata = {
            "cam_illum": [0.5, 1.0, 0.8],
            "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
        illum, ccm = select_awb_inputs(metadata, recompute_awb=False)
        np.testing.assert_allclose(illum, [0.5, 1.0, 0.8])
        np.testing.assert_allclose(ccm, np.eye(3))
        self.assertEqual(select_awb_inputs(metadata, recompute_awb=True), (None, None))

    def test_select_awb_inputs_fails_when_metadata_is_incomplete(self):
        with self.assertRaisesRegex(KeyError, "illuminant"):
            select_awb_inputs({"color_matrix": np.eye(3).tolist()}, recompute_awb=False)
        with self.assertRaisesRegex(KeyError, "CCM"):
            select_awb_inputs({"illum_color": [0.5, 1.0, 0.8]}, recompute_awb=False)

    def test_validate_raw_image_requires_finite_hwc_rgb(self):
        valid = validate_raw_image(np.zeros((4, 5, 3), dtype=np.float32))
        self.assertEqual(valid.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "H x W x 3"):
            validate_raw_image(np.zeros((4, 5), dtype=np.float32))
        invalid = np.zeros((4, 5, 3), dtype=np.float32)
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_raw_image(invalid)

    def test_save_analysis_outputs_writes_stage_manifest_report_and_log(self):
        stages = {
            "raw": np.full((2, 3, 3), 0.1, dtype=np.float32),
            "denoised_raw": np.full((2, 3, 3), 0.2, dtype=np.float32),
            "linear_awb": np.full((2, 3, 3), 0.3, dtype=np.float32),
            "linear_ae": np.full((2, 3, 3), 0.4, dtype=np.float32),
            "gain": np.full((2, 3, 3), 0.5, dtype=np.float32),
            "gtm": np.full((2, 3, 3), 0.6, dtype=np.float32),
            "ltm": np.full((2, 3, 3), 0.7, dtype=np.float32),
            "chroma": np.full((2, 3, 3), 0.75, dtype=np.float32),
            "gamma": np.full((2, 3, 3), 0.8, dtype=np.float32),
            "final": np.full((2, 3, 3), 0.9, dtype=np.float32),
        }
        result = {"stages": stages, "report": {"processing_order": list(stages)}}
        writes = []

        def fake_imwrite(image, output_path, format, quality=95):
            writes.append((np.asarray(image), output_path, format, quality))
            suffix = ".jpg" if format.lower() in {"jpeg", "jpg"} else ".png"
            path = str(Path(output_path).with_suffix(suffix))
            Path(path).write_bytes(b"image")
            return path

        with tempfile.TemporaryDirectory() as tmp:
            manifest = save_analysis_outputs(
                result,
                analysis_dir=tmp,
                basename="scene",
                imwrite_fn=fake_imwrite,
                tensor_to_img_fn=lambda value: value,
                pipeline_log="pipeline trace",
                input_summary={"source": "/data/scene.dng"},
            )
            self.assertEqual(len(writes), 10)
            self.assertEqual(writes[-1][2], "JPEG")
            self.assertTrue(all(item[2] == "PNG-16" for item in writes[:-1]))
            report_path = Path(tmp) / "analysis.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["input"]["source"], "/data/scene.dng")
            self.assertEqual(len(report["output_files"]), 10)
            self.assertEqual(manifest["final"], "09-final.jpg")
            self.assertEqual((Path(tmp) / "pipeline.log").read_text(encoding="utf-8"), "pipeline trace")

    def test_resolve_builtin_path_is_anchored_to_main_directory(self):
        resolved = Path(resolve_builtin_path("../awb_ccm/models/model.pt"))
        self.assertEqual(resolved, (MAIN / "../awb_ccm/models/model.pt").resolve())

    def test_parser_exposes_analysis_controls(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input-file", "x.dng",
            "--photofinishing-model-path", "photo.pth",
            "--denoising-model-path", "denoise.pth",
            "--disable-auto-exposure",
            "--re-compute-awb",
            "--no-downscale-ps",
        ])
        self.assertTrue(args.disable_auto_exposure)
        self.assertTrue(args.re_compute_awb)
        self.assertTrue(args.no_downscale_ps)


if __name__ == "__main__":
    unittest.main()
