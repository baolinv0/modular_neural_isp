"""Command-line stage analysis for the Modular Neural ISP pipeline.

This entry point preserves the repository's existing algorithms. It decodes the
input using the same rules as ``demo.py``, invokes :class:`FullPipelineAnalyzer`,
and exports every observable stage plus a compact JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


_STAGE_FILES = {
    "raw": ("00-raw", "PNG-16"),
    "denoised_raw": ("01-denoised-raw", "PNG-16"),
    "linear_awb": ("02-linear-awb", "PNG-16"),
    "linear_ae": ("03-linear-ae", "PNG-16"),
    "gain": ("04-gain", "PNG-16"),
    "gtm": ("05-gtm", "PNG-16"),
    "ltm": ("06-ltm", "PNG-16"),
    "chroma": ("07-chroma", "PNG-16"),
    "gamma": ("08-gamma", "PNG-16"),
    "final": ("09-final", "JPEG"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and inspect the complete RAW -> AWB/CCM -> AE -> TM pipeline."
    )
    parser.add_argument("--input-file", required=True, help="DNG, PNG, JPEG, or embedded-RAW JPEG input.")
    parser.add_argument("--metadata-json", default=None, help="Optional metadata JSON for a 16-bit PNG input.")
    parser.add_argument("--output-dir", default=".", help="Parent directory for the analysis folder.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")

    parser.add_argument("--photofinishing-model-path", required=True)
    parser.add_argument("--photofinishing-model-config-path", default=None)
    parser.add_argument("--denoising-model-path", required=True)
    parser.add_argument("--denoising-model-config-path", default=None)
    parser.add_argument("--enhancement-model-path", default=None)
    parser.add_argument("--enhancement-model-config-path", default=None)

    parser.add_argument("--re-compute-awb", action="store_true", help="Ignore metadata AWB and estimate illuminant.")
    parser.add_argument("--pref-awb", action="store_true", help="Apply preference AWB mapping after estimation.")
    parser.add_argument("--use-cc-awb", action="store_true", help="Force the cross-camera AWB estimator.")
    parser.add_argument("--adjust-ccm", action="store_true", help="Recompute CCM from calibration metadata.")
    parser.add_argument("--target-cct", type=float, default=None)
    parser.add_argument("--target-tint", type=float, default=None)

    parser.add_argument("--disable-auto-exposure", action="store_true")
    parser.add_argument("--ev-value", type=float, default=0.0, help="Manual EV applied before AWB/AE.")
    parser.add_argument("--denoising-strength", type=float, default=1.0)
    parser.add_argument("--luma-denoising-strength", type=float, default=0.0)
    parser.add_argument("--chroma-denoising-strength", type=float, default=0.0)
    parser.add_argument("--enhancement-strength", type=float, default=1.0)

    parser.add_argument("--post-process-ltm", action="store_true")
    parser.add_argument("--solver-iterations", type=int, default=None)
    parser.add_argument("--no-downscale-ps", action="store_true")

    parser.add_argument("--contrast-amount", type=float, default=0.0)
    parser.add_argument("--vibrance-amount", type=float, default=0.0)
    parser.add_argument("--saturation-amount", type=float, default=0.0)
    parser.add_argument("--highlight-amount", type=float, default=0.0)
    parser.add_argument("--shadow-amount", type=float, default=0.0)
    parser.add_argument("--sharpening-amount", type=float, default=0.0)
    return parser


def resolve_builtin_path(path: str) -> str:
    """Resolve repository-owned relative model paths from the main directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(__file__).resolve().parent / candidate).resolve())


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_json_for_png(input_file: str) -> str:
    """Locate PNG metadata using the same search order as ``demo.py``."""
    path = Path(input_file)
    if path.suffix.lower() != ".png":
        raise ValueError("Metadata lookup is only valid for PNG inputs.")
    candidates = (
        path.with_suffix(".json"),
        path.parent / "data" / f"{path.stem}.json",
        path.parent.parent / "data" / f"{path.stem}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f'Metadata JSON for "{input_file}" was not found. Checked:\n{checked}')


def select_awb_inputs(metadata: Mapping[str, Any], recompute_awb: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Select metadata AWB/CCM values, or signal that both must be estimated."""
    if recompute_awb:
        return None, None
    if metadata is None:
        raise ValueError("Metadata is required when AWB recomputation is disabled.")

    illum_key = next((key for key in ("cam_illum", "illum_color") if key in metadata), None)
    if illum_key is None:
        raise KeyError("Metadata does not contain a camera illuminant ('cam_illum' or 'illum_color').")
    ccm_key = next((key for key in ("color_matrix", "ccm") if key in metadata), None)
    if ccm_key is None:
        raise KeyError("Metadata does not contain a CCM ('color_matrix' or 'ccm').")

    illum = np.asarray(metadata[illum_key], dtype=np.float32).reshape(-1)
    ccm = np.asarray(metadata[ccm_key], dtype=np.float32).reshape(3, 3)
    if illum.size != 3 or not np.isfinite(illum).all():
        raise ValueError(f"Invalid illuminant shape/value from '{illum_key}'.")
    if not np.isfinite(ccm).all():
        raise ValueError(f"Invalid CCM values from '{ccm_key}'.")
    return illum, ccm


def validate_raw_image(raw_img: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_img, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"Expected normalized H x W x 3 raw RGB, got shape {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("Raw input contains non-finite values.")
    return np.clip(raw, 0.0, 1.0)


def _image_to_numpy(value: Any, tensor_to_img_fn: Callable[[Any], np.ndarray]) -> np.ndarray:
    if torch.is_tensor(value):
        image = tensor_to_img_fn(value)
    else:
        image = np.asarray(value)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        elif image.ndim == 4 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Stage image must be H x W x 3, got shape {image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError("Stage image contains non-finite values.")
    return np.clip(image, 0.0, 1.0)


def save_analysis_outputs(
    result: Mapping[str, Any],
    *,
    analysis_dir: str,
    basename: str,
    imwrite_fn: Callable[..., str],
    tensor_to_img_fn: Callable[[Any], np.ndarray],
    pipeline_log: Optional[str],
    input_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Save available stages, compact report, and textual pipeline trace."""
    del basename
    output_dir = Path(analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = result.get("stages")
    if not isinstance(stages, Mapping):
        raise TypeError("result['stages'] must be a mapping.")

    manifest: Dict[str, str] = {}
    for stage_name, (stem, image_format) in _STAGE_FILES.items():
        value = stages.get(stage_name)
        if value is None:
            continue
        image = _image_to_numpy(value, tensor_to_img_fn)
        output_base = output_dir / stem
        written = imwrite_fn(image, str(output_base), image_format, quality=95)
        manifest[stage_name] = Path(written).name

    report = dict(result.get("report") or {})
    report["input"] = dict(input_summary or {})
    report["output_files"] = manifest
    report["analysis_directory"] = str(output_dir.resolve())
    (output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "pipeline.log").write_text(pipeline_log or "", encoding="utf-8")
    return manifest


def _load_input(args: argparse.Namespace, pipeline: Any) -> Tuple[np.ndarray, Dict[str, Any], Optional[str]]:
    from utils.file_utils import read_json_file
    from utils.img_utils import (
        demosaice,
        extract_additional_dng_metadata,
        extract_image_from_dng,
        extract_raw_metadata,
        imread,
        normalize_raw,
    )
    from utils.constants import PATH_TO_LINEARIZATION_MODEL

    input_file = args.input_file
    suffix = Path(input_file).suffix.lower()
    metadata_path: Optional[str] = None

    if suffix == ".dng":
        metadata = extract_raw_metadata(input_file)
        metadata.update(extract_additional_dng_metadata(input_file))
        raw_img = extract_image_from_dng(input_file)
        raw_img = normalize_raw(
            img=raw_img,
            black_level=metadata["black_level"],
            white_level=metadata["white_level"],
        ).astype(np.float32)
        if not (raw_img.ndim == 3 and raw_img.shape[-1] == 4):
            raw_img = demosaice(raw_img, metadata["pattern"], tile_mode=True)
        else:
            raw_img = raw_img[..., :3]
    elif suffix == ".png":
        _, dtype = imread(input_file, return_dtype=True)
        if dtype == np.uint16:
            raw_img = imread(input_file).astype(np.float32)
            metadata_path = args.metadata_json or find_json_for_png(input_file)
            metadata = read_json_file(metadata_path)
        else:
            outputs = pipeline.read_image(input_file, log_messages=True, report_time=True)
            raw_img, metadata = outputs["raw"], outputs["metadata"]
            if raw_img is None or metadata is None:
                pipeline.update_model(
                    linearization_model_path=resolve_builtin_path(PATH_TO_LINEARIZATION_MODEL)
                )
                outputs = pipeline.read_image(input_file, log_messages=True, report_time=True)
                raw_img, metadata = outputs["raw"], outputs["metadata"]
    elif suffix in {".jpg", ".jpeg"}:
        outputs = pipeline.read_image(input_file, log_messages=True, report_time=True)
        raw_img, metadata = outputs["raw"], outputs["metadata"]
        if raw_img is None or metadata is None:
            pipeline.update_model(
                linearization_model_path=resolve_builtin_path(PATH_TO_LINEARIZATION_MODEL)
            )
            outputs = pipeline.read_image(input_file, log_messages=True, report_time=True)
            raw_img, metadata = outputs["raw"], outputs["metadata"]
    else:
        raise ValueError("Supported inputs are DNG, PNG, JPEG, and embedded-RAW JPEG.")

    if raw_img is None or metadata is None:
        raise ValueError("Input decoding or linearization did not produce raw RGB and metadata.")
    return validate_raw_image(raw_img), metadata, metadata_path


def _build_pipeline(args: argparse.Namespace, device: torch.device) -> Any:
    from pipeline import PipeLine
    from utils.constants import (
        PATH_TO_GENERIC_AWB_MODEL,
        PATH_TO_POST_AWB_MODEL,
        PATH_TO_RAW_JPEG_MODEL,
        PATH_TO_S24_AWB_MODEL,
    )

    awb_paths = (
        tuple(
            resolve_builtin_path(path)
            for path in (
                PATH_TO_S24_AWB_MODEL,
                PATH_TO_GENERIC_AWB_MODEL,
                PATH_TO_POST_AWB_MODEL,
            )
        )
        if args.re_compute_awb
        else (None, None, None)
    )
    raw_jpeg_path = (
        resolve_builtin_path(PATH_TO_RAW_JPEG_MODEL)
        if Path(args.input_file).suffix.lower() in {".jpg", ".jpeg"}
        else None
    )
    return PipeLine(
        running_device=device,
        denoising_model_path=args.denoising_model_path,
        denoising_model_config_path=args.denoising_model_config_path,
        generic_denoising_model_path=args.denoising_model_path,
        generic_denoising_model_config_path=args.denoising_model_config_path,
        enhancement_model_path=args.enhancement_model_path,
        enhancement_model_config_path=args.enhancement_model_config_path,
        photofinishing_model_path=args.photofinishing_model_path,
        photofinishing_model_config_path=args.photofinishing_model_config_path,
        s24_awb_model_path=awb_paths[0],
        cc_awb_model_path=awb_paths[1],
        post_awb_model_path=awb_paths[2],
        raw_jpeg_adapter_model_path=raw_jpeg_path,
        log="",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if args.pref_awb and not args.re_compute_awb:
        raise ValueError("--pref-awb requires --re-compute-awb.")
    if args.use_cc_awb and not args.re_compute_awb:
        raise ValueError("--use-cc-awb requires --re-compute-awb.")

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from full_pipeline_analysis import FullPipelineAnalyzer
    from utils.img_utils import img_to_tensor, imwrite, tensor_to_img

    device = resolve_device(args.device)
    pipeline = _build_pipeline(args, device)
    pipeline.eval()
    raw_img, metadata, metadata_path = _load_input(args, pipeline)
    illum, ccm = select_awb_inputs(metadata, recompute_awb=args.re_compute_awb)
    raw_tensor = img_to_tensor(raw_img).unsqueeze(0).to(device=device, dtype=torch.float32)

    analyzer = FullPipelineAnalyzer(pipeline)
    with torch.no_grad():
        result = analyzer.run(
            raw=raw_tensor,
            metadata=metadata,
            illum=illum,
            ccm=ccm,
            auto_exposure=not args.disable_auto_exposure,
            denoising_strength=args.denoising_strength,
            luma_denoising_strength=args.luma_denoising_strength,
            chroma_denoising_strength=args.chroma_denoising_strength,
            enhancement_strength=args.enhancement_strength,
            sharpening_amount=args.sharpening_amount,
            use_cc_awb=args.use_cc_awb,
            awb_user_pref=args.pref_awb,
            adjust_ccm=args.adjust_ccm,
            target_cct=args.target_cct,
            target_tint=args.target_tint,
            downscale_ps=not args.no_downscale_ps,
            post_process_ltm=args.post_process_ltm,
            solver_iter=args.solver_iterations,
            contrast_amount=args.contrast_amount,
            vibrance_amount=args.vibrance_amount,
            saturation_amount=args.saturation_amount,
            highlight_amount=args.highlight_amount,
            shadow_amount=args.shadow_amount,
            ev_scale=args.ev_value,
        )

    analysis_dir = Path(args.output_dir) / f"{input_path.stem}-analysis"
    manifest = save_analysis_outputs(
        result,
        analysis_dir=str(analysis_dir),
        basename=input_path.stem,
        imwrite_fn=imwrite,
        tensor_to_img_fn=tensor_to_img,
        pipeline_log=pipeline.get_log(),
        input_summary={
            "source": str(input_path.resolve()),
            "metadata_json": metadata_path,
            "raw_shape": list(raw_img.shape),
            "raw_dtype": str(raw_img.dtype),
            "device": str(device),
        },
    )
    print(f"Analysis written to: {analysis_dir}")
    print(f"Saved stages: {', '.join(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
