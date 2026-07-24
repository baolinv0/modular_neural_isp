"""Batch-render 16-bit linear RGB images through the photofinishing/TM stage.

The inputs are expected to be three-channel, uint16, linear sRGB images that
have already undergone white balance and camera-to-sRGB color correction.
This entry point intentionally skips RAW denoising, AWB, CCM, and RAW-to-linear
sRGB conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SUPPORTED_INPUT_EXTENSIONS = {".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class BatchRecord:
    input_file: str
    output_files: str
    status: str
    elapsed_seconds: float
    error: str = ""


def select_device(device_name: str) -> torch.device:
    """Resolve a CLI device name to a torch device."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if device_name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps was requested, but Apple MPS is unavailable.")
        return torch.device("mps")
    return torch.device("cpu")


def discover_input_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Return sorted supported image files below ``input_dir``."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    iterator: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )


def load_linear_rgb16(path: Path, device: torch.device) -> torch.Tensor:
    """Load a uint16 RGB image and return normalized NCHW float32 linear RGB."""
    path = Path(path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise ValueError(f"Failed to decode image: {path}")
    if image_bgr.dtype != np.uint16:
        raise ValueError(f"Expected a 16-bit unsigned image, got dtype={image_bgr.dtype} for {path}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected exactly three RGB channels, got shape={image_bgr.shape} for {path}")

    image_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
    image = image_rgb.astype(np.float32) / np.float32(65535.0)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def resolve_config_path(model_path: Path, explicit_config_path: Path | None = None) -> Path:
    """Resolve a model config using the repository's existing path conventions."""
    model_path = Path(model_path)
    if explicit_config_path is not None:
        config_path = Path(explicit_config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        return config_path

    stem = model_path.stem
    candidates = [
        model_path.with_suffix(".json"),
        model_path.parent.parent / "config" / f"{stem}.json",
        model_path.parent.parent / "configs" / f"{stem}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not locate a config for model {model_path}. Checked:\n{checked}")


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_photofinishing_model(model_path: Path, config_path: Path | None, device: torch.device) -> torch.nn.Module:
    """Load the trained photofinishing module."""
    from photofinishing.photofinishing_model import PhotofinishingModule

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Photofinishing model does not exist: {model_path}")
    resolved_config = resolve_config_path(model_path, config_path)
    config = _read_json(resolved_config)
    model = PhotofinishingModule(device=device, use_3d_lut=bool(config.get("use_3d_lut", False)))
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_enhancement_model(model_path: Path, config_path: Path | None, device: torch.device) -> torch.nn.Module:
    """Load the NAFNet detail-enhancement model."""
    from denoising.nafnet_arch import NAFNet

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Enhancement model does not exist: {model_path}")
    resolved_config = resolve_config_path(model_path, config_path)
    config = _read_json(resolved_config)
    model = NAFNet(
        width=config["width"],
        middle_block_num=config["middle_block_num"],
        encoder_block_nums=config["encoder_block_nums"],
        decoder_block_nums=config["decoder_block_nums"],
    ).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_upsampler(device: torch.device) -> torch.nn.Module:
    """Create the same BGU module used by the main pipeline."""
    from upsampling.bilateral_guided_upsampling import BGU

    return BGU(reg_lambda=1e-7, reg_T=True).to(device)


def render_linear_rgb(
    linear_rgb: torch.Tensor,
    photofinishing_model: torch.nn.Module,
    upsampler: torch.nn.Module,
    *,
    enhancement_model: torch.nn.Module | None = None,
    enhancement_strength: float = 1.0,
    downsample_photofinishing: bool = True,
    post_process_ltm: bool = False,
    solver_iterations: int = 50,
    contrast_amount: float = 0.0,
    vibrance_amount: float = 0.0,
    saturation_amount: float = 0.0,
    highlight_amount: float = 0.0,
    shadow_amount: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a color-corrected linear RGB image through TM and detail enhancement.

    Returns:
        (tm_output, enhanced_output). ``enhanced_output`` is None only when no
        enhancement model is supplied.
    """
    if linear_rgb.ndim != 4 or linear_rgb.shape[0] != 1 or linear_rgb.shape[1] != 3:
        raise ValueError(f"Expected linear RGB tensor with shape [1, 3, H, W], got {tuple(linear_rgb.shape)}")
    if not 0.0 <= enhancement_strength <= 1.0:
        raise ValueError("enhancement_strength must be within [0, 1].")

    kwargs = dict(
        return_intermediate=False,
        report_time=False,
        return_params=False,
        post_process_ltm=post_process_ltm,
        solver_iter=solver_iterations,
        contrast_amount=contrast_amount,
        vibrance_amount=vibrance_amount,
        saturation_amount=saturation_amount,
        highlight_amount=highlight_amount,
        shadow_amount=shadow_amount,
    )

    if downsample_photofinishing:
        height, width = linear_rgb.shape[-2:]
        if min(height, width) < 64:
            raise ValueError("Images smaller than 64 pixels on either side require --no-downsampling.")
        low_res_input = F.interpolate(linear_rgb, scale_factor=0.25, mode="bilinear", align_corners=True)
        low_res_output = photofinishing_model(low_res_input, **kwargs)
        tm_output = upsampler(linear_rgb, low_res_input, low_res_output)
    else:
        tm_output = photofinishing_model(linear_rgb, **kwargs)

    tm_output = tm_output.clamp(0.0, 1.0)
    enhanced_output = None
    if enhancement_model is not None:
        enhanced = enhancement_model(tm_output).clamp(0.0, 1.0)
        enhanced_output = ((1.0 - enhancement_strength) * tm_output + enhancement_strength * enhanced).clamp(0.0, 1.0)
    return tm_output, enhanced_output


def save_rendered_image(image: torch.Tensor, output_path: Path, *, output_format: str, jpeg_quality: int) -> Path:
    """Save one rendered tensor as PNG-16 or JPEG."""
    if output_format not in {"png16", "jpeg"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be within [1, 100].")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()

    if output_format == "png16":
        output_path = output_path.with_suffix(".png")
        encoded = np.rint(rgb * 65535.0).astype(np.uint16)
        params: Sequence[int] = ()
    else:
        output_path = output_path.with_suffix(".jpg")
        encoded = np.rint(rgb * 255.0).astype(np.uint8)
        params = (int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality))

    if not cv2.imwrite(str(output_path), encoded[..., ::-1], list(params)):
        raise OSError(f"Failed to save output image: {output_path}")
    return output_path


def _output_specs(output_format: str) -> tuple[str, ...]:
    if output_format == "both":
        return ("png16", "jpeg")
    return (output_format,)


def _write_manifest(path: Path, records: Sequence[BatchRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_file", "output_files", "status", "elapsed_seconds", "error"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "input_file": record.input_file,
                    "output_files": record.output_files,
                    "status": record.status,
                    "elapsed_seconds": f"{record.elapsed_seconds:.6f}",
                    "error": record.error,
                }
            )


def _expected_output_paths(output_base: Path, output_format: str) -> list[Path]:
    expected: list[Path] = []
    for fmt in _output_specs(output_format):
        suffix = ".png" if fmt == "png16" else ".jpg"
        expected.append((output_base.parent / f"{output_base.name}-tm").with_suffix(suffix))
        expected.append((output_base.parent / f"{output_base.name}-tm-enhanced").with_suffix(suffix))
    return expected


def run_batch(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    files = discover_input_files(input_dir, recursive=args.recursive)
    if not files:
        raise FileNotFoundError(
            f"No supported 16-bit inputs found in {input_dir}. Supported extensions: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Found {len(files)} input image(s).")

    photofinishing_model = load_photofinishing_model(
        Path(args.photofinishing_model_path),
        Path(args.photofinishing_config_path) if args.photofinishing_config_path else None,
        device,
    )
    enhancement_model = load_enhancement_model(
        Path(args.enhancement_model_path),
        Path(args.enhancement_config_path) if args.enhancement_config_path else None,
        device,
    )
    upsampler = create_upsampler(device)

    records: list[BatchRecord] = []
    success_count = 0
    for index, input_path in enumerate(files, start=1):
        started = time.perf_counter()
        relative = input_path.relative_to(input_dir)
        output_base = output_dir / relative.parent / relative.stem
        print(f"[{index}/{len(files)}] {relative.as_posix()}")
        try:
            expected_outputs = _expected_output_paths(output_base, args.output_format)
            if args.skip_existing and expected_outputs and all(path.exists() for path in expected_outputs):
                records.append(
                    BatchRecord(
                        input_file=str(input_path),
                        output_files=";".join(str(path) for path in expected_outputs),
                        status="skipped",
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                continue

            linear_rgb = load_linear_rgb16(input_path, device=device)
            with torch.inference_mode():
                tm_output, enhanced_output = render_linear_rgb(
                    linear_rgb,
                    photofinishing_model,
                    upsampler,
                    enhancement_model=enhancement_model,
                    enhancement_strength=args.enhancement_strength,
                    downsample_photofinishing=not args.no_downsampling,
                    post_process_ltm=args.post_process_ltm,
                    solver_iterations=args.solver_iterations,
                    contrast_amount=args.contrast_amount,
                    vibrance_amount=args.vibrance_amount,
                    saturation_amount=args.saturation_amount,
                    highlight_amount=args.highlight_amount,
                    shadow_amount=args.shadow_amount,
                )

            if enhanced_output is None:
                raise RuntimeError("Enhancement output is unexpectedly None. Enhancement model must be provided.")

            output_paths: list[Path] = []
            for fmt in _output_specs(args.output_format):
                output_paths.append(
                    save_rendered_image(
                        tm_output,
                        output_base.parent / f"{output_base.name}-tm",
                        output_format=fmt,
                        jpeg_quality=args.jpeg_quality,
                    )
                )
                output_paths.append(
                    save_rendered_image(
                        enhanced_output,
                        output_base.parent / f"{output_base.name}-tm-enhanced",
                        output_format=fmt,
                        jpeg_quality=args.jpeg_quality,
                    )
                )

            success_count += 1
            records.append(
                BatchRecord(
                    input_file=str(input_path),
                    output_files=";".join(str(path) for path in output_paths),
                    status="ok",
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        except Exception as exc:
            records.append(
                BatchRecord(
                    input_file=str(input_path),
                    output_files="",
                    status="failed",
                    elapsed_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                break

    manifest_path = output_dir / args.manifest_name
    _write_manifest(manifest_path, records)
    failed_count = sum(record.status == "failed" for record in records)
    skipped_count = sum(record.status == "skipped" for record in records)
    print(f"Completed: {success_count} succeeded, {failed_count} failed, {skipped_count} skipped. Manifest: {manifest_path}")
    return 1 if failed_count else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process uint16 linear sRGB images that already include AWB and CCM, starting directly "
            "from the photofinishing/TM stage, then always run detail enhancement and save both results."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing uint16 RGB PNG/TIFF files.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered outputs and the CSV manifest.")
    parser.add_argument("--photofinishing-model-path", required=True, help="Trained photofinishing .pth model.")
    parser.add_argument("--photofinishing-config-path", default=None, help="Optional explicit photofinishing JSON config.")
    parser.add_argument(
        "--enhancement-model-path",
        required=True,
        help="Trained detail-enhancement NAFNet .pth model. Enhancement runs by default and both outputs are saved.",
    )
    parser.add_argument("--enhancement-config-path", default=None, help="Optional explicit enhancement JSON config.")
    parser.add_argument("--enhancement-strength", type=float, default=1.0, help="Enhancement blend strength in [0, 1].")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu", "mps"], default="auto")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan the input directory.")
    parser.add_argument("--no-downsampling", action="store_true", help="Run photofinishing at full resolution and skip BGU.")
    parser.add_argument("--post-process-ltm", action="store_true", help="Enable multi-scale LTM refinement.")
    parser.add_argument("--solver-iterations", type=int, default=50, help="Bilateral-solver iterations for LTM refinement.")
    parser.add_argument("--contrast-amount", type=float, default=0.0)
    parser.add_argument("--vibrance-amount", type=float, default=0.0)
    parser.add_argument("--saturation-amount", type=float, default=0.0)
    parser.add_argument("--highlight-amount", type=float, default=0.0)
    parser.add_argument("--shadow-amount", type=float, default=0.0)
    parser.add_argument(
        "--output-format",
        choices=["png16", "jpeg", "both"],
        default="both",
        help=(
            "Output format. Default is both, so TM and TM-enhanced images "
            "are saved as both PNG16 and JPEG."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--manifest-name", default="batch_results.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
