from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from capture_pipeline.factory import build_capture_pipeline
from capture_pipeline.types import CapturePipelineOutput, RawFrame


def parse_override_ev(value: Optional[float], ev_min: float, ev_max: float) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not ev_min <= value <= ev_max:
        raise ValueError(f"override EV {value} is outside [{ev_min}, {ev_max}].")
    return value


def _metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...], name: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    raise KeyError(f"Metadata is missing {name}; checked keys {keys}.")


def build_raw_frame_from_metadata(mosaic: torch.Tensor, metadata: Mapping[str, Any], *, normalized: bool) -> RawFrame:
    black = _metadata_value(metadata, ("black_level", "black_levels"), "black level") if not normalized else 0.0
    white = _metadata_value(metadata, ("white_level", "sensor_white_level"), "white level") if not normalized else 1.0
    pattern = _metadata_value(metadata, ("cfa_pattern", "pattern", "bayer_pattern"), "CFA pattern")
    if isinstance(pattern, bytes):
        pattern = pattern.decode("utf-8")
    return RawFrame(mosaic=mosaic, black_level=black, white_level=white, cfa_pattern=str(pattern), metadata=dict(metadata), is_normalized=normalized)


def stage_filename(index: int, name: str, *, is_final: bool) -> str:
    extension = "jpg" if is_final else "png"
    return f"{index:02d}-{name.replace('_', '-')}.{extension}"


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _tensor_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu()
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("CLI stage export supports batch size 1.")
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)
    result = array.numpy()
    if result.ndim == 3 and result.shape[2] == 1:
        result = result[:, :, 0]
    return np.clip(result, 0.0, 1.0)


def _write_stage(path: Path, tensor: torch.Tensor, *, final: bool) -> None:
    image = _tensor_image(tensor)
    if final:
        encoded = np.round(image * 255.0).astype(np.uint8)
    else:
        encoded = np.round(image * 65535.0).astype(np.uint16)
    if encoded.ndim == 3 and encoded.shape[2] == 3:
        encoded = cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"Failed to write image: {path}")


def save_pipeline_outputs(result: CapturePipelineOutput, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    stage_items = list(result.stages.items())
    for index, (name, tensor) in enumerate(stage_items):
        final = name == "final_srgb"
        filename = stage_filename(index, name, is_final=final)
        path = output_dir / filename
        _write_stage(path, tensor, final=final)
        written.append(str(path))
    report = {
        "estimated_ev": _to_jsonable(result.ae.ev),
        "ae_confidence": _to_jsonable(result.ae.confidence),
        "illuminant": _to_jsonable(result.awb.illuminant),
        "ccm": _to_jsonable(result.awb.ccm),
        "tone_parameters": _to_jsonable(result.tone.parameters),
        "diagnostics": _to_jsonable(result.diagnostics),
        "stage_files": written,
    }
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    written.append(str(output_dir / "analysis.json"))
    return written


def _read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_input(input_path: Path, metadata_path: Optional[Path], normalized: bool) -> RawFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".dng":
        try:
            from utils.img_utils import extract_additional_dng_metadata, extract_image_from_dng, extract_raw_metadata
        except ImportError as exc:
            raise RuntimeError("DNG input requires repository utils.img_utils and rawpy dependencies.") from exc
        metadata = extract_raw_metadata(str(input_path))
        metadata.update(extract_additional_dng_metadata(str(input_path)))
        mosaic_np = extract_image_from_dng(str(input_path)).astype(np.float32)
    elif suffix == ".png":
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Cannot read input: {input_path}")
        if image.ndim != 2:
            raise ValueError("Bayer PNG input must be single-channel.")
        if metadata_path is None:
            candidate = input_path.with_suffix(".json")
            if not candidate.is_file():
                raise FileNotFoundError("PNG Bayer input requires --metadata-json or a sidecar JSON with the same stem.")
            metadata_path = candidate
        metadata = _read_metadata(metadata_path)
        if normalized:
            max_value = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
            mosaic_np = image.astype(np.float32) / float(max_value)
        else:
            mosaic_np = image.astype(np.float32)
    else:
        raise ValueError("Input must be a Bayer .dng or single-channel .png file.")
    mosaic = torch.from_numpy(mosaic_np).unsqueeze(0).unsqueeze(0)
    return build_raw_frame_from_metadata(mosaic, metadata, normalized=normalized)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the modular RAW capture pipeline.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--metadata-json")
    parser.add_argument("--config-json")
    parser.add_argument("--output-dir", default="capture-results")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--normalized-input", action="store_true")
    parser.add_argument("--override-ev", type=float)
    parser.add_argument("--ev-min", type=float, default=-4.0)
    parser.add_argument("--ev-max", type=float, default=4.0)
    parser.add_argument("--clipping-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--trainable-modules", nargs="*")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "gpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    config = _read_metadata(Path(args.config_json)) if args.config_json else {}
    config.update({"ev_min": args.ev_min, "ev_max": args.ev_max, "clipping_mode": args.clipping_mode})
    if args.trainable_modules is not None:
        config["trainable_modules"] = args.trainable_modules
    pipeline = build_capture_pipeline(config, device)
    frame = _load_input(
        Path(args.input_file), Path(args.metadata_json) if args.metadata_json else None, args.normalized_input
    )
    frame = RawFrame(
        frame.mosaic.to(device), frame.black_level.to(device), frame.white_level.to(device),
        frame.cfa_pattern, frame.metadata, frame.is_normalized,
    )
    override_ev = parse_override_ev(args.override_ev, args.ev_min, args.ev_max)
    with torch.set_grad_enabled(bool(args.trainable_modules)):
        result = pipeline(frame, override_ev=override_ev)
    output_dir = Path(args.output_dir) / f"{Path(args.input_file).stem}-capture"
    files = save_pipeline_outputs(result, output_dir)
    print(f"Saved {len(files)} outputs to {output_dir}")


if __name__ == "__main__":
    main()
