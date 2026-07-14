from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from adjustTM.constants import LEVELS
from adjustTM.transfer import linear_luminance, srgb_to_linear
from .image_io import fit_pad_tensor, read_linear_png16, unpad_tensor, write_srgb_png16
from .methods import MethodRunner, load_runners
from .schemas import canonical_hash, read_json, write_json_atomic


def _mean_log_luma(image: torch.Tensor) -> float:
    return float(linear_luminance(srgb_to_linear(image)).clamp_min(1e-4).log().mean())


def _chroma_rg_mae(first: torch.Tensor, second: torch.Tensor) -> float:
    first_linear = srgb_to_linear(first)
    second_linear = srgb_to_linear(second)
    first_rg = first_linear[:, :2] / first_linear.sum(dim=1, keepdim=True).clamp_min(1e-6)
    second_rg = second_linear[:, :2] / second_linear.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return float((first_rg - second_rg).abs().mean())


def _append_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_cached_outputs(
    *,
    manifest: Mapping[str, object],
    runners: Mapping[str, MethodRunner],
    output_root: str | Path,
    protocol_hash: str,
    levels: Sequence[tuple[str, float]] = LEVELS,
    dense_alphas: Sequence[float] = tuple(np.linspace(-1.0, 1.0, 41)),
    max_side: int | None = 512,
    multiple: int = 16,
    device: str | torch.device = "cpu",
) -> dict[str, list[dict[str, object]]]:
    output_root = Path(output_root)
    identity_path = output_root / "cache_identity.json"
    identity = {
        "manifest_hash": str(manifest.get("content_sha256", canonical_hash(manifest))),
        "protocol_hash": protocol_hash,
        "methods": {name: dict(runner.metadata()) for name, runner in runners.items()},
        "levels": {name: alpha for name, alpha in levels},
        "dense_alphas": [float(alpha) for alpha in dense_alphas],
        "max_side": max_side,
        "multiple": multiple,
    }
    identity["identity_hash"] = canonical_hash(identity)
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("identity_hash") != identity["identity_hash"]:
            raise RuntimeError("Output cache exists with a different benchmark identity")
    else:
        write_json_atomic(identity_path, identity)

    inference_records: list[dict[str, object]] = []
    dense_records: list[dict[str, object]] = []
    for scene in manifest["scenes"]:  # type: ignore[index]
        scene_id = str(scene["scene_id"])
        image = read_linear_png16(scene["input"]["path"], device=device)
        image, geometry = fit_pad_tensor(image, max_side=max_side, multiple=multiple)
        for method_name, runner in runners.items():
            zero_result = runner.predict(image, 0.0)["output"]
            zero_output = unpad_tensor(zero_result, geometry)
            zero_drift = 0.0
            if hasattr(runner, "zero_reference"):
                zero_reference = unpad_tensor(runner.zero_reference(image), geometry)
                zero_drift = float((zero_output - zero_reference).abs().max())
                if zero_drift > 1e-7:
                    raise RuntimeError(f"alpha=0 identity violation for {method_name}/{scene_id}: {zero_drift}")
            for level_name, alpha in levels:
                start = time.perf_counter()
                result = runner.predict(image, float(alpha))
                output = unpad_tensor(result["output"], geometry)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"Non-finite output: {method_name}/{scene_id}/{level_name}")
                output_path = output_root / method_name / level_name / scene_id
                write_srgb_png16(output_path, output)
                inference_records.append({
                    "scene_id": scene_id,
                    "method": method_name,
                    "level": level_name,
                    "alpha": float(alpha),
                    "output_path": str(output_path),
                    "runtime_ms": elapsed_ms,
                    "height": int(output.shape[-2]),
                    "width": int(output.shape[-1]),
                    "alpha_zero_max_drift": zero_drift,
                })
            for alpha in dense_alphas:
                result = runner.predict(image, float(alpha))
                output = unpad_tensor(result["output"], geometry)
                dense_record = {
                    "scene_id": scene_id,
                    "method": method_name,
                    "alpha": float(alpha),
                    "mean_log_luma": _mean_log_luma(output),
                    "clip_ratio": float((output >= 0.999).float().mean()),
                    "deep_shadow_ratio": float((linear_luminance(srgb_to_linear(output)) <= 0.01).float().mean()),
                    "chroma_rg_drift_from_zero": _chroma_rg_mae(output, zero_output),
                }
                if "gain_factor" in result:
                    dense_record["gain_factor"] = float(result["gain_factor"].float().mean())
                if "gtm_params" in result:
                    dense_record["gtm_parameters"] = [float(value) for value in result["gtm_params"].float().reshape(-1)[:3]]
                dense_records.append(dense_record)
    _append_jsonl(output_root / "inference_records.jsonl", inference_records)
    _append_jsonl(output_root / "dense_control_records.jsonl", dense_records)
    return {"inference_records": inference_records, "dense_records": dense_records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and cache outputs for all adjustTM benchmark methods")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dense-steps", type=int, default=41)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--multiple", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    protocol = read_json(args.protocol)
    runners = load_runners(args.methods, args.device)
    dense_alphas = np.linspace(-1.0, 1.0, args.dense_steps) if args.dense_steps > 0 else []
    generate_cached_outputs(
        manifest=manifest,
        runners=runners,
        output_root=args.output_root,
        protocol_hash=canonical_hash(protocol),
        dense_alphas=dense_alphas,
        max_side=args.max_side,
        multiple=args.multiple,
        device=args.device,
    )


if __name__ == "__main__":
    main()
