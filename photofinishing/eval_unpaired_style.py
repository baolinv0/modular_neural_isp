"""Compare baseline and adapted non-pixel style distances on a held-out split."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

try:
    from .train_unpaired_style import _forward_training, _load_model, _move_batch, _resolve_device
    from .unpaired_chroma_heads import ChromaHead, configure_chroma_head
    from .unpaired_reference_data import ReferenceStyleDataset
    from .unpaired_style_losses import (
        chroma_histogram_loss, chroma_moment_loss, log_exposure_loss, luma_distribution_loss,
        luma_percentile_loss, saturation_distribution_loss,
    )
except ImportError:
    from train_unpaired_style import _forward_training, _load_model, _move_batch, _resolve_device
    from unpaired_chroma_heads import ChromaHead, configure_chroma_head
    from unpaired_reference_data import ReferenceStyleDataset
    from unpaired_style_losses import (
        chroma_histogram_loss, chroma_moment_loss, log_exposure_loss, luma_distribution_loss,
        luma_percentile_loss, saturation_distribution_loss,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline and adapted non-pixel style distances")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--baseline-load", required=True)
    parser.add_argument("--adapted-load", required=True)
    parser.add_argument(
        "--adapted-run-config",
        default=None,
        help="Adapted run_config.json; defaults to the directory beside --adapted-load",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-mode", choices=["linear_srgb", "raw_metadata"], default="linear_srgb")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--use-3d-lut", action="store_true")
    return parser


def _read_adapted_run_config(
    adapted_checkpoint: str,
    configured_path: Optional[str],
) -> Tuple[Dict[str, object], Optional[Path]]:
    if configured_path is not None:
        config_path = Path(configured_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Adapted run config not found: {config_path}")
    else:
        candidate = Path(adapted_checkpoint).resolve().parent / "run_config.json"
        if not candidate.is_file():
            return {"chroma_head": ChromaHead.FULL_LUT.value}, None
        config_path = candidate

    payload: Dict[str, object] = json.loads(config_path.read_text(encoding="utf-8"))
    raw_head = str(payload.get("chroma_head", ChromaHead.FULL_LUT.value))
    try:
        head = ChromaHead(raw_head)
    except ValueError as exc:
        raise ValueError(f"unknown chroma_head in {config_path}: {raw_head}") from exc
    payload["chroma_head"] = head.value

    if head is ChromaHead.AFFINE_RESIDUAL:
        if payload.get("stage") != "chroma":
            raise ValueError(f"affine_residual run config must be a chroma-stage run: {config_path}")
        try:
            matrix_limit = float(payload["affine_matrix_limit"])
            bias_limit = float(payload["affine_bias_limit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"affine run config is missing valid limits: {config_path}") from exc
        if not math.isfinite(matrix_limit) or not math.isfinite(bias_limit) or matrix_limit <= 0 or bias_limit <= 0:
            raise ValueError(f"affine run config requires positive affine limits: {config_path}")
        payload["affine_matrix_limit"] = matrix_limit
        payload["affine_bias_limit"] = bias_limit
    return payload, config_path


def _load_adapted_model(
    checkpoint: str,
    device: torch.device,
    use_3d_lut: bool,
    run_config: Dict[str, object],
) -> torch.nn.Module:
    head = ChromaHead(str(run_config.get("chroma_head", ChromaHead.FULL_LUT.value)))
    if head is ChromaHead.FULL_LUT:
        return _load_model(checkpoint, device, use_3d_lut)

    try:
        from .photofinishing_model import PhotofinishingModule
    except ImportError:
        from photofinishing_model import PhotofinishingModule

    model = PhotofinishingModule(device=device, use_3d_lut=use_3d_lut)
    configure_chroma_head(
        model,
        head,
        matrix_limit=float(run_config["affine_matrix_limit"]),
        bias_limit=float(run_config["affine_bias_limit"]),
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _metrics(output: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    return {
        "exposure": float(log_exposure_loss(output, reference)),
        "percentiles": float(luma_percentile_loss(output, reference)),
        "luma_distribution": float(luma_distribution_loss(output, reference)),
        "chroma_histogram": float(chroma_histogram_loss(output, reference)),
        "chroma_moments": float(chroma_moment_loss(output, reference)),
        "saturation": float(saturation_distribution_loss(output, reference)),
    }


@torch.no_grad()
def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    baseline = _load_model(args.baseline_load, device, args.use_3d_lut).eval()
    adapted_config, adapted_config_path = _read_adapted_run_config(args.adapted_load, args.adapted_run_config)
    adapted = _load_adapted_model(args.adapted_load, device, args.use_3d_lut, adapted_config).eval()
    dataset = ReferenceStyleDataset(args.manifest, args.split, args.image_size, args.input_mode)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    sample_metrics = []
    sums = {"baseline": {}, "adapted": {}}
    for batch in loader:
        input_image, reference = _move_batch(batch, device)
        baseline_output = _forward_training(baseline, input_image)["output"]
        adapted_output = _forward_training(adapted, input_image)["output"]
        before = _metrics(baseline_output, reference)
        after = _metrics(adapted_output, reference)
        sample_id = batch["sample_id"][0]
        sample_metrics.append({
            "sample_id": sample_id,
            "baseline": before,
            "adapted": after,
            "improvement": {name: before[name] - after[name] for name in before},
        })
        for group, values in (("baseline", before), ("adapted", after)):
            for name, value in values.items():
                sums[group][name] = sums[group].get(name, 0.0) + value
    if not sample_metrics:
        raise RuntimeError("Empty evaluation loader")
    count = len(sample_metrics)
    baseline_mean = {name: value / count for name, value in sums["baseline"].items()}
    adapted_mean = {name: value / count for name, value in sums["adapted"].items()}
    payload = {
        "split": args.split,
        "num_samples": count,
        "adapted_chroma_head": adapted_config["chroma_head"],
        "adapted_run_config": str(adapted_config_path) if adapted_config_path else None,
        "baseline_mean": baseline_mean,
        "adapted_mean": adapted_mean,
        "mean_improvement": {name: baseline_mean[name] - adapted_mean[name] for name in baseline_mean},
        "samples": sample_metrics,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
