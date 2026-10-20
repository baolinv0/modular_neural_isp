"""Compare baseline and adapted non-pixel style distances on a held-out split."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

try:
    from .train_unpaired_style import _forward_training, _load_model, _move_batch, _resolve_device
    from .unpaired_reference_data import ReferenceStyleDataset
    from .unpaired_style_losses import (
        chroma_histogram_loss, chroma_moment_loss, log_exposure_loss, luma_distribution_loss,
        luma_percentile_loss, saturation_distribution_loss,
    )
except ImportError:
    from train_unpaired_style import _forward_training, _load_model, _move_batch, _resolve_device
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-mode", choices=["linear_srgb", "raw_metadata"], default="linear_srgb")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--use-3d-lut", action="store_true")
    return parser


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
    adapted = _load_model(args.adapted_load, device, args.use_3d_lut).eval()
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
