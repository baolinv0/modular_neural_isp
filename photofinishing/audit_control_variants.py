#!/usr/bin/env python3
"""Audit realized trainable parameter counts for all control variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .controlled_photofinishing import (
        ControlledLuminancePhotofinishing,
        find_target_params_per_head,
    )
    from .train_brightness_control import (
        CONTROL_METHODS,
        _import_baseline,
        _load_checkpoint_with_report,
    )
except ImportError:
    from controlled_photofinishing import (
        ControlledLuminancePhotofinishing,
        find_target_params_per_head,
    )
    from train_brightness_control import (
        CONTROL_METHODS,
        _import_baseline,
        _load_checkpoint_with_report,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--parameter-budget", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-relative-deviation", type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for method in CONTROL_METHODS:
        baseline = _import_baseline(device)
        _load_checkpoint_with_report(
            baseline, args.baseline_checkpoint, device=device,
            allow_partial=args.allow_partial_checkpoint)
        request, _ = find_target_params_per_head(
            baseline, method, args.parameter_budget)
        model = ControlledLuminancePhotofinishing(
            baseline, method=method,
            target_params_per_head=request)
        actual = model.trainable_parameter_count()
        rows.append({
            "method": method,
            "per_head_request": request,
            "trainable_parameters": actual,
            "relative_to_budget": (
                actual - args.parameter_budget) / args.parameter_budget,
        })

    counts = [row["trainable_parameters"] for row in rows]
    reference = sum(counts) / len(counts)
    for row in rows:
        row["relative_to_mean"] = (
            row["trainable_parameters"] - reference) / reference
    report = {
        "desired_budget": args.parameter_budget,
        "mean_realized": reference,
        "variants": rows,
        "pass": max(abs(row["relative_to_mean"]) for row in rows)
        <= args.max_relative_deviation,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
