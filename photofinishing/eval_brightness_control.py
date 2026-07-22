#!/usr/bin/env python3
"""Evaluate one brightness-control adapter with a common nine-level protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .brightness_ops import log_luminance_from_srgb
    from .control_data import BrightnessSceneDataset, LEVELS
    from .controlled_photofinishing import ControlledLuminancePhotofinishing
    from .train_brightness_control import _import_baseline, _load_checkpoint_with_report
except ImportError:
    from brightness_ops import log_luminance_from_srgb
    from control_data import BrightnessSceneDataset, LEVELS
    from controlled_photofinishing import ControlledLuminancePhotofinishing
    from train_brightness_control import _import_baseline, _load_checkpoint_with_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--adapter-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    adapter_payload = torch.load(args.adapter_checkpoint, map_location="cpu")
    method = adapter_payload["control_method"]
    per_head_request = int(adapter_payload["per_head_request"])

    baseline = _import_baseline(device)
    baseline_report = _load_checkpoint_with_report(
        baseline, args.baseline_checkpoint, device=device,
        allow_partial=args.allow_partial_checkpoint)
    model = ControlledLuminancePhotofinishing(
        baseline, method=method,
        target_params_per_head=per_head_request).to(device)
    incompatible = model.load_state_dict(
        adapter_payload["adapter_state_dict"], strict=False)
    adapter_missing = [
        key for key in incompatible.missing_keys
        if key.startswith(("gain_control.", "gtm_control."))
    ]
    if adapter_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Adapter state mismatch: missing={adapter_missing}, "
            f"unexpected={incompatible.unexpected_keys}")
    model.eval()

    dataset = BrightnessSceneDataset(
        args.input_dir, args.gt_root, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    level_abs_errors = [[] for _ in LEVELS]
    monotonic_violations = 0
    total_adjacent = 0
    fully_monotonic_scenes = 0
    anchor_l1 = []
    curve_mae = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            image = batch["input"].to(device)
            targets = batch["targets"].to(device)
            batch_size = image.shape[0]
            predictions = []
            for level_idx, (_, alpha_value) in enumerate(LEVELS):
                alpha = image.new_full((batch_size, 1), alpha_value)
                prediction = model(image, alpha, training_mode=True)["output"]
                predictions.append(prediction)
                error = (
                    log_luminance_from_srgb(prediction)
                    - log_luminance_from_srgb(targets[:, level_idx])
                ).abs().mean(dim=(1, 2, 3))
                level_abs_errors[level_idx].extend(error.cpu().tolist())
            pred = torch.stack(predictions, dim=1)
            pred_b = torch.stack([
                log_luminance_from_srgb(pred[:, idx]).mean(dim=(1, 2, 3))
                for idx in range(len(LEVELS))
            ], dim=1)
            target_b = torch.stack([
                log_luminance_from_srgb(targets[:, idx]).mean(dim=(1, 2, 3))
                for idx in range(len(LEVELS))
            ], dim=1)
            diffs = pred_b[:, 1:] - pred_b[:, :-1]
            violations = diffs <= 0
            monotonic_violations += int(violations.sum())
            total_adjacent += int(violations.numel())
            fully_monotonic_scenes += int((~violations).all(dim=1).sum())

            pred_range = (pred_b[:, -1] - pred_b[:, 0]).clamp_min(1e-6)
            target_range = (target_b[:, -1] - target_b[:, 0]).clamp_min(1e-6)
            pred_r = (pred_b - pred_b[:, :1]) / pred_range[:, None]
            target_r = (target_b - target_b[:, :1]) / target_range[:, None]
            curve_mae.extend((pred_r - target_r).abs().mean(dim=1).cpu().tolist())
            anchor_l1.extend(
                (pred[:, 4] - targets[:, 4]).abs().mean(dim=(1, 2, 3)).cpu().tolist())

    report = {
        "control_method": method,
        "num_scenes": len(dataset),
        "trainable_parameters": model.trainable_parameter_count(),
        "baseline_load_report": baseline_report,
        "per_level_log_luminance_mae": {
            level_name: sum(values) / len(values)
            for (level_name, _), values in zip(LEVELS, level_abs_errors)
        },
        "mean_log_luminance_mae": (
            sum(sum(values) for values in level_abs_errors)
            / sum(len(values) for values in level_abs_errors)
        ),
        "monotonic_violation_rate": monotonic_violations / max(total_adjacent, 1),
        "fully_monotonic_scene_rate": fully_monotonic_scenes / max(len(dataset), 1),
        "mean_normalized_curve_mae": sum(curve_mae) / max(len(curve_mae), 1),
        "mean_alpha_zero_l1": sum(anchor_l1) / max(len(anchor_l1), 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
