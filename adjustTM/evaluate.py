from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CONTROL_METHODS, LEVELS
from .dataset import MultiLevelDataset
from .metrics import (
    chroma_rg_mae,
    clipping_ratio,
    deep_shadow_ratio,
    log_luma_mae,
    luminance_psnr,
    luminance_ssim,
    mean_log_luminance,
    summarize_values,
    trajectory_metrics,
)
from .model import ControlledBrightnessISP, load_baseline_checkpoint, load_control_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an adjustTM control checkpoint")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--control-method", choices=CONTROL_METHODS, required=True)
    parser.add_argument("--output", default="adjustTM/results.json")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dense-steps", type=int, default=41)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _mean_dict(records: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(values) / max(len(values), 1) for key, values in records.items()}


def _latency_ms(
    model: ControlledBrightnessISP,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> float:
    alpha = torch.zeros(x.shape[0], device=device)
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            model(x, alpha)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max(runs, 1)):
            model(x, alpha)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / max(runs, 1) / x.shape[0]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = MultiLevelDataset(args.input_dir, args.gt_root, image_size=args.image_size)
    if args.max_scenes:
        selected = dataset.scene_names[: args.max_scenes]
        dataset = MultiLevelDataset(args.input_dir, args.gt_root, image_size=args.image_size, scene_names=selected)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = ControlledBrightnessISP(args.control_method, device=device).to(device)
    load_baseline_checkpoint(model.baseline, args.baseline_checkpoint, map_location=device)
    model.freeze_baseline()
    load_control_checkpoint(model, args.control_checkpoint, map_location=device)
    model.assert_baseline_frozen()
    model.eval()

    per_level: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    scene_curves: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    gain_values: list[float] = []
    gtm_values: list[list[float]] = [[], [], []]
    start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate levels"):
            x = batch["in_image"].to(device, non_blocking=True)
            gt = batch["gt_image"].to(device, non_blocking=True)
            alpha = batch["alpha"].to(device, non_blocking=True)
            result = model(x, alpha)
            pred = result["output"]
            zero = model(x, torch.zeros_like(alpha))["output"]
            batch_metrics = {
                "log_luma_mae": log_luma_mae(pred, gt),
                "luma_psnr": luminance_psnr(pred, gt),
                "luma_ssim": luminance_ssim(pred, gt),
                "chroma_rg_mae_to_gt": chroma_rg_mae(pred, gt),
                "chroma_rg_drift_from_zero": chroma_rg_mae(pred, zero),
                "clip_ratio": clipping_ratio(pred),
                "deep_shadow_ratio": deep_shadow_ratio(pred),
            }
            pred_means = mean_log_luminance(pred)
            target_means = mean_log_luminance(gt)
            gains = result["gain_factor"].view(x.shape[0], -1).mean(dim=1)
            gtm = result["gtm_params"].view(x.shape[0], 3)

            for index, (scene_name, level_name) in enumerate(zip(batch["scene_name"], batch["level_name"])):
                for metric_name, values in batch_metrics.items():
                    per_level[level_name][metric_name].append(float(values[index]))
                scene_curves[scene_name][level_name] = (float(pred_means[index]), float(target_means[index]))
                gain_values.append(float(gains[index]))
                for parameter_index in range(3):
                    gtm_values[parameter_index].append(float(gtm[index, parameter_index]))

        level_order = [name for name, _ in LEVELS]
        trajectory_records: dict[str, list[float]] = defaultdict(list)
        for scene_name, records in scene_curves.items():
            if set(records) != set(level_order):
                missing = sorted(set(level_order) - set(records))
                raise RuntimeError(f"Incomplete level trajectory for {scene_name}: missing={missing}")
            pred_curve = torch.tensor([records[level][0] for level in level_order])
            target_curve = torch.tensor([records[level][1] for level in level_order])
            for key, value in trajectory_metrics(pred_curve, target_curve).items():
                trajectory_records[key].append(value)

        violations = 0
        transitions = 0
        strict_scene_passes = 0
        zero_drifts: list[float] = []
        alphas = torch.linspace(-1.0, 1.0, args.dense_steps, device=device)
        unique_indices = list(range(0, len(dataset), len(LEVELS)))
        for index in tqdm(unique_indices, desc="dense alpha sweep"):
            item = dataset[index]
            x = item["in_image"].unsqueeze(0).to(device)
            means = []
            for alpha in alphas:
                out = model(x, alpha.view(1))["output"]
                means.append(mean_log_luminance(out)[0])
            means_tensor = torch.stack(means)
            step_violations = int((means_tensor[1:] + 1e-7 < means_tensor[:-1]).sum())
            violations += step_violations
            transitions += args.dense_steps - 1
            strict_scene_passes += int(step_violations == 0)
            zero = model(x, torch.zeros(1, device=device))["output"]
            base = model.forward_baseline(x)["output"]
            zero_drifts.append(float((zero - base).abs().max()))

        latency_item = dataset[0]["in_image"].unsqueeze(0).to(device)
        latency = _latency_ms(model, latency_item, device, args.latency_warmup, args.latency_runs)

    summary = {
        "control_method": args.control_method,
        "scene_count": len(dataset.scene_names),
        "parameter_report": model.parameter_report(),
        "per_level": {level: _mean_dict(metrics) for level, metrics in per_level.items()},
        "trajectory": _mean_dict(trajectory_records),
        "dense_monotonic_violation_rate": violations / max(transitions, 1),
        "dense_monotonic_scene_pass_rate": strict_scene_passes / max(len(unique_indices), 1),
        "alpha_zero_max_drift": max(zero_drifts, default=0.0),
        "gain_factor": summarize_values(gain_values),
        "gtm_parameters": {
            f"parameter_{index}": summarize_values(values) for index, values in enumerate(gtm_values)
        },
        "latency_ms_per_image": latency,
        "evaluation_seconds": time.perf_counter() - start,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
