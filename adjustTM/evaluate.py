from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CONTROL_METHODS
from .dataset import MultiLevelDataset
from .model import ControlledBrightnessISP, load_baseline_checkpoint, load_control_checkpoint
from .transfer import linear_luminance, srgb_to_linear


def log_luma_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y = linear_luminance(srgb_to_linear(pred)).clamp_min(1e-4).log()
    target_y = linear_luminance(srgb_to_linear(target)).clamp_min(1e-4).log()
    return (pred_y - target_y).abs().mean()


def luminance_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y = linear_luminance(srgb_to_linear(pred))
    target_y = linear_luminance(srgb_to_linear(target))
    mse = (pred_y - target_y).square().mean().clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = MultiLevelDataset(args.input_dir, args.gt_root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = ControlledBrightnessISP(args.control_method, device=device).to(device)
    load_baseline_checkpoint(model.baseline, args.baseline_checkpoint, map_location=device)
    load_control_checkpoint(model, args.control_checkpoint, map_location=device)
    model.eval()

    per_level: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate levels"):
            x = batch["in_image"].to(device)
            gt = batch["gt_image"].to(device)
            alpha = batch["alpha"].to(device)
            pred = model(x, alpha)["output"]
            for idx, level in enumerate(batch["level_name"]):
                per_level[level]["log_luma_mae"].append(float(log_luma_mae(pred[idx:idx+1], gt[idx:idx+1])))
                per_level[level]["luma_psnr"].append(float(luminance_psnr(pred[idx:idx+1], gt[idx:idx+1])))
                per_level[level]["clip_ratio"].append(float((pred[idx:idx+1] >= 0.999).float().mean()))
                pred_y = linear_luminance(srgb_to_linear(pred[idx:idx+1]))
                per_level[level]["deep_shadow_ratio"].append(float((pred_y <= 0.01).float().mean()))

        unique_indices = list(range(0, len(dataset), 9))
        violations = 0
        transitions = 0
        zero_drifts = []
        alphas = torch.linspace(-1.0, 1.0, args.dense_steps, device=device)
        for index in tqdm(unique_indices, desc="dense alpha sweep"):
            item = dataset[index]
            x = item["in_image"].unsqueeze(0).to(device)
            means = []
            for alpha in alphas:
                out = model(x, alpha.view(1))["output"]
                means.append(linear_luminance(srgb_to_linear(out)).mean())
            means_tensor = torch.stack(means)
            violations += int((means_tensor[1:] + 1e-7 < means_tensor[:-1]).sum())
            transitions += args.dense_steps - 1
            zero = model(x, torch.zeros(1, device=device))["output"]
            base = model.forward_baseline(x)["output"]
            zero_drifts.append(float((zero - base).abs().max()))

    summary = {
        "control_method": args.control_method,
        "parameter_report": model.parameter_report(),
        "per_level": {
            level: {metric: sum(values) / max(len(values), 1) for metric, values in metrics.items()}
            for level, metrics in per_level.items()
        },
        "dense_monotonic_violation_rate": violations / max(transitions, 1),
        "alpha_zero_max_drift": max(zero_drifts, default=0.0),
        "evaluation_seconds": time.perf_counter() - start,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
