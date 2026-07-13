from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import (
    CONTROL_METHODS,
    DEFAULT_PARAMETER_TOLERANCE,
    DEFAULT_TARGET_CONTROL_PARAMS,
)
from .dataset import MultiLevelPairDataset
from .losses import BrightnessOnlyLoss
from .model import ControlledBrightnessISP, load_baseline_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zero_anchor_outputs(
    model: ControlledBrightnessISP,
    x: torch.Tensor,
    alpha_low: torch.Tensor,
    alpha_high: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mask = torch.isclose(alpha_low, torch.zeros_like(alpha_low)) | torch.isclose(
        alpha_high, torch.zeros_like(alpha_high)
    )
    if not torch.any(mask):
        return None, None
    x_zero = x[mask]
    alpha_zero = torch.zeros(x_zero.shape[0], device=x.device, dtype=x.dtype)
    pred_zero = model(x_zero, alpha_zero, training_mode=True)["output"]
    with torch.no_grad():
        baseline_zero = model.forward_baseline(x_zero, training_mode=True)["output"]
    return baseline_zero, pred_zero


def enforce_parameter_budget(
    report: dict,
    target: int,
    tolerance: float,
    allow_mismatch: bool,
) -> None:
    trainable = int(report["trainable"])
    relative_error = abs(trainable - target) / max(target, 1)
    if relative_error > tolerance and not allow_mismatch:
        raise RuntimeError(
            f"Control parameter count {trainable} differs from target {target} by "
            f"{relative_error:.1%}, exceeding tolerance {tolerance:.1%}. "
            "Use --allow-parameter-mismatch only for intentional ablations."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train brightness-control adapters on frozen Gain/GTM modules")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--control-method", choices=CONTROL_METHODS, required=True)
    parser.add_argument("--output-dir", default="adjustTM/checkpoints")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-grad", type=float, default=0.2)
    parser.add_argument("--lambda-mono", type=float, default=0.1)
    parser.add_argument("--lambda-zero", type=float, default=0.5)
    parser.add_argument("--margin-per-alpha", type=float, default=0.01)
    parser.add_argument("--target-control-params", type=int, default=DEFAULT_TARGET_CONTROL_PARAMS)
    parser.add_argument("--parameter-tolerance", type=float, default=DEFAULT_PARAMETER_TOLERANCE)
    parser.add_argument("--allow-parameter-mismatch", action="store_true")
    parser.add_argument("--geometric-aug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset = MultiLevelPairDataset(
        input_dir=args.input_dir,
        gt_root=args.gt_root,
        image_size=args.image_size,
        geometric_aug=args.geometric_aug,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )

    model = ControlledBrightnessISP(control_method=args.control_method, device=device).to(device)
    load_baseline_checkpoint(model.baseline, args.baseline_checkpoint, map_location=device)
    model.freeze_baseline()
    report = model.parameter_report()
    enforce_parameter_budget(
        report,
        target=args.target_control_params,
        tolerance=args.parameter_tolerance,
        allow_mismatch=args.allow_parameter_mismatch,
    )
    trainable = list(model.control_parameters())
    if not trainable:
        raise RuntimeError("No trainable control parameters found")

    objective = BrightnessOnlyLoss(
        lambda_grad=args.lambda_grad,
        lambda_mono=args.lambda_mono,
        lambda_zero=args.lambda_zero,
        margin_per_alpha=args.margin_per_alpha,
    )
    optimizer = Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr / 100)

    output_dir = Path(args.output_dir) / args.control_method
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**vars(args), "parameter_report": report}, indent=2, default=str), encoding="utf-8"
    )

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running = {key: 0.0 for key in ("total", "log_luminance", "gradient", "monotonic", "zero_anchor")}
        start = time.perf_counter()
        progress = tqdm(loader, desc=f"{args.control_method} epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            x = batch["in_image"].to(device, non_blocking=True)
            gt_low = batch["gt_low"].to(device, non_blocking=True)
            gt_high = batch["gt_high"].to(device, non_blocking=True)
            alpha_low = batch["alpha_low"].to(device, non_blocking=True)
            alpha_high = batch["alpha_high"].to(device, non_blocking=True)

            pred_low = model(x, alpha_low, training_mode=True)
            pred_high = model(x, alpha_high, training_mode=True)
            baseline_zero, pred_zero = _zero_anchor_outputs(model, x, alpha_low, alpha_high)
            losses = objective(
                pred_low=pred_low["output"],
                pred_high=pred_high["output"],
                gt_low=gt_low,
                gt_high=gt_high,
                alpha_low=alpha_low,
                alpha_high=alpha_high,
                baseline_zero=baseline_zero,
                pred_zero=pred_zero,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for key in running:
                running[key] += float(losses[key].detach())
            progress.set_postfix(loss=f"{float(losses['total'].detach()):.4f}")
        scheduler.step()
        steps = max(len(loader), 1)
        epoch_log = {key: value / steps for key, value in running.items()}
        epoch_log.update({"epoch": epoch + 1, "seconds": time.perf_counter() - start})
        with (output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_log) + "\n")
        torch.save(
            {
                "control_method": args.control_method,
                "control_state_dict": model.control_state_dict(),
                "epoch": epoch + 1,
                "parameter_report": report,
                "args": vars(args),
            },
            output_dir / f"control_epoch_{epoch + 1:03d}.pth",
        )


if __name__ == "__main__":
    main()
