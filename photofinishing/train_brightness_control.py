#!/usr/bin/env python3
"""Train one Gain/GTM brightness-control adapter on a frozen baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .control_data import BrightnessPairDataset, LEVELS
    from .control_losses import BrightnessControlLoss
    from .controlled_photofinishing import (
        ControlledLuminancePhotofinishing,
        find_target_params_per_head,
    )
except ImportError:
    from control_data import BrightnessPairDataset, LEVELS
    from control_losses import BrightnessControlLoss
    from controlled_photofinishing import (
        ControlledLuminancePhotofinishing,
        find_target_params_per_head,
    )

CONTROL_METHODS = ("param_residual", "parallel_adapter", "film", "dual_lora")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-training-dir", required=True, type=Path)
    parser.add_argument("--gt-training-root", required=True, type=Path)
    parser.add_argument("--input-validation-dir", required=True, type=Path)
    parser.add_argument("--gt-validation-root", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--control-method", choices=CONTROL_METHODS, required=True)
    parser.add_argument("--parameter-budget", type=int, default=2048,
                        help="Desired total trainable adapter parameters across Gain and GTM.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-luma-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=0.2)
    parser.add_argument("--monotonic-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--validation-max-batches", type=int, default=0)
    return parser


def _import_baseline(device: torch.device):
    try:
        from photofinishing_model import PhotofinishingModule
    except ImportError:
        from photofinishing.photofinishing_model import PhotofinishingModule
    return PhotofinishingModule(device=device, use_3d_lut=False)


def _load_checkpoint_with_report(baseline: torch.nn.Module, path: Path, *,
                                 device: torch.device, allow_partial: bool):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = baseline.load_state_dict(state, strict=not allow_partial)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if allow_partial:
        critical_missing = [
            key for key in missing
            if key.startswith(("_gain_net", "_gtm_net", "_ltm_net"))
        ]
        if critical_missing:
            raise RuntimeError(f"Critical baseline weights are missing: {critical_missing[:20]}")
    for parameter in baseline.parameters():
        parameter.requires_grad = False
    baseline.eval()
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _make_model(args: argparse.Namespace, device: torch.device):
    baseline = _import_baseline(device)
    load_report = _load_checkpoint_with_report(
        baseline, args.baseline_checkpoint, device=device,
        allow_partial=args.allow_partial_checkpoint)
    per_head_request, realized = find_target_params_per_head(
        baseline, args.control_method, args.parameter_budget)
    model = ControlledLuminancePhotofinishing(
        baseline, method=args.control_method,
        target_params_per_head=per_head_request).to(device)
    actual = model.trainable_parameter_count()
    if actual != realized:
        raise RuntimeError(f"Parameter audit mismatch: model={actual}, search={realized}")
    return model, per_head_request, load_report


def _run_epoch(model, loader, loss_fn, device, *, optimizer, scaler,
               amp: bool, max_batches: int = 0):
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "log_luma": 0.0, "gradient": 0.0,
            "monotonic": 0.0, "anchor": 0.0}
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, leave=False, desc="train" if training else "val")
        for batch_idx, batch in enumerate(progress):
            if max_batches and batch_idx >= max_batches:
                break
            image = batch["input"].to(device, non_blocking=True)
            target_low = batch["target_low"].to(device, non_blocking=True)
            target_high = batch["target_high"].to(device, non_blocking=True)
            target_anchor = batch["target_anchor"].to(device, non_blocking=True)
            alpha_low = batch["alpha_low"].to(device, non_blocking=True)
            alpha_high = batch["alpha_high"].to(device, non_blocking=True)
            alpha_zero = torch.zeros_like(alpha_low)

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                enabled=amp and device.type == "cuda"):
                pred_low = model(image, alpha_low, training_mode=True)["output"]
                pred_high = model(image, alpha_high, training_mode=True)["output"]
                pred_anchor = model(image, alpha_zero, training_mode=True)["output"]
                loss, details = loss_fn(
                    pred_low, pred_high, target_low, target_high,
                    pred_anchor, target_anchor)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            count += 1
            for key in sums:
                sums[key] += details[key]
            progress.set_postfix(loss=f"{details['total']:.4f}")
    if count == 0:
        raise RuntimeError("No batches were processed")
    return {key: value / count for key, value in sums.items()}


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.parameter_budget <= 0:
        raise ValueError("parameter-budget must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = BrightnessPairDataset(
        args.input_training_dir, args.gt_training_root,
        image_size=args.image_size, seed=args.seed, geometric_aug=True)
    val_set = BrightnessPairDataset(
        args.input_validation_dir, args.gt_validation_root,
        image_size=args.image_size, seed=args.seed + 100_000,
        geometric_aug=False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False)

    model, per_head_request, load_report = _make_model(args, device)
    optimizer = torch.optim.AdamW(
        list(model.control_parameters()), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda")
    loss_fn = BrightnessControlLoss(
        log_luma_weight=args.log_luma_weight,
        gradient_weight=args.gradient_weight,
        monotonic_weight=args.monotonic_weight,
        anchor_weight=args.anchor_weight)

    metadata = {
        "control_method": args.control_method,
        "desired_parameter_budget": args.parameter_budget,
        "per_head_request": per_head_request,
        "actual_trainable_parameters": model.trainable_parameter_count(),
        "levels": [{"name": name, "alpha": alpha} for name, alpha in LEVELS],
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "checkpoint_load_report": load_report,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        train_metrics = _run_epoch(
            model, train_loader, loss_fn, device,
            optimizer=optimizer, scaler=scaler, amp=args.amp)
        val_metrics = _run_epoch(
            model, val_loader, loss_fn, device,
            optimizer=None, scaler=None, amp=args.amp,
            max_batches=args.validation_max_batches)
        row = {"epoch": epoch + 1, "train": train_metrics,
               "validation": val_metrics}
        history.append(row)
        print(json.dumps(row))
        payload = {
            "adapter_state_dict": {
                key: value.cpu() for key, value in model.state_dict().items()
                if key.startswith(("gain_control.", "gtm_control."))
            },
            "control_method": args.control_method,
            "per_head_request": per_head_request,
            "actual_trainable_parameters": model.trainable_parameter_count(),
            "epoch": epoch + 1,
        }
        torch.save(payload, args.output_dir / "last_adapter.pth")
        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]
            payload["validation"] = val_metrics
            torch.save(payload, args.output_dir / "best_adapter.pth")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
