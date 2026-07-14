from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CONTROL_METHODS, DEFAULT_PARAMETER_TOLERANCE, DEFAULT_TARGET_CONTROL_PARAMS
from .dataset import MultiLevelDataset, MultiLevelPairDataset, discover_scene_names
from .losses import BrightnessOnlyLoss
from .manifest import create_or_load_split_manifest, write_sample_index
from .metrics import chroma_rg_mae, log_luma_mae, luminance_psnr, luminance_ssim
from .model import ControlledBrightnessISP, load_baseline_checkpoint, load_control_checkpoint
from .sampler import LevelBalancedBatchSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def enforce_parameter_budget(report: dict, target: int, tolerance: float, allow_mismatch: bool) -> None:
    trainable = int(report["trainable"])
    relative_error = abs(trainable - target) / max(target, 1)
    if relative_error > tolerance and not allow_mismatch:
        raise RuntimeError(
            f"Control parameter count {trainable} differs from target {target} by "
            f"{relative_error:.1%}, exceeding tolerance {tolerance:.1%}. "
            "Use --allow-parameter-mismatch only for intentional ablations."
        )


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    *,
    model: ControlledBrightnessISP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_metric: float,
    bad_epochs: int,
    report: dict,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "version": 2,
        "control_method": model.control_method,
        "control_state_dict": model.control_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": _capture_rng_state(),
        "epoch": epoch,
        "best_metric": best_metric,
        "bad_epochs": bad_epochs,
        "parameter_report": report,
        "args": vars(args),
    }


def _resume_training(
    path: str | Path,
    *,
    model: ControlledBrightnessISP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = load_control_checkpoint(model, path, map_location=device)
    for key in ("optimizer_state_dict", "scheduler_state_dict", "epoch"):
        if key not in checkpoint:
            raise RuntimeError(f"Resume checkpoint is missing {key}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint["epoch"]), float(checkpoint.get("best_metric", float("inf"))), int(
        checkpoint.get("bad_epochs", 0)
    )


def validate(
    model: ControlledBrightnessISP,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    totals = {"log_luma_mae": 0.0, "luma_psnr": 0.0, "luma_ssim": 0.0, "chroma_rg_mae": 0.0}
    sample_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            x = batch["in_image"].to(device, non_blocking=True)
            gt = batch["gt_image"].to(device, non_blocking=True)
            alpha = batch["alpha"].to(device, non_blocking=True)
            pred = model(x, alpha)["output"]
            metrics = {
                "log_luma_mae": log_luma_mae(pred, gt),
                "luma_psnr": luminance_psnr(pred, gt),
                "luma_ssim": luminance_ssim(pred, gt),
                "chroma_rg_mae": chroma_rg_mae(pred, gt),
            }
            batch_size = x.shape[0]
            sample_count += batch_size
            for key, values in metrics.items():
                totals[key] += float(values.sum())
    if sample_count == 0:
        return {}
    return {key: value / sample_count for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train brightness-control adapters on frozen Gain/GTM modules")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--control-method", choices=CONTROL_METHODS, required=True)
    parser.add_argument("--output-dir", default="adjustTM/checkpoints")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--val-batch-size", type=int, default=8)
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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")

    output_root = Path(args.output_dir)
    method_dir = output_root / args.control_method
    method_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else output_root / "manifests"
    split_path = Path(args.split_manifest) if args.split_manifest else manifest_dir / f"split_seed_{args.seed}.json"

    scene_names = discover_scene_names(args.input_dir)
    split = create_or_load_split_manifest(scene_names, split_path, val_fraction=args.val_fraction, seed=args.seed)
    train_dataset = MultiLevelPairDataset(
        input_dir=args.input_dir,
        gt_root=args.gt_root,
        image_size=args.image_size,
        geometric_aug=args.geometric_aug,
        seed=args.seed,
        scene_names=split["train"],
    )
    val_dataset = (
        MultiLevelDataset(
            input_dir=args.input_dir,
            gt_root=args.gt_root,
            image_size=args.image_size,
            scene_names=split["val"],
        )
        if split["val"]
        else None
    )
    write_sample_index(train_dataset, manifest_dir / "train_sample_index.json")
    if val_dataset is not None:
        write_sample_index(val_dataset, manifest_dir / "val_sample_index.json")

    batch_sampler = LevelBalancedBatchSampler(train_dataset, batch_size=args.batch_size, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed + 1),
            persistent_workers=args.num_workers > 0,
        )
        if val_dataset is not None
        else None
    )

    model = ControlledBrightnessISP(control_method=args.control_method, device=device).to(device)
    load_baseline_checkpoint(model.baseline, args.baseline_checkpoint, map_location=device)
    model.freeze_baseline()
    model.assert_baseline_frozen()
    report = model.parameter_report()
    enforce_parameter_budget(report, args.target_control_params, args.parameter_tolerance, args.allow_parameter_mismatch)
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
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_metric = float("inf")
    bad_epochs = 0
    if args.resume:
        start_epoch, best_metric, bad_epochs = _resume_training(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    config = {
        **vars(args),
        "parameter_report": report,
        "baseline_sha256": file_sha256(args.baseline_checkpoint),
        "split_manifest": str(split_path),
        "split": split,
    }
    (method_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        model.train()
        model.assert_baseline_frozen()
        running = {key: 0.0 for key in ("total", "log_luminance", "gradient", "monotonic", "zero_anchor")}
        completed_steps = 0
        start = time.perf_counter()
        progress = tqdm(train_loader, desc=f"{args.control_method} epoch {epoch + 1}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            x = batch["in_image"].to(device, non_blocking=True)
            gt_low = batch["gt_low"].to(device, non_blocking=True)
            gt_high = batch["gt_high"].to(device, non_blocking=True)
            alpha_low = batch["alpha_low"].to(device, non_blocking=True)
            alpha_high = batch["alpha_high"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
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
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index}: {losses}")
            scaler.scale(losses["total"]).backward()
            if any(parameter.grad is not None for parameter in model.baseline.parameters()):
                raise RuntimeError("Frozen baseline received parameter gradients")
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            completed_steps += 1
            for key in running:
                running[key] += float(losses[key].detach())
            progress.set_postfix(loss=f"{float(losses['total'].detach()):.4f}")

        if completed_steps == 0:
            raise RuntimeError("No training batches were completed")
        scheduler.step()
        epoch_log = {f"train_{key}": value / completed_steps for key, value in running.items()}
        validation = validate(model, val_loader, device, max_batches=args.max_val_batches) if val_loader else {}
        epoch_log.update({f"val_{key}": value for key, value in validation.items()})
        selection_metric = validation.get("log_luma_mae", epoch_log["train_total"])
        improved = selection_metric < best_metric
        if improved:
            best_metric = selection_metric
            bad_epochs = 0
        else:
            bad_epochs += 1
        epoch_log.update(
            {
                "epoch": epoch + 1,
                "seconds": time.perf_counter() - start,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selection_metric": selection_metric,
                "best_metric": best_metric,
            }
        )
        with (method_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_log) + "\n")

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            best_metric=best_metric,
            bad_epochs=bad_epochs,
            report=report,
            args=args,
        )
        _atomic_torch_save(payload, method_dir / "control_last.pth")
        if improved:
            _atomic_torch_save(payload, method_dir / "control_best.pth")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _atomic_torch_save(payload, method_dir / f"control_epoch_{epoch + 1:03d}.pth")
        if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            break


if __name__ == "__main__":
    main()
