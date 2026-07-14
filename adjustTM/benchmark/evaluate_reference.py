from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from pathlib import Path
from typing import Mapping, Sequence

import torch

from adjustTM.constants import LEVELS
from adjustTM.transfer import linear_luminance, srgb_to_linear
from .image_io import read_srgb_png, resize_like
from .schemas import read_json, semantic_group


def _rgb_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def _ssim_global(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    mu_x = pred.mean(dim=dims)
    mu_y = target.mean(dim=dims)
    var_x = pred.var(dim=dims, unbiased=False)
    var_y = target.var(dim=dims, unbiased=False)
    cov = ((pred - mu_x[:, None, None, None]) * (target - mu_y[:, None, None, None])).mean(dim=dims)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)).clamp_min(1e-12)


def _chroma_rg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = srgb_to_linear(pred)
    t = srgb_to_linear(target)
    p_rg = p[:, :2] / p.sum(dim=1, keepdim=True).clamp_min(1e-6)
    t_rg = t[:, :2] / t.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (p_rg - t_rg).abs().mean(dim=(1, 2, 3))


def _gradient_mae(pred_y: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    pred_dx = pred_y[..., :, 1:] - pred_y[..., :, :-1]
    target_dx = target_y[..., :, 1:] - target_y[..., :, :-1]
    pred_dy = pred_y[..., 1:, :] - pred_y[..., :-1, :]
    target_dy = target_y[..., 1:, :] - target_y[..., :-1, :]
    return 0.5 * ((pred_dx - target_dx).abs().mean(dim=(1, 2, 3)) + (pred_dy - target_dy).abs().mean(dim=(1, 2, 3)))


def compute_reference_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_model=None) -> dict[str, float]:
    target = resize_like(target, pred)
    pred_linear = srgb_to_linear(pred)
    target_linear = srgb_to_linear(target)
    pred_y = linear_luminance(pred_linear)
    target_y = linear_luminance(target_linear)
    y_mse = (pred_y - target_y).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    log_mae = (pred_y.clamp_min(1e-4).log() - target_y.clamp_min(1e-4).log()).abs().mean(dim=(1, 2, 3))
    metrics = {
        "pred_mean_log_luma": float(pred_y.clamp_min(1e-4).log().mean()),
        "target_mean_log_luma": float(target_y.clamp_min(1e-4).log().mean()),
        "rgb_psnr": float(_rgb_psnr(pred, target).mean()),
        "rgb_ssim": float(_ssim_global(pred, target).mean()),
        "luma_psnr": float((-10.0 * torch.log10(y_mse)).mean()),
        "luma_ssim": float(_ssim_global(pred_y, target_y).mean()),
        "log_luma_mae": float(log_mae.mean()),
        "gradient_mae": float(_gradient_mae(pred_y, target_y).mean()),
        "chroma_rg_mae_to_gt": float(_chroma_rg(pred, target).mean()),
        "clip_ratio": float((pred >= 0.999).float().mean()),
        "deep_shadow_ratio": float((pred_y <= 0.01).float().mean()),
    }
    if lpips_model is not None:
        with torch.no_grad():
            metrics["lpips"] = float(lpips_model(pred * 2 - 1, target * 2 - 1).mean())
    return metrics


def evaluate_cached_outputs(
    *,
    manifest: Mapping[str, object],
    output_root: str | Path,
    methods: Sequence[str],
    levels: Sequence[str],
    device: str | torch.device,
    lpips_model=None,
) -> list[dict[str, object]]:
    output_root = Path(output_root)
    records: list[dict[str, object]] = []
    for scene in manifest["scenes"]:  # type: ignore[index]
        scene_id = str(scene["scene_id"])
        for level in levels:
            target = read_srgb_png(scene["gt"][level]["path"], device=device)
            for method in methods:
                pred_path = output_root / method / level / scene_id
                if not pred_path.is_file():
                    raise FileNotFoundError(pred_path)
                pred = read_srgb_png(pred_path, device=device)
                records.append({
                    "scene_id": scene_id,
                    "method": method,
                    "level": level,
                    "alpha": float(dict(LEVELS)[level]),
                    "semantic_group": semantic_group(level),
                    "metrics": compute_reference_metrics(pred, target, lpips_model=lpips_model),
                })
    return records


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    a, b = _rankdata(first), _rankdata(second)
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else float(np.array_equal(a, b))


def build_trajectory_records(records):
    grouped = defaultdict(dict)
    level_order = [name for name, _ in LEVELS]
    for record in records:
        grouped[(record["method"], record["scene_id"])][record["level"]] = (
            float(record["metrics"]["pred_mean_log_luma"]),
            float(record["metrics"]["target_mean_log_luma"]),
        )
    output = []
    for (method, scene_id), values in sorted(grouped.items()):
        missing = sorted(set(level_order) - set(values))
        if missing:
            raise RuntimeError(f"Incomplete trajectory for {method}/{scene_id}: {missing}")
        pred = np.asarray([values[level][0] for level in level_order], dtype=np.float64)
        target = np.asarray([values[level][1] for level in level_order], dtype=np.float64)
        pred_steps = np.diff(pred)
        target_steps = np.diff(target)
        metrics = {
            "curve_mae": float(np.mean(np.abs(pred - target))),
            "adjacent_step_mae": float(np.mean(np.abs(pred_steps - target_steps))),
            "endpoint_range_error": float(abs((pred[-1] - pred[0]) - (target[-1] - target[0]))),
            "spearman": _spearman(pred, target),
            "nine_level_violation_rate": float(np.mean(pred_steps < -1e-7)),
            "strictly_monotonic": float(np.all(pred_steps >= -1e-7)),
        }
        output.append({"method": method, "scene_id": scene_id, "metrics": metrics})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached outputs against semantic GT groups")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", default=[name for name, _ in LEVELS])
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lpips_model = None
    if args.lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()
    records = evaluate_cached_outputs(
        manifest=read_json(args.manifest), output_root=args.output_root, methods=args.methods,
        levels=args.levels, device=args.device, lpips_model=lpips_model
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    complete_levels = set(args.levels) == {name for name, _ in LEVELS}
    if args.trajectory_output or complete_levels:
        trajectory_output = Path(args.trajectory_output) if args.trajectory_output else output.with_name("trajectory_records.jsonl")
        with trajectory_output.open("w", encoding="utf-8") as handle:
            for record in build_trajectory_records(records):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
