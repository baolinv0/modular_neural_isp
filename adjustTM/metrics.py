from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from .transfer import linear_luminance, srgb_to_linear


def mean_log_luminance(srgb: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    linear = srgb_to_linear(srgb)
    return linear_luminance(linear).clamp_min(eps).log().mean(dim=(1, 2, 3))


def log_luma_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y = linear_luminance(srgb_to_linear(pred)).clamp_min(1e-4).log()
    target_y = linear_luminance(srgb_to_linear(target)).clamp_min(1e-4).log()
    return (pred_y - target_y).abs().mean(dim=(1, 2, 3))


def luminance_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y = linear_luminance(srgb_to_linear(pred))
    target_y = linear_luminance(srgb_to_linear(target))
    mse = (pred_y - target_y).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def luminance_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pred_y = linear_luminance(srgb_to_linear(pred))
    target_y = linear_luminance(srgb_to_linear(target))
    spatial_min = min(pred_y.shape[-2:])
    window = min(window_size, spatial_min)
    if window % 2 == 0:
        window -= 1
    window = max(window, 1)
    padding = window // 2

    def pool(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=window, stride=1, padding=padding)

    mu_x = pool(pred_y)
    mu_y = pool(target_y)
    sigma_x = pool(pred_y * pred_y) - mu_x * mu_x
    sigma_y = pool(target_y * target_y) - mu_y * mu_y
    sigma_xy = pool(pred_y * target_y) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    score = numerator / denominator.clamp_min(1e-12)
    return score.mean(dim=(1, 2, 3)).clamp(-1.0, 1.0)


def chroma_rg_mae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_linear = srgb_to_linear(pred)
    target_linear = srgb_to_linear(target)
    pred_sum = pred_linear.sum(dim=1, keepdim=True).clamp_min(eps)
    target_sum = target_linear.sum(dim=1, keepdim=True).clamp_min(eps)
    pred_rg = pred_linear[:, :2] / pred_sum
    target_rg = target_linear[:, :2] / target_sum
    return (pred_rg - target_rg).abs().mean(dim=(1, 2, 3))


def clipping_ratio(srgb: torch.Tensor, threshold: float = 0.999) -> torch.Tensor:
    return (srgb >= threshold).float().mean(dim=(1, 2, 3))


def deep_shadow_ratio(srgb: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    y = linear_luminance(srgb_to_linear(srgb))
    return (y <= threshold).float().mean(dim=(1, 2, 3))


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().flatten().to(dtype=torch.float64, device="cpu")
    count = values.numel()
    order = sorted(range(count), key=lambda index: (float(values[index]), index))
    ranks = torch.empty(count, dtype=torch.float64)
    start = 0
    while start < count:
        end = start + 1
        while end < count and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() != second.numel() or first.numel() < 2:
        raise ValueError("Spearman correlation requires equal vectors with at least two values")
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = torch.sqrt(first_centered.square().sum() * second_centered.square().sum())
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(first_rank, second_rank) else 0.0
    return float((first_centered * second_centered).sum() / denominator)


def trajectory_metrics(pred_curve: torch.Tensor, target_curve: torch.Tensor) -> dict[str, float]:
    pred = pred_curve.detach().flatten().to(dtype=torch.float64, device="cpu")
    target = target_curve.detach().flatten().to(dtype=torch.float64, device="cpu")
    if pred.shape != target.shape or pred.numel() < 2:
        raise ValueError("trajectory curves must have the same shape and at least two levels")
    pred_steps = pred[1:] - pred[:-1]
    target_steps = target[1:] - target[:-1]
    return {
        "curve_mae": float((pred - target).abs().mean()),
        "adjacent_step_mae": float((pred_steps - target_steps).abs().mean()),
        "endpoint_range_error": float(abs((pred[-1] - pred[0]) - (target[-1] - target[0]))),
        "spearman": spearman_correlation(pred, target),
        "monotonic_violation_rate": float((pred_steps < 0).double().mean()),
        "strictly_monotonic": float(bool(torch.all(pred_steps >= 0))),
    }


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {"min": math.nan, "median": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "min": float(tensor.min()),
        "median": float(tensor.median()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }
