from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .transfer import linear_luminance, srgb_to_linear


def _log_luminance(srgb: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.log(linear_luminance(srgb_to_linear(srgb)).clamp_min(eps))


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps).mean()


def log_luminance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _charbonnier(_log_luminance(pred) - _log_luminance(target))


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y = linear_luminance(srgb_to_linear(pred))
    target_y = linear_luminance(srgb_to_linear(target))
    pred_dx = pred_y[..., :, 1:] - pred_y[..., :, :-1]
    target_dx = target_y[..., :, 1:] - target_y[..., :, :-1]
    pred_dy = pred_y[..., 1:, :] - pred_y[..., :-1, :]
    target_dy = target_y[..., 1:, :] - target_y[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def monotonic_hinge_loss(
    pred_low: torch.Tensor,
    pred_high: torch.Tensor,
    alpha_low: torch.Tensor,
    alpha_high: torch.Tensor,
    margin_per_alpha: float = 0.01,
) -> torch.Tensor:
    if torch.any(alpha_high <= alpha_low):
        raise ValueError("Expected alpha_low < alpha_high")
    low_mean = _log_luminance(pred_low).mean(dim=(1, 2, 3))
    high_mean = _log_luminance(pred_high).mean(dim=(1, 2, 3))
    margin = margin_per_alpha * (alpha_high - alpha_low).to(low_mean)
    return torch.relu(margin - (high_mean - low_mean)).mean()


class BrightnessOnlyLoss(nn.Module):
    def __init__(
        self,
        lambda_grad: float = 0.2,
        lambda_mono: float = 0.1,
        lambda_zero: float = 0.5,
        margin_per_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_mono = lambda_mono
        self.lambda_zero = lambda_zero
        self.margin_per_alpha = margin_per_alpha

    def forward(
        self,
        *,
        pred_low: torch.Tensor,
        pred_high: torch.Tensor,
        gt_low: torch.Tensor,
        gt_high: torch.Tensor,
        alpha_low: torch.Tensor,
        alpha_high: torch.Tensor,
        baseline_zero: torch.Tensor | None,
        pred_zero: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        reconstruction = 0.5 * (
            log_luminance_loss(pred_low, gt_low) + log_luminance_loss(pred_high, gt_high)
        )
        gradients = 0.5 * (gradient_loss(pred_low, gt_low) + gradient_loss(pred_high, gt_high))
        monotonic = monotonic_hinge_loss(
            pred_low,
            pred_high,
            alpha_low,
            alpha_high,
            margin_per_alpha=self.margin_per_alpha,
        )
        if baseline_zero is None or pred_zero is None:
            zero_anchor = reconstruction.new_zeros(())
        else:
            zero_anchor = F.l1_loss(pred_zero, baseline_zero)
        total = (
            reconstruction
            + self.lambda_grad * gradients
            + self.lambda_mono * monotonic
            + self.lambda_zero * zero_anchor
        )
        return {
            "total": total,
            "log_luminance": reconstruction,
            "gradient": gradients,
            "monotonic": monotonic,
            "zero_anchor": zero_anchor,
        }
