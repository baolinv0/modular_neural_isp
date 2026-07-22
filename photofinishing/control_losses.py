"""Brightness-only losses shared by all control variants."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .brightness_ops import log_luminance_from_srgb, luminance_gradient
except ImportError:
    from brightness_ops import log_luminance_from_srgb, luminance_gradient


def pairwise_monotonic_loss(pred_low: torch.Tensor, pred_high: torch.Tensor,
                            margin: float = 0.0) -> torch.Tensor:
    low_brightness = log_luminance_from_srgb(pred_low).mean(dim=(1, 2, 3))
    high_brightness = log_luminance_from_srgb(pred_high).mean(dim=(1, 2, 3))
    return F.relu(float(margin) - (high_brightness - low_brightness)).mean()


class BrightnessControlLoss(nn.Module):
    def __init__(self, *, log_luma_weight: float = 1.0,
                 gradient_weight: float = 0.2,
                 monotonic_weight: float = 0.1,
                 anchor_weight: float = 0.5,
                 monotonic_margin: float = 0.0) -> None:
        super().__init__()
        self.log_luma_weight = log_luma_weight
        self.gradient_weight = gradient_weight
        self.monotonic_weight = monotonic_weight
        self.anchor_weight = anchor_weight
        self.monotonic_margin = monotonic_margin

    @staticmethod
    def _reconstruction(pred: torch.Tensor, target: torch.Tensor):
        pred_log = log_luminance_from_srgb(pred)
        target_log = log_luminance_from_srgb(target)
        log_loss = F.l1_loss(pred_log, target_log)
        pred_dx, pred_dy = luminance_gradient(pred_log)
        target_dx, target_dy = luminance_gradient(target_log)
        grad_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
        return log_loss, grad_loss

    def forward(self, pred_low: torch.Tensor, pred_high: torch.Tensor,
                target_low: torch.Tensor, target_high: torch.Tensor,
                pred_anchor: torch.Tensor, target_anchor: torch.Tensor):
        log_low, grad_low = self._reconstruction(pred_low, target_low)
        log_high, grad_high = self._reconstruction(pred_high, target_high)
        log_loss = 0.5 * (log_low + log_high)
        gradient_loss = 0.5 * (grad_low + grad_high)
        monotonic_loss = pairwise_monotonic_loss(
            pred_low, pred_high, self.monotonic_margin)
        anchor_loss = F.l1_loss(pred_anchor, target_anchor)
        total = (
            self.log_luma_weight * log_loss
            + self.gradient_weight * gradient_loss
            + self.monotonic_weight * monotonic_loss
            + self.anchor_weight * anchor_loss
        )
        details = {
            "total": float(total.detach()),
            "log_luma": float(log_loss.detach()),
            "gradient": float(gradient_loss.detach()),
            "monotonic": float(monotonic_loss.detach()),
            "anchor": float(anchor_loss.detach()),
        }
        return total, details
