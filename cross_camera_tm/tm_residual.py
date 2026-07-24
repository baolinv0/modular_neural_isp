from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .adapters import LUMA_WEIGHTS, _piecewise_curve


@dataclass(frozen=True)
class TMResidualOutput:
    image: torch.Tensor
    curve_y: torch.Tensor
    log_gain_map: torch.Tensor


def _gaussian_smooth(value: torch.Tensor) -> torch.Tensor:
    kernel = value.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0))
    kernel = torch.outer(kernel, kernel)
    kernel = (kernel / kernel.sum()).view(1, 1, 5, 5)
    padded = functional.pad(value, (2, 2, 2, 2), mode="replicate")
    return functional.conv2d(padded, kernel)


class TMResidualAdapter(nn.Module):
    """Luminance-only full-image residual with 69 scalar parameters by default."""

    def __init__(self, *, curve_points: int = 6, grid_size: int = 8, max_log_gain: float = 0.20):
        super().__init__()
        if curve_points < 3 or grid_size < 2:
            raise ValueError("invalid residual adapter dimensions")
        self.curve_points = curve_points
        self.grid_size = grid_size
        self.max_log_gain = float(max_log_gain)
        self.curve_logits = nn.Parameter(torch.zeros(curve_points - 1))
        self.log_gain_grid = nn.Parameter(torch.zeros(1, 1, grid_size, grid_size))

    def forward(self, image: torch.Tensor, *, soft_roi: torch.Tensor | None = None) -> TMResidualOutput:
        if image.ndim != 4 or image.shape[1] != 3 or not torch.isfinite(image).all():
            raise ValueError("TM residual input must be finite [B,3,H,W]")
        increments = functional.softplus(self.curve_logits).unsqueeze(0).expand(image.shape[0], -1)
        curve_y = torch.cat(
            (torch.zeros_like(increments[:, :1]), torch.cumsum(increments, dim=1)), dim=1
        )
        curve_y = curve_y / curve_y[:, -1:].clamp_min(1e-8)
        weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        luma = (image * weights).sum(dim=1, keepdim=True)
        curved = _piecewise_curve(luma, curve_y)
        grid = torch.tanh(self.log_gain_grid) * self.max_log_gain
        grid = grid.expand(image.shape[0], -1, -1, -1)
        log_gain = functional.interpolate(
            grid, size=image.shape[-2:], mode="bicubic", align_corners=False
        )
        log_gain = _gaussian_smooth(log_gain).clamp(-self.max_log_gain, self.max_log_gain)
        if soft_roi is not None:
            if soft_roi.shape != luma.shape:
                raise ValueError("soft_roi must have shape [B,1,H,W]")
            log_gain = log_gain * soft_roi.to(image.dtype).clamp(0.0, 1.0)
        corrected_luma = (curved * torch.exp(log_gain)).clamp(0.0, 1.0)
        corrected = (image * (corrected_luma / luma.clamp_min(1e-6))).clamp(0.0, 1.0)
        return TMResidualOutput(image=corrected, curve_y=curve_y, log_gain_map=log_gain)


def source_identity_loss(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if output.shape != source.shape:
        raise ValueError("source identity tensors must have matching shapes")
    return (output - source).abs().mean()


def source_gt_non_regression_loss(
    adapted_error: torch.Tensor, baseline_error: torch.Tensor, *, tolerance: float
) -> torch.Tensor:
    return (adapted_error - baseline_error - tolerance).clamp_min(0.0).mean()
