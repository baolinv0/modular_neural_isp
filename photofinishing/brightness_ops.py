"""Brightness-only image operations for controllable photofinishing."""
from __future__ import annotations

import torch


def linear_rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB tensor, got {tuple(rgb.shape)}")
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def srgb_oetf(linear: torch.Tensor) -> torch.Tensor:
    linear = linear.clamp(0.0, 1.0)
    return torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * torch.pow(linear.clamp_min(0.0031308), 1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


def srgb_eotf(srgb: torch.Tensor) -> torch.Tensor:
    srgb = srgb.clamp(0.0, 1.0)
    return torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
    ).clamp(0.0, 1.0)


def apply_luminance_scale(
    linear_rgb: torch.Tensor,
    target_luminance: torch.Tensor,
    *,
    scale_min: float = 0.125,
    scale_max: float = 8.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply one shared RGB scale per pixel, preserving chromaticity before clipping."""
    source_luminance = linear_rgb_to_luminance(linear_rgb)
    scale = (target_luminance / source_luminance.clamp_min(eps)).clamp(scale_min, scale_max)
    return (linear_rgb * scale).clamp(0.0, 1.0)


def log_luminance_from_srgb(srgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log2(linear_rgb_to_luminance(srgb_eotf(srgb)).clamp_min(eps))


def luminance_gradient(log_luminance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if log_luminance.ndim != 4 or log_luminance.shape[1] != 1:
        raise ValueError("Expected Bx1xHxW log-luminance tensor")
    dx = log_luminance[..., :, 1:] - log_luminance[..., :, :-1]
    dy = log_luminance[..., 1:, :] - log_luminance[..., :-1, :]
    return dx, dy
