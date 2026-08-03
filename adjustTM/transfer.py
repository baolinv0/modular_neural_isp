from __future__ import annotations

import torch


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Apply the standard sRGB OETF to linear RGB values in [0, 1]."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(x.clamp_min(0.0031308), 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Apply the inverse standard sRGB transfer function."""
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow((x + 0.055) / 1.055, 2.4),
    )


def linear_luminance(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB tensor, got {tuple(rgb.shape)}")
    coeffs = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (rgb * coeffs).sum(dim=1, keepdim=True)
