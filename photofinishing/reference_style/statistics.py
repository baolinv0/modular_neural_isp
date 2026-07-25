"""Differentiable distribution statistics for unaligned style comparison."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    matrix = rgb.new_tensor([
        [0.2990, 0.5870, 0.1140],
        [-0.168736, -0.331264, 0.5000],
        [0.5000, -0.418688, -0.081312],
    ])
    flat = rgb.permute(0, 2, 3, 1)
    ycbcr = torch.matmul(flat, matrix.t())
    return ycbcr.permute(0, 3, 1, 2)


def soft_histogram(x: torch.Tensor, bins: int = 32, sigma: float = 0.025,
                   min_value: float = 0.0, max_value: float = 1.0) -> torch.Tensor:
    if bins < 2 or sigma <= 0:
        raise ValueError("invalid histogram configuration")
    centers = torch.linspace(min_value, max_value, bins, device=x.device, dtype=x.dtype)
    flat = x.flatten(start_dim=1).unsqueeze(-1)
    weights = torch.exp(-0.5 * ((flat - centers) / sigma) ** 2)
    hist = weights.mean(dim=1)
    return hist / hist.sum(dim=-1, keepdim=True).clamp_min(_EPS)


def channel_moments(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(2, 3))
    std = x.std(dim=(2, 3), unbiased=False)
    return torch.cat([mean, std], dim=1)


def luminance_occupancy(y: torch.Tensor) -> torch.Tensor:
    shadows = torch.sigmoid((0.20 - y) / 0.03).mean(dim=(2, 3))
    highlights = torch.sigmoid((y - 0.80) / 0.03).mean(dim=(2, 3))
    midtones = (1.0 - shadows - highlights).clamp_min(0.0)
    return torch.cat([shadows, midtones, highlights], dim=1)


def global_contrast(y: torch.Tensor) -> torch.Tensor:
    return y.std(dim=(2, 3), unbiased=False)


def normalized_gradient_map(y: torch.Tensor) -> torch.Tensor:
    y = y / y.mean(dim=(2, 3), keepdim=True).clamp_min(_EPS)
    gx = y[..., :, 1:] - y[..., :, :-1]
    gy = y[..., 1:, :] - y[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.cat([gx, gy], dim=1)
