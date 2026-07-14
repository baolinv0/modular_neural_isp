from __future__ import annotations

import torch

from adjustTM.transfer import linear_luminance, linear_to_srgb, srgb_to_linear


def exposure_transform(srgb: torch.Tensor, ev: float | torch.Tensor) -> torch.Tensor:
    if isinstance(ev, (int, float)) and float(ev) == 0.0:
        return srgb.clone()
    linear = srgb_to_linear(srgb)
    factor = torch.as_tensor(ev, dtype=linear.dtype, device=linear.device)
    while factor.ndim < linear.ndim:
        factor = factor.unsqueeze(-1)
    return linear_to_srgb((linear * torch.pow(torch.tensor(2.0, dtype=linear.dtype, device=linear.device), factor)).clamp(0.0, 1.0))


def luminance_gamma_transform(srgb: torch.Tensor, gamma: float | torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if isinstance(gamma, (int, float)) and float(gamma) == 1.0:
        return srgb.clone()
    linear = srgb_to_linear(srgb)
    luminance = linear_luminance(linear).clamp_min(eps)
    parameter = torch.as_tensor(gamma, dtype=linear.dtype, device=linear.device)
    while parameter.ndim < luminance.ndim:
        parameter = parameter.unsqueeze(-1)
    target_luminance = luminance.pow(parameter)
    scale = target_luminance / luminance
    transformed = (linear * scale).clamp(0.0, 1.0)
    return linear_to_srgb(transformed)
