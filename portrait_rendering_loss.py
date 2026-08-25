"""Orthogonal portrait-rendering losses for brightness, skin color, tone, and face/background relation.

The module intentionally implements only four losses:

1. L_exp   : face exposure (DC luminance) difference in EV-like log2 luminance.
2. L_color : skin chromaticity difference in CIE Lab a*/b* only (L* excluded).
3. L_tone  : exposure-normalized face tone-distribution difference in log2 luminance.
4. L_fb    : face/background relative-exposure difference in EV-like log2 luminance.

Inputs are display-referred sRGB tensors in [0, 1]. Masks are shared between candidate and
reference and are expected to already represent corresponding semantic regions.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

QuantileMode = Literal["hard", "soft"]


def _check_image(name: str, image: Tensor) -> None:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B, 3, H, W], got {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError(f"{name} must be a floating-point tensor")


def _normalize_mask(name: str, mask: Tensor, image: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W]")
    if mask.shape[0] != image.shape[0] or mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"{name} spatial/batch shape {tuple(mask.shape)} does not match image {tuple(image.shape)}"
        )
    mask = mask.to(device=image.device, dtype=image.dtype)
    counts = (mask > 0).flatten(1).sum(dim=1)
    if torch.any(counts == 0):
        bad = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"{name} has no valid pixels for batch samples {bad}")
    return mask.clamp(0.0, 1.0)


def srgb_to_linear(rgb: Tensor) -> Tensor:
    """Convert display-referred sRGB in [0, 1] to linear RGB."""
    rgb = rgb.clamp(0.0, 1.0)
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(rgb: Tensor) -> Tensor:
    """Convert linear RGB in [0, 1] to display-referred sRGB."""
    rgb = rgb.clamp(0.0, 1.0)
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * torch.pow(rgb.clamp_min(1e-12), 1.0 / 2.4) - 0.055,
    )


def rgb_to_luminance(rgb: Tensor) -> Tensor:
    """Return relative linear-light luminance Y with shape [B, 1, H, W]."""
    _check_image("rgb", rgb)
    linear = srgb_to_linear(rgb)
    weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (linear * weights).sum(dim=1, keepdim=True)


def rgb_to_lab(rgb: Tensor) -> Tensor:
    """Differentiable sRGB-D65 -> CIE Lab conversion, output shape [B, 3, H, W]."""
    _check_image("rgb", rgb)
    linear = srgb_to_linear(rgb)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]

    # sRGB D65 -> XYZ, XYZ normalized to D65 reference white.
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    x = x / 0.95047
    z = z / 1.08883

    delta = 6.0 / 29.0
    delta3 = delta**3
    linear_slope = 1.0 / (3.0 * delta**2)
    linear_offset = 4.0 / 29.0

    def f(t: Tensor) -> Tensor:
        return torch.where(
            t > delta3,
            torch.pow(t.clamp_min(1e-12), 1.0 / 3.0),
            linear_slope * t + linear_offset,
        )

    fx, fy, fz = f(x), f(y), f(z)
    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return torch.cat([l_star, a_star, b_star], dim=1)


def _soft_quantile_1d(values: Tensor, q: float, temperature: float) -> Tensor:
    """Smooth rank-local interpolation over sorted values.

    Sorting is piecewise differentiable w.r.t. values. A Gaussian-like softmax over normalized
    ranks avoids the single-order-statistic gradient of a hard quantile while keeping the result
    local to the requested quantile.
    """
    if values.numel() == 1:
        return values[0]
    sorted_values = torch.sort(values).values
    ranks = torch.linspace(0.0, 1.0, sorted_values.numel(), device=values.device, dtype=values.dtype)
    tau = max(float(temperature), 1e-4)
    logits = -0.5 * ((ranks - float(q)) / tau) ** 2
    weights = torch.softmax(logits, dim=0)
    return torch.sum(weights * sorted_values)


def masked_quantile(
    values: Tensor,
    mask: Tensor,
    q: float,
    *,
    mode: QuantileMode = "hard",
    soft_temperature: float = 0.05,
) -> Tensor:
    """Per-sample masked quantile for single-channel maps.

    Returns a tensor of shape [B]. The implementation loops over batch samples because semantic
    masks naturally contain a variable number of valid pixels.
    """
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("values must have shape [B, 1, H, W] or [B, H, W]")
    if mask.shape != values.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match values shape {tuple(values.shape)}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if mode not in {"hard", "soft"}:
        raise ValueError(f"Unsupported quantile mode: {mode}")

    outputs: list[Tensor] = []
    valid_mask = mask > 0
    for batch_index in range(values.shape[0]):
        sample = values[batch_index][valid_mask[batch_index]]
        if sample.numel() == 0:
            raise ValueError(f"mask has no valid pixels for batch sample {batch_index}")
        if mode == "hard":
            outputs.append(torch.quantile(sample, q))
        else:
            outputs.append(_soft_quantile_1d(sample, q, soft_temperature))
    return torch.stack(outputs)


def face_exposure_loss(
    candidate: Tensor,
    reference: Tensor,
    face_mask: Tensor,
    *,
    eps: float = 1e-6,
    quantile_mode: QuantileMode = "hard",
    soft_quantile_temperature: float = 0.05,
) -> Tensor:
    """L_exp: absolute difference of face median log2 luminance (EV-like units)."""
    yc = rgb_to_luminance(candidate)
    yr = rgb_to_luminance(reference)
    log_c = torch.log2(yc + eps)
    log_r = torch.log2(yr + eps)
    med_c = masked_quantile(
        log_c,
        face_mask,
        0.5,
        mode=quantile_mode,
        soft_temperature=soft_quantile_temperature,
    )
    med_r = masked_quantile(
        log_r,
        face_mask,
        0.5,
        mode=quantile_mode,
        soft_temperature=soft_quantile_temperature,
    )
    return torch.mean(torch.abs(med_c - med_r))


def skin_chromaticity_loss(
    candidate: Tensor,
    reference: Tensor,
    skin_mask: Tensor,
    *,
    quantile_mode: QuantileMode = "hard",
    soft_quantile_temperature: float = 0.05,
) -> Tensor:
    """L_color: Euclidean difference in robust skin a*/b*; CIE Lab L* is intentionally excluded."""
    lab_c = rgb_to_lab(candidate)
    lab_r = rgb_to_lab(reference)

    stats_c: list[Tensor] = []
    stats_r: list[Tensor] = []
    for channel in (1, 2):
        stats_c.append(
            masked_quantile(
                lab_c[:, channel : channel + 1],
                skin_mask,
                0.5,
                mode=quantile_mode,
                soft_temperature=soft_quantile_temperature,
            )
        )
        stats_r.append(
            masked_quantile(
                lab_r[:, channel : channel + 1],
                skin_mask,
                0.5,
                mode=quantile_mode,
                soft_temperature=soft_quantile_temperature,
            )
        )

    da = stats_c[0] - stats_r[0]
    db = stats_c[1] - stats_r[1]
    return torch.mean(torch.sqrt(da * da + db * db + 1e-12) - 1e-6)


def face_tone_shape_loss(
    candidate: Tensor,
    reference: Tensor,
    face_mask: Tensor,
    *,
    eps: float = 1e-6,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.75, 0.90),
    quantile_mode: QuantileMode = "hard",
    soft_quantile_temperature: float = 0.05,
) -> Tensor:
    """L_tone: compare exposure-normalized face log-luminance quantiles.

    Subtracting each image's own face median removes the face luminance DC component, so this term
    responds primarily to tonal-shape changes such as shadow lift, contrast flattening, and
    highlight compression rather than global face exposure.
    """
    yc = rgb_to_luminance(candidate)
    yr = rgb_to_luminance(reference)
    log_c = torch.log2(yc + eps)
    log_r = torch.log2(yr + eps)

    med_c = masked_quantile(
        log_c,
        face_mask,
        0.5,
        mode=quantile_mode,
        soft_temperature=soft_quantile_temperature,
    )
    med_r = masked_quantile(
        log_r,
        face_mask,
        0.5,
        mode=quantile_mode,
        soft_temperature=soft_quantile_temperature,
    )
    centered_c = log_c - med_c.view(-1, 1, 1, 1)
    centered_r = log_r - med_r.view(-1, 1, 1, 1)

    differences: list[Tensor] = []
    for q in quantiles:
        qc = masked_quantile(
            centered_c,
            face_mask,
            q,
            mode=quantile_mode,
            soft_temperature=soft_quantile_temperature,
        )
        qr = masked_quantile(
            centered_r,
            face_mask,
            q,
            mode=quantile_mode,
            soft_temperature=soft_quantile_temperature,
        )
        differences.append(torch.abs(qc - qr))
    return torch.stack(differences, dim=0).mean()


def face_background_relation_loss(
    candidate: Tensor,
    reference: Tensor,
    face_mask: Tensor,
    background_mask: Tensor,
    *,
    eps: float = 1e-6,
    quantile_mode: QuantileMode = "hard",
    soft_quantile_temperature: float = 0.05,
) -> Tensor:
    """L_fb: difference in face-vs-background relative exposure (EV-like units)."""
    yc = torch.log2(rgb_to_luminance(candidate) + eps)
    yr = torch.log2(rgb_to_luminance(reference) + eps)

    def relation(log_y: Tensor) -> Tensor:
        face = masked_quantile(
            log_y,
            face_mask,
            0.5,
            mode=quantile_mode,
            soft_temperature=soft_quantile_temperature,
        )
        background = masked_quantile(
            log_y,
            background_mask,
            0.5,
            mode=quantile_mode,
            soft_temperature=soft_quantile_temperature,
        )
        return face - background

    return torch.mean(torch.abs(relation(yc) - relation(yr)))


class PortraitRenderingLoss(nn.Module):
    """Weighted combination of the four orthogonal portrait-rendering losses."""

    def __init__(
        self,
        *,
        w_exp: float = 0.25,
        w_color: float = 0.25,
        w_tone: float = 0.30,
        w_fb: float = 0.20,
        quantile_mode: QuantileMode = "hard",
        soft_quantile_temperature: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        weights = (w_exp, w_color, w_tone, w_fb)
        if any(weight < 0 for weight in weights):
            raise ValueError("Loss weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("At least one loss weight must be positive")
        if quantile_mode not in {"hard", "soft"}:
            raise ValueError(f"Unsupported quantile mode: {quantile_mode}")
        if soft_quantile_temperature <= 0:
            raise ValueError("soft_quantile_temperature must be > 0")

        self.w_exp = float(w_exp)
        self.w_color = float(w_color)
        self.w_tone = float(w_tone)
        self.w_fb = float(w_fb)
        self.quantile_mode = quantile_mode
        self.soft_quantile_temperature = float(soft_quantile_temperature)
        self.eps = float(eps)

    def forward(
        self,
        candidate: Tensor,
        reference: Tensor,
        face_mask: Tensor,
        skin_mask: Tensor,
        background_mask: Tensor,
    ) -> dict[str, Tensor]:
        _check_image("candidate", candidate)
        _check_image("reference", reference)
        if candidate.shape != reference.shape:
            raise ValueError(
                f"candidate/reference shapes must match, got {tuple(candidate.shape)} and {tuple(reference.shape)}"
            )
        if candidate.device != reference.device:
            raise ValueError("candidate and reference must be on the same device")

        face_mask = _normalize_mask("face_mask", face_mask, candidate)
        skin_mask = _normalize_mask("skin_mask", skin_mask, candidate)
        background_mask = _normalize_mask("background_mask", background_mask, candidate)

        common = {
            "quantile_mode": self.quantile_mode,
            "soft_quantile_temperature": self.soft_quantile_temperature,
        }
        loss_exp = face_exposure_loss(
            candidate, reference, face_mask, eps=self.eps, **common
        )
        loss_color = skin_chromaticity_loss(candidate, reference, skin_mask, **common)
        loss_tone = face_tone_shape_loss(
            candidate, reference, face_mask, eps=self.eps, **common
        )
        loss_fb = face_background_relation_loss(
            candidate,
            reference,
            face_mask,
            background_mask,
            eps=self.eps,
            **common,
        )
        total = (
            self.w_exp * loss_exp
            + self.w_color * loss_color
            + self.w_tone * loss_tone
            + self.w_fb * loss_fb
        )
        return {
            "loss": total,
            "loss_exp": loss_exp,
            "loss_color": loss_color,
            "loss_tone": loss_tone,
            "loss_fb": loss_fb,
        }


__all__ = [
    "PortraitRenderingLoss",
    "face_exposure_loss",
    "skin_chromaticity_loss",
    "face_tone_shape_loss",
    "face_background_relation_loss",
    "masked_quantile",
    "srgb_to_linear",
    "linear_to_srgb",
    "rgb_to_luminance",
    "rgb_to_lab",
]
