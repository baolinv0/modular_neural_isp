"""Non-pixel-aligned losses for staged photofinishing adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


def _assert_image(image: torch.Tensor, name: str) -> None:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,H,W]")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite values")


def rgb_to_luma(image: torch.Tensor) -> torch.Tensor:
    _assert_image(image, "image")
    weights = image.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    _assert_image(image, "image")
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


def _flatten_per_image(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


def log_exposure_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    pred_y = _flatten_per_image(rgb_to_luma(prediction)).clamp_min(_EPS)
    ref_y = _flatten_per_image(rgb_to_luma(reference)).clamp_min(_EPS)
    pred_key = torch.log(pred_y).mean(dim=1)
    ref_key = torch.log(ref_y).mean(dim=1)
    return F.l1_loss(pred_key, ref_key)


def quantile_curve(value: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    flat = _flatten_per_image(value)
    return torch.quantile(flat, quantiles.to(device=value.device, dtype=value.dtype), dim=1).transpose(0, 1)


def luma_percentile_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> torch.Tensor:
    q = prediction.new_tensor(tuple(quantiles))
    return F.l1_loss(quantile_curve(rgb_to_luma(prediction), q), quantile_curve(rgb_to_luma(reference), q))


def luma_distribution_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    num_quantiles: int = 33,
) -> torch.Tensor:
    """Approximates 1-D Wasserstein distance with luminance quantile curves."""
    if num_quantiles < 3:
        raise ValueError("num_quantiles must be >= 3")
    q = torch.linspace(0.01, 0.99, num_quantiles, device=prediction.device, dtype=prediction.dtype)
    return F.l1_loss(quantile_curve(rgb_to_luma(prediction), q), quantile_curve(rgb_to_luma(reference), q))


def _pooled_chroma(image: torch.Tensor, pool_size: int) -> torch.Tensor:
    chroma = rgb_to_ycbcr(image)[:, 1:]
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    chroma = F.adaptive_avg_pool2d(chroma, output_size=(pool_size, pool_size))
    return chroma.flatten(2).transpose(1, 2)


def soft_chroma_histogram(
    image: torch.Tensor,
    bins: int = 16,
    sigma: float = 0.05,
    pool_size: int = 32,
) -> torch.Tensor:
    """Differentiable normalized two-dimensional CbCr histogram."""
    if bins < 4 or sigma <= 0:
        raise ValueError("bins must be >= 4 and sigma must be positive")
    chroma = _pooled_chroma(image, pool_size=pool_size)
    centers = torch.linspace(-0.5, 0.5, bins, device=image.device, dtype=image.dtype)
    cb = chroma[..., 0:1]
    cr = chroma[..., 1:2]
    cb_weights = torch.exp(-0.5 * ((cb - centers) / sigma).square())
    cr_weights = torch.exp(-0.5 * ((cr - centers) / sigma).square())
    histogram = torch.einsum("bni,bnj->bij", cb_weights, cr_weights)
    return histogram / histogram.sum(dim=(1, 2), keepdim=True).clamp_min(_EPS)


def chroma_histogram_loss(prediction: torch.Tensor, reference: torch.Tensor, bins: int = 16) -> torch.Tensor:
    return F.l1_loss(soft_chroma_histogram(prediction, bins=bins), soft_chroma_histogram(reference, bins=bins))


def chroma_moment_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    prediction_chroma = _pooled_chroma(prediction, pool_size=32)
    reference_chroma = _pooled_chroma(reference, pool_size=32)

    def moments(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = value.mean(dim=1)
        centered = value - mean[:, None, :]
        covariance = torch.einsum("bni,bnj->bij", centered, centered) / max(value.shape[1] - 1, 1)
        return mean, covariance

    prediction_mean, prediction_covariance = moments(prediction_chroma)
    reference_mean, reference_covariance = moments(reference_chroma)
    return F.l1_loss(prediction_mean, reference_mean) + F.l1_loss(prediction_covariance, reference_covariance)


def saturation(image: torch.Tensor) -> torch.Tensor:
    _assert_image(image, "image")
    maximum = image.max(dim=1, keepdim=True).values
    minimum = image.min(dim=1, keepdim=True).values
    return (maximum - minimum) / maximum.clamp_min(_EPS)


def saturation_distribution_loss(prediction: torch.Tensor, reference: torch.Tensor, num_quantiles: int = 17) -> torch.Tensor:
    q = torch.linspace(0.02, 0.98, num_quantiles, device=prediction.device, dtype=prediction.dtype)
    return F.l1_loss(quantile_curve(saturation(prediction), q), quantile_curve(saturation(reference), q))


def luminance_preservation_loss(prediction: torch.Tensor, frozen_stage1_output: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(rgb_to_luma(prediction), rgb_to_luma(frozen_stage1_output.detach()))


def total_variation(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
        + (value[..., :, 1:] - value[..., :, :-1]).abs().mean()
    )


def lut_delta_regularization(current_lut: torch.Tensor, frozen_stage1_lut: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if current_lut.shape != frozen_stage1_lut.shape:
        raise ValueError("Current and frozen LUT shapes differ")
    delta = current_lut - frozen_stage1_lut.detach()
    return delta.abs().mean(), total_variation(delta)


@dataclass(frozen=True)
class Stage1LossWeights:
    exposure: float = 1.0
    percentiles: float = 1.0
    distribution: float = 1.0
    parameter_anchor: float = 1e-4


@dataclass(frozen=True)
class Stage2LossWeights:
    chroma_histogram: float = 1.0
    chroma_moments: float = 0.5
    saturation: float = 0.25
    y_preserve: float = 1.0
    lut_anchor: float = 0.1
    lut_smoothness: float = 0.05


class Stage1UnpairedLoss(nn.Module):
    def __init__(self, weights: Stage1LossWeights = Stage1LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        parameter_anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        terms = {
            "exposure": log_exposure_loss(prediction, reference),
            "percentiles": luma_percentile_loss(prediction, reference),
            "distribution": luma_distribution_loss(prediction, reference),
            "parameter_anchor": parameter_anchor,
        }
        total = sum(getattr(self.weights, name) * value for name, value in terms.items())
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite Stage-1 loss")
        return total, terms


class Stage2UnpairedLoss(nn.Module):
    def __init__(self, weights: Stage2LossWeights = Stage2LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        frozen_stage1_output: torch.Tensor,
        current_lut: torch.Tensor,
        frozen_stage1_lut: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        lut_anchor, lut_smoothness = lut_delta_regularization(current_lut, frozen_stage1_lut)
        terms = {
            "chroma_histogram": chroma_histogram_loss(prediction, reference),
            "chroma_moments": chroma_moment_loss(prediction, reference),
            "saturation": saturation_distribution_loss(prediction, reference),
            "y_preserve": luminance_preservation_loss(prediction, frozen_stage1_output),
            "lut_anchor": lut_anchor,
            "lut_smoothness": lut_smoothness,
        }
        total = sum(getattr(self.weights, name) * value for name, value in terms.items())
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite Stage-2 loss")
        return total, terms
