"""Non-pixel-aligned luminance and chroma distribution losses."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Stage1LossWeights, Stage2LossWeights

EPS = 1e-6


def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
  weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
  return (rgb * weights).sum(dim=1, keepdim=True)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
  y = rgb_to_luminance(rgb)
  cb = (rgb[:, 2:3] - y) * 0.5389
  cr = (rgb[:, 0:1] - y) * 0.6350
  return torch.cat([y, cb, cr], dim=1)


def rgb_saturation(rgb: torch.Tensor) -> torch.Tensor:
  maximum = rgb.max(dim=1, keepdim=True).values
  minimum = rgb.min(dim=1, keepdim=True).values
  return (maximum - minimum) / maximum.clamp_min(EPS)


def _mask_or_ones(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
  if mask is None:
    return torch.ones_like(value[:, :1])
  if mask.ndim != 4 or mask.shape[0] != value.shape[0]:
    raise ValueError("mask must have shape Bx1xHxW")
  if mask.shape[-2:] != value.shape[-2:]:
    mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
  return mask.float().clamp(0, 1)


def masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
  weights = _mask_or_ones(value, mask)
  if value.shape[1] != weights.shape[1]:
    weights = weights.expand(-1, value.shape[1], -1, -1)
  numerator = (value * weights).sum(dim=(1, 2, 3))
  denominator = weights.sum(dim=(1, 2, 3)).clamp_min(EPS)
  return numerator / denominator


def soft_histogram_1d(
    value: torch.Tensor,
    *,
    bins: int,
    sigma: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  if bins <= 1 or sigma <= 0:
    raise ValueError("bins must be > 1 and sigma must be positive")
  if value.shape[1] != 1:
    raise ValueError("soft_histogram_1d expects one channel")
  centers = torch.linspace(min_value, max_value, bins, device=value.device, dtype=value.dtype)
  flat = value.flatten(2).transpose(1, 2)
  weights = _mask_or_ones(value, mask).flatten(2).transpose(1, 2)
  kernel = torch.exp(-0.5 * ((flat - centers.view(1, 1, -1)) / sigma) ** 2)
  histogram = (kernel * weights).sum(dim=1)
  return histogram / histogram.sum(dim=1, keepdim=True).clamp_min(EPS)


def cdf_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    bins: int,
    sigma: float,
    first_mask: Optional[torch.Tensor] = None,
    second_mask: Optional[torch.Tensor] = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
  first_hist = soft_histogram_1d(
    first, bins=bins, sigma=sigma, mask=first_mask, min_value=min_value, max_value=max_value)
  second_hist = soft_histogram_1d(
    second, bins=bins, sigma=sigma, mask=second_mask, min_value=min_value, max_value=max_value)
  return (first_hist.cumsum(dim=1) - second_hist.cumsum(dim=1)).abs().mean(dim=1)


def percentile_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_mask: Optional[torch.Tensor] = None,
    second_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  quantiles = first.new_tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
  losses = []
  first_weights = _mask_or_ones(first, first_mask)
  second_weights = _mask_or_ones(second, second_mask)
  for batch_index in range(first.shape[0]):
    first_values = first[batch_index].flatten()[first_weights[batch_index].flatten() > 0.5]
    second_values = second[batch_index].flatten()[second_weights[batch_index].flatten() > 0.5]
    if first_values.numel() == 0 or second_values.numel() == 0:
      losses.append(first.new_tensor(0.0))
      continue
    losses.append((torch.quantile(first_values, quantiles) - torch.quantile(second_values, quantiles)).abs().mean())
  return torch.stack(losses)


def log_exposure_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_mask: Optional[torch.Tensor] = None,
    second_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  first_log = masked_mean(torch.log(first.clamp_min(EPS)), first_mask)
  second_log = masked_mean(torch.log(second.clamp_min(EPS)), second_mask)
  return (first_log - second_log).abs()


def tone_region_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_mask: Optional[torch.Tensor] = None,
    second_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  first_support = _mask_or_ones(first, first_mask)
  second_support = _mask_or_ones(second, second_mask)
  thresholds = ((-1.0, 0.20), (0.20, 0.80), (0.80, 2.0))
  losses = []
  sharpness = 30.0
  for low, high in thresholds:
    if low < 0:
      first_region = torch.sigmoid((high - first) * sharpness)
      second_region = torch.sigmoid((high - second) * sharpness)
    elif high > 1:
      first_region = torch.sigmoid((first - low) * sharpness)
      second_region = torch.sigmoid((second - low) * sharpness)
    else:
      first_region = torch.sigmoid((first - low) * sharpness) * torch.sigmoid((high - first) * sharpness)
      second_region = torch.sigmoid((second - low) * sharpness) * torch.sigmoid((high - second) * sharpness)
    first_region = first_region * first_support
    second_region = second_region * second_support
    first_occupancy = first_region.sum(dim=(1, 2, 3)) / first_support.sum(dim=(1, 2, 3)).clamp_min(EPS)
    second_occupancy = second_region.sum(dim=(1, 2, 3)) / second_support.sum(dim=(1, 2, 3)).clamp_min(EPS)
    first_mean = (first * first_region).sum(dim=(1, 2, 3)) / first_region.sum(dim=(1, 2, 3)).clamp_min(EPS)
    second_mean = (second * second_region).sum(dim=(1, 2, 3)) / second_region.sum(dim=(1, 2, 3)).clamp_min(EPS)
    losses.append((first_occupancy - second_occupancy).abs() + (first_mean - second_mean).abs())
  return torch.stack(losses, dim=0).mean(dim=0)


def _sobel(value: torch.Tensor) -> torch.Tensor:
  kernel_x = value.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
  kernel_y = kernel_x.transpose(2, 3)
  gx = F.conv2d(value, kernel_x, padding=1)
  gy = F.conv2d(value, kernel_y, padding=1)
  return torch.sqrt(gx.square() + gy.square() + EPS)


def edge_anchor_loss(output: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  return (_sobel(rgb_to_luminance(output)) - _sobel(rgb_to_luminance(anchor))).abs().mean(dim=(1, 2, 3))


def high_frequency_anchor_loss(output: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  output_y = rgb_to_luminance(output)
  anchor_y = rgb_to_luminance(anchor)
  output_hp = output_y - F.avg_pool2d(output_y, 5, stride=1, padding=2)
  anchor_hp = anchor_y - F.avg_pool2d(anchor_y, 5, stride=1, padding=2)
  output_energy = output_hp.square().mean(dim=(1, 2, 3)).clamp_min(EPS)
  anchor_energy = anchor_hp.square().mean(dim=(1, 2, 3)).clamp_min(EPS)
  return (torch.log(output_energy) - torch.log(anchor_energy)).abs()


def _weighted_region_loss(
    function: Callable[..., torch.Tensor],
    output_value: torch.Tensor,
    reference_value: torch.Tensor,
    output_masks: Optional[torch.Tensor],
    reference_masks: Optional[torch.Tensor],
    region_valid: Optional[torch.Tensor],
    region_weights: Optional[torch.Tensor],
    **kwargs,
) -> torch.Tensor:
  batch = output_value.shape[0]
  if output_masks is None or output_masks.shape[1] == 0:
    return output_value.new_zeros(batch)
  total = output_value.new_zeros(batch)
  denominator = output_value.new_zeros(batch)
  for region_index in range(output_masks.shape[1]):
    valid = region_valid[:, region_index] if region_valid is not None else output_value.new_ones(batch)
    weight = region_weights[:, region_index] if region_weights is not None else output_value.new_ones(batch)
    effective = valid * weight
    region_loss = function(
      output_value,
      reference_value,
      first_mask=output_masks[:, region_index],
      second_mask=reference_masks[:, region_index],
      **kwargs,
    )
    total = total + region_loss * effective
    denominator = denominator + effective
  return total / denominator.clamp_min(1.0)


def soft_chroma_histogram(
    cbcr: torch.Tensor,
    *,
    bins: int,
    sigma: float,
    mask: Optional[torch.Tensor] = None,
    sample_size: int = 64,
) -> torch.Tensor:
  if cbcr.shape[1] != 2:
    raise ValueError("cbcr must have two channels")
  cbcr = F.adaptive_avg_pool2d(cbcr, (sample_size, sample_size))
  mask = _mask_or_ones(cbcr[:, :1], mask)
  mask = F.adaptive_avg_pool2d(mask, (sample_size, sample_size)).flatten(2).squeeze(1)
  centers = torch.linspace(-0.5, 0.5, bins, device=cbcr.device, dtype=cbcr.dtype)
  cb = cbcr[:, 0].flatten(1)
  cr = cbcr[:, 1].flatten(1)
  cb_weight = torch.exp(-0.5 * ((cb.unsqueeze(-1) - centers) / sigma) ** 2)
  cr_weight = torch.exp(-0.5 * ((cr.unsqueeze(-1) - centers) / sigma) ** 2)
  histogram = torch.einsum("bni,bnj,bn->bij", cb_weight, cr_weight, mask)
  return histogram / histogram.sum(dim=(1, 2), keepdim=True).clamp_min(EPS)


def chroma_histogram_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    bins: int,
    sigma: float,
    first_mask: Optional[torch.Tensor] = None,
    second_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  first_hist = soft_chroma_histogram(first, bins=bins, sigma=sigma, mask=first_mask)
  second_hist = soft_chroma_histogram(second, bins=bins, sigma=sigma, mask=second_mask)
  return (first_hist - second_hist).abs().mean(dim=(1, 2))


def chroma_moment_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_mask: Optional[torch.Tensor] = None,
    second_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  def moments(value: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = _mask_or_ones(value[:, :1], mask).flatten(2)
    flat = value.flatten(2)
    normalizer = weights.sum(dim=2, keepdim=True).clamp_min(EPS)
    mean = (flat * weights).sum(dim=2, keepdim=True) / normalizer
    centered = flat - mean
    covariance = torch.einsum("bcn,bdn,bkn->bcd", centered, centered, weights) / normalizer
    return mean.squeeze(2), covariance
  first_mean, first_cov = moments(first, first_mask)
  second_mean, second_cov = moments(second, second_mask)
  return (first_mean - second_mean).abs().mean(dim=1) + (first_cov - second_cov).abs().mean(dim=(1, 2))


def lut_regularization(lut: torch.Tensor, identity: torch.Tensor, max_displacement: float = 0.20) -> Dict[str, torch.Tensor]:
  if identity.shape[0] == 1 and lut.shape[0] > 1:
    identity = identity.expand(lut.shape[0], -1, -1, -1)
  displacement = lut - identity
  identity_loss = displacement.abs().mean(dim=(1, 2, 3))
  tv_h = (lut[:, :, 1:, :] - lut[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
  tv_w = (lut[:, :, :, 1:] - lut[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
  bound = F.relu(displacement.abs() - max_displacement).mean(dim=(1, 2, 3))
  return {"identity": identity_loss, "total_variation": tv_h + tv_w, "bound": bound}


class Stage1UnpairedLoss(nn.Module):
  """Train Gain/GTM with global and semantic luminance distribution supervision."""

  def __init__(self, weights: Stage1LossWeights, bins: int = 32, sigma: float = 0.03):
    super().__init__()
    self.weights = weights
    self.bins = bins
    self.sigma = sigma

  def forward(
      self,
      output: torch.Tensor,
      reference: torch.Tensor,
      baseline: torch.Tensor,
      confidence: torch.Tensor,
      output_masks: Optional[torch.Tensor] = None,
      reference_masks: Optional[torch.Tensor] = None,
      region_valid: Optional[torch.Tensor] = None,
      region_weights: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    output_y = rgb_to_luminance(output)
    reference_y = rgb_to_luminance(reference)
    components = {
      "exposure": log_exposure_loss(output_y, reference_y),
      "luminance_cdf": cdf_distance(output_y, reference_y, bins=self.bins, sigma=self.sigma),
      "percentiles": percentile_loss(output_y, reference_y),
      "tone_regions": tone_region_loss(output_y, reference_y),
      "semantic_regions": _weighted_region_loss(
        cdf_distance, output_y, reference_y, output_masks, reference_masks,
        region_valid, region_weights, bins=self.bins, sigma=self.sigma),
      "edge_anchor": edge_anchor_loss(output, baseline),
      "high_frequency_anchor": high_frequency_anchor_loss(output, baseline),
      "residual_anchor": (output - baseline).abs().mean(dim=(1, 2, 3)),
    }
    weighted = output.new_zeros(output.shape[0])
    for name, weight in asdict(self.weights).items():
      weighted = weighted + components[name] * weight
    confidence = confidence.reshape(-1).to(output)
    total = (weighted * confidence).sum() / confidence.sum().clamp_min(EPS)
    return total, {name: value.mean().detach() for name, value in components.items()}


class Stage2UnpairedLoss(nn.Module):
  """Train chroma LUT while preserving the frozen Stage-1 luminance result."""

  def __init__(self, weights: Stage2LossWeights, bins: int = 16, sigma: float = 0.04):
    super().__init__()
    self.weights = weights
    self.bins = bins
    self.sigma = sigma

  def forward(
      self,
      output: torch.Tensor,
      reference: torch.Tensor,
      stage1_anchor: torch.Tensor,
      cbcr_lut: torch.Tensor,
      identity_lut: torch.Tensor,
      confidence: torch.Tensor,
      output_masks: Optional[torch.Tensor] = None,
      reference_masks: Optional[torch.Tensor] = None,
      region_valid: Optional[torch.Tensor] = None,
      region_weights: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    output_ycbcr = rgb_to_ycbcr(output)
    reference_ycbcr = rgb_to_ycbcr(reference)
    output_cbcr = output_ycbcr[:, 1:]
    reference_cbcr = reference_ycbcr[:, 1:]
    output_sat = rgb_saturation(output)
    reference_sat = rgb_saturation(reference)
    regularizers = lut_regularization(cbcr_lut, identity_lut)
    components = {
      "chroma_histogram": chroma_histogram_loss(
        output_cbcr, reference_cbcr, bins=self.bins, sigma=self.sigma),
      "chroma_moments": chroma_moment_loss(output_cbcr, reference_cbcr),
      "saturation_cdf": cdf_distance(output_sat, reference_sat, bins=32, sigma=0.03),
      "semantic_regions": _weighted_region_loss(
        chroma_histogram_loss, output_cbcr, reference_cbcr, output_masks,
        reference_masks, region_valid, region_weights, bins=self.bins, sigma=self.sigma),
      "luminance_preserve": (
        rgb_to_luminance(output) - rgb_to_luminance(stage1_anchor.detach())
      ).abs().mean(dim=(1, 2, 3)),
      "edge_anchor": edge_anchor_loss(output, stage1_anchor.detach()),
      "lut_identity": regularizers["identity"],
      "lut_total_variation": regularizers["total_variation"],
      "lut_bound": regularizers["bound"],
    }
    weighted = output.new_zeros(output.shape[0])
    for name, weight in asdict(self.weights).items():
      weighted = weighted + components[name] * weight
    confidence = confidence.reshape(-1).to(output)
    total = (weighted * confidence).sum() / confidence.sum().clamp_min(EPS)
    return total, {name: value.mean().detach() for name, value in components.items()}
