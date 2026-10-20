"""Non-pixel-aligned style losses for staged photofinishing fine-tuning."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
  """BT.709 display-referred luminance proxy."""
  if rgb.ndim != 4 or rgb.shape[1] != 3:
    raise ValueError(f"Expected BCHW RGB, got {tuple(rgb.shape)}")
  weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
  return (rgb.clamp(0.0, 1.0) * weights).sum(dim=1, keepdim=True)


def rgb_to_ycbcr_style(rgb: torch.Tensor) -> torch.Tensor:
  """Differentiable YCbCr-like transform for display-referred style statistics."""
  y = rgb_to_luma(rgb)
  r, _, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
  cb = (b - y) / 1.8556
  cr = (r - y) / 1.5748
  return torch.cat([y, cb, cr], dim=1)


def _flatten_distribution(x: torch.Tensor) -> torch.Tensor:
  return x.flatten(start_dim=2)


def _resample_sorted(x: torch.Tensor, samples: int = 2048) -> torch.Tensor:
  if x.shape[-1] == samples:
    return x
  return F.interpolate(x, size=samples, mode="linear", align_corners=True)


def sorted_distribution_loss(a: torch.Tensor, b: torch.Tensor, log_domain: bool = False) -> torch.Tensor:
  """One-dimensional Wasserstein-like loss invariant to spatial permutation."""
  if a.shape[:2] != b.shape[:2]:
    raise ValueError("Batch and channel dimensions must match")
  a_flat = _flatten_distribution(a)
  b_flat = _flatten_distribution(b)
  if log_domain:
    a_flat = torch.log(a_flat.clamp_min(_EPS))
    b_flat = torch.log(b_flat.clamp_min(_EPS))
  a_sorted = _resample_sorted(torch.sort(a_flat, dim=-1).values)
  b_sorted = _resample_sorted(torch.sort(b_flat, dim=-1).values)
  return F.l1_loss(a_sorted, b_sorted)


def _moments(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  flat = _flatten_distribution(x)
  return flat.mean(dim=-1), flat.std(dim=-1, unbiased=False)


def moment_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  mean_a, std_a = _moments(a)
  mean_b, std_b = _moments(b)
  return F.l1_loss(mean_a, mean_b) + F.l1_loss(std_a, std_b)


def soft_occupancy(x: torch.Tensor, threshold: float, direction: str,
                   sharpness: float = 40.0) -> torch.Tensor:
  if direction == "below":
    probability = torch.sigmoid((threshold - x) * sharpness)
  elif direction == "above":
    probability = torch.sigmoid((x - threshold) * sharpness)
  else:
    raise ValueError("direction must be 'below' or 'above'")
  return probability.mean(dim=(1, 2, 3))


def luminance_style_loss(output: torch.Tensor, reference: torch.Tensor
                         ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
  """Matches brightness and tone distributions without pixel correspondence."""
  y_out = rgb_to_luma(output)
  y_ref = rgb_to_luma(reference)
  distribution = sorted_distribution_loss(y_out, y_ref, log_domain=True)
  moments = moment_loss(y_out, y_ref)
  shadow = F.l1_loss(
    soft_occupancy(y_out, 0.10, "below"),
    soft_occupancy(y_ref, 0.10, "below"),
  )
  highlight = F.l1_loss(
    soft_occupancy(y_out, 0.90, "above"),
    soft_occupancy(y_ref, 0.90, "above"),
  )
  loss = distribution + 0.5 * moments + 0.25 * (shadow + highlight)
  return loss, {
    "luma_distribution": distribution,
    "luma_moments": moments,
    "shadow_occupancy": shadow,
    "highlight_occupancy": highlight,
  }


def _channel_covariance(x: torch.Tensor) -> torch.Tensor:
  flat = _flatten_distribution(x)
  centered = flat - flat.mean(dim=-1, keepdim=True)
  return centered @ centered.transpose(1, 2) / max(flat.shape[-1] - 1, 1)


def chroma_style_loss(output: torch.Tensor, reference: torch.Tensor
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
  """Matches CbCr distribution and covariance without spatial alignment."""
  cbcr_out = rgb_to_ycbcr_style(output)[:, 1:]
  cbcr_ref = rgb_to_ycbcr_style(reference)[:, 1:]
  marginal = sorted_distribution_loss(cbcr_out, cbcr_ref)
  moments = moment_loss(cbcr_out, cbcr_ref)
  covariance = F.l1_loss(_channel_covariance(cbcr_out), _channel_covariance(cbcr_ref))
  magnitude_out = torch.linalg.vector_norm(cbcr_out, dim=1, keepdim=True)
  magnitude_ref = torch.linalg.vector_norm(cbcr_ref, dim=1, keepdim=True)
  magnitude = sorted_distribution_loss(magnitude_out, magnitude_ref)
  loss = marginal + 0.5 * moments + 0.5 * covariance + 0.5 * magnitude
  return loss, {
    "chroma_distribution": marginal,
    "chroma_moments": moments,
    "chroma_covariance": covariance,
    "chroma_magnitude": magnitude,
  }


class ToneReferenceLoss(nn.Module):
  """Stage-1 loss: AGT supplies only non-aligned luminance statistics."""

  def forward(self, output: torch.Tensor, reference: torch.Tensor,
              sample_weight: Optional[torch.Tensor] = None):
    loss, details = luminance_style_loss(output, reference)
    if sample_weight is not None:
      loss = loss * sample_weight.float().mean()
    return loss, details


class ChromaReferenceLoss(nn.Module):
  """Stage-2 loss: AGT chroma statistics plus same-input luminance anchoring."""

  def __init__(self, luma_anchor_weight: float = 2.0):
    super().__init__()
    self.luma_anchor_weight = float(luma_anchor_weight)

  def forward(self, output: torch.Tensor, reference: torch.Tensor,
              anchor_output: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
    chroma, details = chroma_style_loss(output, reference)
    luma_anchor = F.l1_loss(rgb_to_luma(output), rgb_to_luma(anchor_output.detach()))
    loss = chroma + self.luma_anchor_weight * luma_anchor
    if sample_weight is not None:
      loss = loss * sample_weight.float().mean()
    details["luma_anchor"] = luma_anchor
    return loss, details
