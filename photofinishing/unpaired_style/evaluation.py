"""Independent non-pixel holdout evaluation for unpaired style adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import (
    cdf_distance,
    chroma_histogram_loss,
    chroma_moment_loss,
    edge_anchor_loss,
    high_frequency_anchor_loss,
    log_exposure_loss,
    percentile_loss,
    rgb_saturation,
    rgb_to_luminance,
    rgb_to_ycbcr,
    tone_region_loss,
)


def _forward_output(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
  payload = model(image, training_mode=True)
  if not isinstance(payload, Mapping) or "output" not in payload:
    raise RuntimeError("PhotofinishingModule evaluation requires training-mode output dictionary")
  output = payload["output"]
  if not torch.isfinite(output).all():
    raise RuntimeError("NON_FINITE_EVALUATION_OUTPUT")
  return output


def _clipping_metrics(image: torch.Tensor, epsilon: float = 1.0 / 255.0) -> Dict[str, torch.Tensor]:
  return {
    "shadow_clip": (image <= epsilon).float().mean(dim=(1, 2, 3)),
    "highlight_clip": (image >= 1.0 - epsilon).float().mean(dim=(1, 2, 3)),
  }


def evaluate_batch(
    baseline: torch.Tensor,
    adapted: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, torch.Tensor]:
  """Compute distribution style metrics and aligned content regressions."""
  baseline_y = rgb_to_luminance(baseline)
  adapted_y = rgb_to_luminance(adapted)
  reference_y = rgb_to_luminance(reference)
  baseline_cbcr = rgb_to_ycbcr(baseline)[:, 1:]
  adapted_cbcr = rgb_to_ycbcr(adapted)[:, 1:]
  reference_cbcr = rgb_to_ycbcr(reference)[:, 1:]
  baseline_sat = rgb_saturation(baseline)
  adapted_sat = rgb_saturation(adapted)
  reference_sat = rgb_saturation(reference)

  metrics = {
    "baseline_exposure_error": log_exposure_loss(baseline_y, reference_y),
    "adapted_exposure_error": log_exposure_loss(adapted_y, reference_y),
    "baseline_luminance_cdf": cdf_distance(baseline_y, reference_y, bins=32, sigma=0.03),
    "adapted_luminance_cdf": cdf_distance(adapted_y, reference_y, bins=32, sigma=0.03),
    "baseline_percentile_error": percentile_loss(baseline_y, reference_y),
    "adapted_percentile_error": percentile_loss(adapted_y, reference_y),
    "baseline_tone_region_error": tone_region_loss(baseline_y, reference_y),
    "adapted_tone_region_error": tone_region_loss(adapted_y, reference_y),
    "baseline_chroma_histogram": chroma_histogram_loss(
      baseline_cbcr, reference_cbcr, bins=16, sigma=0.04),
    "adapted_chroma_histogram": chroma_histogram_loss(
      adapted_cbcr, reference_cbcr, bins=16, sigma=0.04),
    "baseline_chroma_moments": chroma_moment_loss(baseline_cbcr, reference_cbcr),
    "adapted_chroma_moments": chroma_moment_loss(adapted_cbcr, reference_cbcr),
    "baseline_saturation_cdf": cdf_distance(baseline_sat, reference_sat, bins=32, sigma=0.03),
    "adapted_saturation_cdf": cdf_distance(adapted_sat, reference_sat, bins=32, sigma=0.03),
    "content_edge_drift": edge_anchor_loss(adapted, baseline),
    "content_high_frequency_drift": high_frequency_anchor_loss(adapted, baseline),
    "content_luminance_drift": (adapted_y - baseline_y).abs().mean(dim=(1, 2, 3)),
  }
  metrics.update({f"baseline_{name}": value for name, value in _clipping_metrics(baseline).items()})
  metrics.update({f"adapted_{name}": value for name, value in _clipping_metrics(adapted).items()})
  metrics["luminance_style_improvement"] = (
    metrics["baseline_exposure_error"] + metrics["baseline_luminance_cdf"]
    + metrics["baseline_percentile_error"] + metrics["baseline_tone_region_error"]
    - metrics["adapted_exposure_error"] - metrics["adapted_luminance_cdf"]
    - metrics["adapted_percentile_error"] - metrics["adapted_tone_region_error"]
  )
  metrics["chroma_style_improvement"] = (
    metrics["baseline_chroma_histogram"] + metrics["baseline_chroma_moments"]
    + metrics["baseline_saturation_cdf"] - metrics["adapted_chroma_histogram"]
    - metrics["adapted_chroma_moments"] - metrics["adapted_saturation_cdf"]
  )
  for name, value in metrics.items():
    if not torch.isfinite(value).all():
      raise RuntimeError(f"NON_FINITE_EVALUATION_METRIC:{name}")
  return metrics


def _aggregate(per_sample: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
  if not per_sample:
    raise RuntimeError("evaluation produced no samples")
  names = sorted(name for name in per_sample[0] if name != "sample_id")
  return {
    name: {
      "mean": mean(float(item[name]) for item in per_sample),
      "median": median(float(item[name]) for item in per_sample),
    }
    for name in names
  }


def evaluate_models(
    *,
    baseline_model: nn.Module,
    adapted_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: str,
) -> Dict[str, object]:
  baseline_model.to(device).eval()
  adapted_model.to(device).eval()
  per_sample: List[Dict[str, float]] = []
  with torch.no_grad():
    for batch in loader:
      images = batch["input"].to(device)
      references = batch["reference"].to(device)
      baseline = _forward_output(baseline_model, images)
      adapted = _forward_output(adapted_model, images)
      metrics = evaluate_batch(baseline, adapted, references)
      sample_ids = list(batch["sample_id"])
      for batch_index, sample_id in enumerate(sample_ids):
        record: Dict[str, float] = {"sample_id": str(sample_id)}
        for name, values in metrics.items():
          record[name] = float(values[batch_index].detach().cpu())
        per_sample.append(record)
  report = {
    "num_samples": len(per_sample),
    "aggregate": _aggregate(per_sample),
    "per_sample": per_sample,
  }
  destination = Path(output_path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
  return report
