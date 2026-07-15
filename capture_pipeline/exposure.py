from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from .types import AEOutput, RawFrame


def _coerce_ev(ev: torch.Tensor | float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = ev if torch.is_tensor(ev) else torch.as_tensor(ev, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.expand(batch)
    elif value.ndim == 1 and value.shape[0] == batch:
        pass
    elif value.numel() == batch:
        value = value.reshape(batch)
    else:
        raise ValueError(f"EV must be scalar or have shape [B]; got {tuple(value.shape)}.")
    if not torch.isfinite(value).all():
        raise ValueError("EV must contain finite values.")
    return value


def _cfa_labels(pattern: str) -> list[tuple[str, tuple[slice, slice]]]:
    names: list[str] = []
    green_index = 0
    for ch in pattern:
        if ch == "G":
            green_index += 1
            names.append(f"G{green_index}")
        else:
            names.append(ch)
    positions = [
        (slice(0, None, 2), slice(0, None, 2)),
        (slice(0, None, 2), slice(1, None, 2)),
        (slice(1, None, 2), slice(0, None, 2)),
        (slice(1, None, 2), slice(1, None, 2)),
    ]
    return list(zip(names, positions))


class HistogramRawAE(nn.Module):
    """Deterministic RAW-domain AE using the median of green Bayer samples."""

    def __init__(self, target_gray: float = 0.18, ev_min: float = -4.0, ev_max: float = 4.0, eps: float = 1e-6):
        super().__init__()
        if not 0.0 < target_gray <= 1.0:
            raise ValueError("target_gray must be in (0, 1].")
        if ev_min >= ev_max:
            raise ValueError("ev_min must be smaller than ev_max.")
        self.target_gray = float(target_gray)
        self.ev_min = float(ev_min)
        self.ev_max = float(ev_max)
        self.eps = float(eps)

    def forward(self, raw: RawFrame) -> AEOutput:
        frame = raw.normalized()
        green_planes = []
        for label, (rows, cols) in _cfa_labels(frame.cfa_pattern):
            if label.startswith("G"):
                green_planes.append(frame.mosaic[:, :, rows, cols].flatten(1))
        green = torch.cat(green_planes, dim=1)
        median = green.median(dim=1).values.clamp_min(self.eps)
        ev_unclamped = torch.log2(torch.as_tensor(self.target_gray, device=green.device, dtype=green.dtype) / median)
        ev = ev_unclamped.clamp(self.ev_min, self.ev_max)
        q10 = torch.quantile(green, 0.10, dim=1)
        q90 = torch.quantile(green, 0.90, dim=1)
        spread = (q90 - q10).clamp_min(0.0)
        confidence = (spread / (spread + 0.05)).clamp(0.0, 1.0)
        diagnostics: Dict[str, Any] = {
            "target_gray": self.target_gray,
            "green_median": median.detach().cpu().tolist(),
            "ev_unclamped": ev_unclamped.detach().cpu().tolist(),
        }
        return AEOutput(ev=ev, confidence=confidence, diagnostics=diagnostics)


class LearnedAEAdapter(nn.Module):
    """Wrap a separately trained AE model while preserving its state-dict keys."""

    def __init__(self, model: nn.Module, ev_min: float = -4.0, ev_max: float = 4.0):
        super().__init__()
        if ev_min >= ev_max:
            raise ValueError("ev_min must be smaller than ev_max.")
        self.model = model
        self.ev_min = float(ev_min)
        self.ev_max = float(ev_max)

    def forward(self, raw: RawFrame) -> AEOutput:
        frame = raw.normalized()
        result = self.model(frame.mosaic)
        if isinstance(result, dict):
            if "ev" not in result:
                raise KeyError("Learned AE dictionary output must contain 'ev'.")
            ev_raw = result["ev"]
            confidence_raw = result.get("confidence")
            diagnostics = {k: v for k, v in result.items() if k not in {"ev", "confidence"}}
        else:
            ev_raw = result
            confidence_raw = None
            diagnostics = {}
        if not torch.is_tensor(ev_raw):
            raise TypeError("Learned AE output must be a tensor or a dictionary containing a tensor 'ev'.")
        ev = _coerce_ev(ev_raw, frame.batch_size, frame.mosaic.device, frame.mosaic.dtype).clamp(self.ev_min, self.ev_max)
        if confidence_raw is None:
            confidence = torch.ones_like(ev)
        else:
            confidence = _coerce_ev(confidence_raw, frame.batch_size, frame.mosaic.device, frame.mosaic.dtype).clamp(0.0, 1.0)
        return AEOutput(ev=ev, confidence=confidence, diagnostics=diagnostics)


class RawExposureSynthesizer(nn.Module):
    """Apply deterministic EV scaling to normalized Bayer RAW."""

    def __init__(
        self,
        ev_min: float = -4.0,
        ev_max: float = 4.0,
        clipping_mode: str = "hard",
        soft_beta: float = 12.0,
        deep_shadow_threshold: float = 1e-3,
    ):
        super().__init__()
        if ev_min >= ev_max:
            raise ValueError("ev_min must be smaller than ev_max.")
        if clipping_mode not in {"hard", "soft"}:
            raise ValueError("clipping_mode must be 'hard' or 'soft'.")
        if soft_beta <= 0:
            raise ValueError("soft_beta must be positive.")
        self.ev_min = float(ev_min)
        self.ev_max = float(ev_max)
        self.clipping_mode = clipping_mode
        self.soft_beta = float(soft_beta)
        self.deep_shadow_threshold = float(deep_shadow_threshold)

    def _soft_clip(self, value: torch.Tensor) -> torch.Tensor:
        beta = self.soft_beta
        upper = 1.0 - F.softplus(beta * (1.0 - value)) / beta
        return upper.clamp_min(0.0)

    def forward(self, raw: RawFrame, ev: torch.Tensor | float) -> tuple[RawFrame, Dict[str, Any]]:
        frame = raw.normalized()
        ev_tensor = _coerce_ev(ev, frame.batch_size, frame.mosaic.device, frame.mosaic.dtype)
        if ((ev_tensor < self.ev_min) | (ev_tensor > self.ev_max)).any():
            raise ValueError(f"EV is outside supported range [{self.ev_min}, {self.ev_max}].")
        scale = torch.pow(torch.as_tensor(2.0, device=ev_tensor.device, dtype=ev_tensor.dtype), ev_tensor)
        scaled = frame.mosaic * scale.view(-1, 1, 1, 1)
        if self.clipping_mode == "hard":
            exposed_mosaic = scaled.clamp(0.0, 1.0)
        else:
            exposed_mosaic = self._soft_clip(scaled)
        exposed = frame.with_mosaic(exposed_mosaic, is_normalized=True)

        saturated = scaled >= 1.0
        deep_shadow = exposed_mosaic <= self.deep_shadow_threshold
        per_cfa = []
        for batch_idx in range(frame.batch_size):
            item: Dict[str, float] = {}
            for label, (rows, cols) in _cfa_labels(frame.cfa_pattern):
                item[label] = float(saturated[batch_idx : batch_idx + 1, :, rows, cols].float().mean().detach().cpu())
            per_cfa.append(item)
        diagnostics: Dict[str, Any] = {
            "predicted_ev": ev_tensor.detach().cpu().tolist(),
            "applied_ev": ev_tensor.detach().cpu().tolist(),
            "exposure_scale": scale.detach().cpu().tolist(),
            "saturation_ratio": saturated.flatten(1).float().mean(dim=1).detach().cpu().tolist(),
            "deep_shadow_ratio": deep_shadow.flatten(1).float().mean(dim=1).detach().cpu().tolist(),
            "per_cfa_saturation_ratio": per_cfa,
            "clipping_mode": self.clipping_mode,
        }
        return exposed, diagnostics
