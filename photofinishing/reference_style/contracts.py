"""Data contracts for non-pixel-aligned reference-style training."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch


class TrainingStage(str, Enum):
    LUMA = "luma"
    CHROMA = "chroma"


@dataclass(frozen=True)
class ReferenceStyleLossWeights:
    histogram: float = 1.0
    moments: float = 0.25
    occupancy: float = 0.25
    contrast: float = 0.2
    content_anchor: float = 0.15
    luma_preservation: float = 1.0
    neutral_preservation: float = 0.2
    parameter_drift: float = 1e-4

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0 or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class ReferenceStyleBatch:
    """Same-scene but non-pixel-aligned source/reference batch."""

    source: torch.Tensor
    reference: torch.Tensor
    metadata: Mapping[str, object] | None = None

    def validate(self) -> None:
        for name, tensor in (("source", self.source), ("reference", self.reference)):
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError(f"{name} must have shape [B,3,H,W]")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf")
            if tensor.min() < 0 or tensor.max() > 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.source.shape[0] != self.reference.shape[0]:
            raise ValueError("source and reference batch sizes must match")
