from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class ValidationStatus(str, Enum):
    VALID = "VALID"
    VALID_WITH_CONFOUNDERS = "VALID_WITH_CONFOUNDERS"
    INVALID = "INVALID"


@dataclass(slots=True)
class ImageFeatures:
    rgb: np.ndarray
    luminance: np.ndarray
    lab: np.ndarray


@dataclass(slots=True)
class ROISet:
    face: np.ndarray
    skin: np.ndarray
    face_ring: np.ndarray
    background: np.ndarray
    skin_source: str = "segmented"
    face_bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class ObjectiveResult:
    score: float
    evidence: dict[str, Any]
    components: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True)
class DimensionScore:
    score: float
    objective: float
    perceptual: float | None
    evidence: dict[str, Any]
    perceptual_gap: str | None = None
    confidence: float | None = None


@dataclass(slots=True)
class VariantScores:
    brightness: DimensionScore
    color: DimensionScore
    tone: DimensionScore
    overall: float

    def as_dict(self) -> dict[str, Any]:
        return {"brightness": self.brightness.score, "color": self.color.score, "tone": self.tone.score, "overall": self.overall}


@dataclass(slots=True)
class Failure:
    scene: str
    dimension: str
    type: str
    severity: float
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {"scene": self.scene, "dimension": self.dimension, "type": self.type, "severity": round(float(self.severity), 4), "description": self.description}


@dataclass(slots=True)
class SceneResult:
    scene_id: str
    tags: tuple[str, ...]
    family: str
    split: str
    validation: ValidationStatus
    validation_reasons: list[str]
    candidate: VariantScores | None
    baseline: VariantScores | None
    delta: dict[str, float]
    pairwise: str | None
    pairwise_confidence: float | None
    failures: list[Failure]
    content_guard: dict[str, Any]
    fidelity_guard: dict[str, Any] | None
    vlm_valid: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AggregateVariant:
    overall: float
    brightness: float
    color: float
    tone: float
    p10: dict[str, float]
    worst: dict[str, float]
    failure_rate: float
    failure_rate_by_dimension: dict[str, float]


@dataclass(slots=True)
class AggregateResult:
    candidate: AggregateVariant
    baseline: AggregateVariant
    delta: dict[str, float]
    regression_rate: float
    new_severe_failures: int
    valid_scenes: int
    total_scenes: int
    vlm_valid_scenes: int
    content_guard_failures: int
    split: str | None = None


@dataclass(slots=True)
class DecisionResult:
    decision: str
    passed: bool
    reasons: list[str]


@dataclass(slots=True)
class SceneSpec:
    id: str
    source: Path
    baseline: Path
    candidate: Path
    reference: Path
    tags: tuple[str, ...] = ()
    family: str = "all"
    split: str = "optimization"
    face_bbox: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    aligned_reference: bool = False
