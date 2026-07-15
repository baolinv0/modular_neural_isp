from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class DatasetClass(str, Enum):
    CLEAN = "clean"
    BOUNDARY = "boundary"
    INVALID = "invalid"


class FailureCode(str, Enum):
    BRIGHTNESS_UNDER = "F1_brightness_under"
    BRIGHTNESS_OVER = "F2_brightness_over"
    HIGHLIGHT_CLIPPING = "F3_highlight_clipping"
    SHADOW_CRUSH = "F4_shadow_crush"
    CHROMA_DRIFT = "F5_chroma_drift"
    CONTROL_CURVE = "F6_control_curve"
    REGIONAL_INCONSISTENCY = "F7_regional_inconsistency"
    STRUCTURAL_ARTIFACT = "F8_structural_artifact"


@dataclass(frozen=True)
class SceneGateResult:
    scene_name: str
    classification: DatasetClass
    reasons: tuple[str, ...]
    level_luminance: tuple[float, ...]
    monotonic_violation_rate: float
    endpoint_range_ev: float
    endpoint_clip_ratio: float
    endpoint_shadow_ratio: float
    max_chroma_drift: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classification"] = self.classification.value
        return payload


@dataclass(frozen=True)
class DatasetGateSummary:
    total_scenes: int
    counts: Mapping[DatasetClass, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_scenes": self.total_scenes,
            "counts": {key.value: int(value) for key, value in self.counts.items()},
        }


@dataclass(frozen=True)
class DatasetGateReport:
    summary: DatasetGateSummary
    scenes: tuple[SceneGateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


@dataclass(frozen=True)
class QwenSceneResult:
    scene_id: str
    action: str
    failure_reasons: tuple[str, ...]
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "action": self.action,
            "failure_reasons": list(self.failure_reasons),
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class QwenTMQAResults:
    summary: Mapping[str, Any]
    scenes: Mapping[str, QwenSceneResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "scenes": {name: result.to_dict() for name, result in self.scenes.items()},
        }


@dataclass(frozen=True)
class DataTask:
    failure_code: FailureCode
    target_module: str
    capability_gap: str
    positive_scenes: tuple[str, ...]
    boundary_scenes: tuple[str, ...]
    hard_negative_scenes: tuple[str, ...]
    regression_anchor_scenes: tuple[str, ...]
    required_supervision: tuple[str, ...]
    acceptance_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_code"] = self.failure_code.value
        return payload


@dataclass(frozen=True)
class DataPrescription:
    tasks: tuple[DataTask, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": [task.to_dict() for task in self.tasks]}
