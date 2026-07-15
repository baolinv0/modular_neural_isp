from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class DistributionStatus(str, Enum):
    IN_DOMAIN = "in_domain"
    BOUNDARY = "boundary"
    OOD = "ood"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: object) -> "DistributionStatus":
        normalized = str(value or "unknown").strip().lower().replace("-", "_")
        aliases = {
            "id": cls.IN_DOMAIN,
            "in_distribution": cls.IN_DOMAIN,
            "indomain": cls.IN_DOMAIN,
            "in_domain": cls.IN_DOMAIN,
            "boundary": cls.BOUNDARY,
            "borderline": cls.BOUNDARY,
            "ood": cls.OOD,
            "out_of_distribution": cls.OOD,
            "out_distribution": cls.OOD,
            "unknown": cls.UNKNOWN,
            "none": cls.UNKNOWN,
        }
        return aliases.get(normalized, cls.UNKNOWN)


@dataclass(frozen=True)
class CandidateScore:
    scene_id: str
    level: str
    alpha: float
    output_path: str
    overall_score: float
    confidence: float = 1.0
    action: str = "KEEP"
    hard_failures: tuple[str, ...] = ()
    distribution_status: DistributionStatus = DistributionStatus.UNKNOWN
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return abs(float(self.alpha)) <= 1e-8 or self.level == "a_000" or bool(self.metadata.get("is_baseline", False))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["distribution_status"] = self.distribution_status.value
        payload["hard_failures"] = list(self.hard_failures)
        payload["metrics"] = dict(self.metrics)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class TeacherSelection:
    scene_id: str
    baseline_path: str
    selected_path: str
    baseline_score: float
    selected_score: float
    selected_alpha: float
    selected_level: str
    score_delta: float
    confidence: float
    sample_weight: float
    distribution_status: DistributionStatus
    status: str
    reason: str
    rejected_candidates: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["distribution_status"] = self.distribution_status.value
        payload["rejected_candidates"] = [dict(item) for item in self.rejected_candidates]
        payload["metadata"] = dict(self.metadata)
        return payload

@dataclass(frozen=True)
class TeacherRecord:
    scene_id: str
    input_path: str
    baseline_path: str
    target_path: str
    selected_alpha: float
    selected_level: str
    baseline_score: float
    selected_score: float
    score_delta: float
    confidence: float
    sample_weight: float
    status: str
    split: str
    distribution_status: DistributionStatus
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["distribution_status"] = self.distribution_status.value
        payload["metadata"] = dict(self.metadata)
        return payload
