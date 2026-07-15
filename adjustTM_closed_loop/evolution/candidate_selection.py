from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import CandidateScore, DistributionStatus, TeacherSelection


@dataclass(frozen=True)
class SelectionConfig:
    min_improvement: float = 0.03
    min_confidence: float = 0.70
    allow_boundary: bool = True
    boundary_weight: float = 0.35
    score_tolerance: float = 1e-6
    improvement_scale: float = 0.15
    accepted_actions: tuple[str, ...] = ("KEEP",)

    def __post_init__(self) -> None:
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be non-negative")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must lie in [0,1]")
        if not 0 <= self.boundary_weight <= 1:
            raise ValueError("boundary_weight must lie in [0,1]")
        if self.improvement_scale <= 0:
            raise ValueError("improvement_scale must be positive")


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    return current


def _first(payload: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _hard_failures(payload: Mapping[str, Any]) -> tuple[str, ...]:
    direct = _first(payload, ("hard_failures", "fatal_defects", "hard_failure_reasons"))
    if isinstance(direct, (list, tuple)):
        return tuple(str(item) for item in direct)
    hard_gate = payload.get("hard_gate")
    if isinstance(hard_gate, Mapping):
        reasons = hard_gate.get("reasons")
        passed = hard_gate.get("passed")
        if isinstance(reasons, (list, tuple)) and (passed is False or reasons):
            return tuple(str(item) for item in reasons)
    return ()


def candidate_from_mapping(payload: Mapping[str, Any]) -> CandidateScore:
    scene_id = _first(payload, ("scene_id", "scene_name", "filename", "name"))
    if not scene_id:
        raise ValueError("Candidate score record is missing scene_id/scene_name")
    alpha = float(_first(payload, ("alpha", "control", "control_value"), 0.0))
    level = str(_first(payload, ("level", "level_name"), "a_000" if abs(alpha) <= 1e-8 else f"alpha_{alpha:+.4f}"))
    output_path = _first(payload, ("output_path", "image_path", "path"))
    if not output_path:
        raise ValueError(f"Candidate {scene_id}/{level} is missing output_path")
    score = _first(payload, ("overall_score", "score", "quality_score"))
    if score is None:
        score = _nested(payload, "decision", "score")
    if score is None:
        raise ValueError(f"Candidate {scene_id}/{level} is missing overall score")
    confidence = _first(payload, ("confidence", "score_confidence"))
    if confidence is None:
        confidence = _nested(payload, "decision", "confidence")
    if confidence is None:
        confidence = 1.0
    action = _first(payload, ("action", "final_action"))
    if action is None:
        action = _nested(payload, "decision", "action")
    action = str(action or "KEEP").upper()
    distribution = _first(payload, ("distribution_status", "domain_status", "distribution"), "unknown")
    metrics_raw = payload.get("metrics", {})
    metrics: dict[str, float] = {}
    if isinstance(metrics_raw, Mapping):
        for key, value in metrics_raw.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    consumed = {
        "scene_id", "scene_name", "filename", "name", "alpha", "control", "control_value",
        "level", "level_name", "output_path", "image_path", "path", "overall_score", "score",
        "quality_score", "confidence", "score_confidence", "action", "final_action", "decision",
        "hard_failures", "fatal_defects", "hard_failure_reasons", "hard_gate", "distribution_status",
        "domain_status", "distribution", "metrics",
    }
    metadata = {str(k): v for k, v in payload.items() if k not in consumed}
    candidate = CandidateScore(
        scene_id=str(scene_id),
        level=level,
        alpha=alpha,
        output_path=str(output_path),
        overall_score=float(score),
        confidence=float(confidence),
        action=action,
        hard_failures=_hard_failures(payload),
        distribution_status=DistributionStatus.coerce(distribution),
        metrics=metrics,
        metadata=metadata,
    )
    if not math.isfinite(candidate.overall_score) or not math.isfinite(candidate.confidence):
        raise ValueError(f"Candidate {scene_id}/{level} contains non-finite score/confidence")
    return candidate


def _flatten_json_payload(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        raise TypeError("Score file must contain a list or mapping")
    for key in ("candidates", "records", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    scenes = payload.get("scenes")
    if isinstance(scenes, Mapping):
        records: list[Mapping[str, Any]] = []
        for scene_id, scene_value in scenes.items():
            if isinstance(scene_value, list):
                for item in scene_value:
                    if isinstance(item, Mapping):
                        records.append({"scene_id": scene_id, **dict(item)})
            elif isinstance(scene_value, Mapping):
                candidates = scene_value.get("candidates")
                if isinstance(candidates, list):
                    for item in candidates:
                        if isinstance(item, Mapping):
                            records.append({"scene_id": scene_id, **dict(item)})
        return records
    return [payload]


def load_candidate_scores(path: str | Path) -> list[CandidateScore]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = _flatten_json_payload(json.loads(path.read_text(encoding="utf-8")))
    candidates = [candidate_from_mapping(record) for record in records]
    resolved = []
    for candidate in candidates:
        output_path = Path(candidate.output_path)
        if not output_path.is_absolute():
            output_path = (path.parent / output_path).resolve()
            candidate = replace(candidate, output_path=str(output_path))
        resolved.append(candidate)
    return resolved


class CandidateSelector:
    def __init__(self, config: SelectionConfig = SelectionConfig()) -> None:
        self.config = config

    def _rejection_reasons(self, candidate: CandidateScore) -> list[str]:
        reasons: list[str] = []
        if candidate.hard_failures:
            reasons.extend(candidate.hard_failures)
        if candidate.action not in self.config.accepted_actions:
            reasons.append(f"action:{candidate.action}")
        if candidate.confidence < self.config.min_confidence:
            reasons.append("low_confidence")
        if candidate.distribution_status is DistributionStatus.OOD:
            reasons.append("out_of_distribution")
        if candidate.distribution_status is DistributionStatus.UNKNOWN:
            reasons.append("unknown_distribution")
        if candidate.distribution_status is DistributionStatus.BOUNDARY and not self.config.allow_boundary:
            reasons.append("boundary_disallowed")
        return reasons

    def _weight(self, candidate: CandidateScore, score_delta: float) -> float:
        domain_weight = self.config.boundary_weight if candidate.distribution_status is DistributionStatus.BOUNDARY else 1.0
        margin_weight = min(1.0, max(0.05, score_delta / self.config.improvement_scale))
        return float(max(0.0, min(1.0, candidate.confidence * domain_weight * margin_weight)))

    def select_scene(self, candidates: Sequence[CandidateScore]) -> TeacherSelection:
        if not candidates:
            raise ValueError("A scene requires at least one candidate")
        scene_ids = {candidate.scene_id for candidate in candidates}
        if len(scene_ids) != 1:
            raise ValueError(f"select_scene received multiple scene IDs: {sorted(scene_ids)}")
        baselines = [candidate for candidate in candidates if candidate.is_baseline]
        if len(baselines) != 1:
            raise ValueError(f"Scene must contain exactly one baseline candidate; found {len(baselines)}")
        baseline = baselines[0]
        rejected: list[Mapping[str, Any]] = []
        eligible: list[CandidateScore] = []
        for candidate in candidates:
            if candidate is baseline:
                continue
            reasons = self._rejection_reasons(candidate)
            if reasons:
                rejected.append({"level": candidate.level, "alpha": candidate.alpha, "score": candidate.overall_score, "reasons": reasons})
            else:
                eligible.append(candidate)

        if not eligible:
            return self._baseline_result(baseline, rejected, "no_safe_candidate")
        best_score = max(candidate.overall_score for candidate in eligible)
        tied = [candidate for candidate in eligible if best_score - candidate.overall_score <= self.config.score_tolerance]
        best = min(tied, key=lambda item: (abs(item.alpha), item.alpha, item.level))
        delta = float(best.overall_score - baseline.overall_score)
        if delta + self.config.score_tolerance < self.config.min_improvement:
            return self._baseline_result(baseline, rejected, "improvement_below_margin", best_candidate=best)
        return TeacherSelection(
            scene_id=baseline.scene_id,
            baseline_path=baseline.output_path,
            selected_path=best.output_path,
            baseline_score=baseline.overall_score,
            selected_score=best.overall_score,
            selected_alpha=best.alpha,
            selected_level=best.level,
            score_delta=delta,
            confidence=best.confidence,
            sample_weight=self._weight(best, delta),
            distribution_status=best.distribution_status,
            status="improved",
            reason="best_safe_candidate",
            rejected_candidates=tuple(rejected),
            metadata={
                "baseline_level": baseline.level,
                "baseline_candidate": baseline.to_dict(),
                "selected_candidate": best.to_dict(),
            },
        )

    def _baseline_result(
        self,
        baseline: CandidateScore,
        rejected: Sequence[Mapping[str, Any]],
        reason: str,
        *,
        best_candidate: CandidateScore | None = None,
    ) -> TeacherSelection:
        metadata: dict[str, Any] = {
            "baseline_level": baseline.level,
            "baseline_candidate": baseline.to_dict(),
        }
        if best_candidate is not None:
            metadata["best_alternative"] = best_candidate.to_dict()
        return TeacherSelection(
            scene_id=baseline.scene_id,
            baseline_path=baseline.output_path,
            selected_path=baseline.output_path,
            baseline_score=baseline.overall_score,
            selected_score=baseline.overall_score,
            selected_alpha=0.0,
            selected_level=baseline.level,
            score_delta=0.0,
            confidence=baseline.confidence,
            sample_weight=1.0,
            distribution_status=baseline.distribution_status,
            status="baseline_anchor",
            reason=reason,
            rejected_candidates=tuple(rejected),
            metadata=metadata,
        )

    def select_all(self, candidates: Iterable[CandidateScore]) -> list[TeacherSelection]:
        grouped: dict[str, list[CandidateScore]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.scene_id].append(candidate)
        return [self.select_scene(grouped[scene_id]) for scene_id in sorted(grouped)]
