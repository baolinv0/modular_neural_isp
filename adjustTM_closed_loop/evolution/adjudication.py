from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .teacher_manifest import load_teacher_manifest


@dataclass(frozen=True)
class AdjudicationConfig:
    """Acceptance thresholds for a distilled automatic TM policy.

    Positive values are expressed in the evaluator's score units. Anchor
    regressions are reported as positive magnitudes, e.g. a mean delta of
    -0.02 is a 0.02 mean regression.
    """

    min_target_mean_delta: float = 0.03
    min_target_win_rate: float = 0.50
    min_overall_mean_delta: float = 0.0
    max_anchor_mean_regression: float = 0.01
    max_anchor_regression_rate: float = 0.10
    regression_epsilon: float = 0.01
    reject_new_hard_failures: bool = True
    evaluation_splits: tuple[str, ...] = ("test",)
    require_anchor_scenes: bool = True
    require_independent_evaluator: bool = True

    def __post_init__(self) -> None:
        for name in ("min_target_win_rate", "max_anchor_regression_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.regression_epsilon < 0:
            raise ValueError("regression_epsilon must be non-negative")
        if self.max_anchor_mean_regression < 0:
            raise ValueError("max_anchor_mean_regression must be non-negative")
        if not self.evaluation_splits:
            raise ValueError("evaluation_splits must not be empty")


def _iter_score_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("scores", "scenes", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                rows = []
                for scene_id, row in value.items():
                    item = dict(row) if isinstance(row, Mapping) else {"score": row}
                    item.setdefault("scene_id", scene_id)
                    rows.append(item)
                return rows
        # Also accept a direct scene_id -> score mapping.
        if payload and all(not isinstance(value, (list, tuple)) for value in payload.values()):
            rows = []
            for scene_id, value in payload.items():
                if isinstance(value, Mapping):
                    item = dict(value)
                    item.setdefault("scene_id", scene_id)
                else:
                    item = {"scene_id": scene_id, "score": value}
                rows.append(item)
            return rows
    raise ValueError(f"Unsupported quality score format: {path}")


def _load_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    scores: dict[str, dict[str, Any]] = {}
    for row in _iter_score_rows(path):
        if not isinstance(row, Mapping):
            raise ValueError(f"Quality score row must be an object: {row!r}")
        scene_id = row.get("scene_id", row.get("scene_name", row.get("id")))
        if scene_id is None:
            raise ValueError(f"Quality score row is missing scene_id: {row}")
        score_value = row.get("score", row.get("overall_score", row.get("quality_score")))
        if score_value is None:
            raise ValueError(f"Quality score row is missing score: {row}")
        scene_id = str(scene_id)
        if scene_id in scores:
            raise ValueError(f"Duplicate quality score for scene: {scene_id}")
        failures = row.get("hard_failures", row.get("fatal_defects", row.get("hard_gate_failures", []))) or []
        if isinstance(failures, str):
            failures = [failures]
        evaluator_id = row.get("evaluator_id", row.get("evaluator", row.get("model_id")))
        scores[scene_id] = {
            "score": float(score_value),
            "hard_failures": tuple(sorted(str(item) for item in failures)),
            "evaluator_id": str(evaluator_id) if evaluator_id is not None else None,
            "raw": dict(row),
        }
    return scores


def _summary(scene_ids: list[str], baseline: Mapping[str, Mapping[str, Any]], student: Mapping[str, Mapping[str, Any]], *, regression_epsilon: float) -> dict[str, Any]:
    deltas = [float(student[name]["score"]) - float(baseline[name]["score"]) for name in scene_ids]
    if not deltas:
        return {
            "count": 0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "win_rate": 0.0,
            "regression_rate": 0.0,
            "min_delta": 0.0,
            "max_delta": 0.0,
        }
    wins = sum(delta > 0 for delta in deltas)
    regressions = sum(delta < -regression_epsilon for delta in deltas)
    return {
        "count": len(deltas),
        "mean_delta": float(statistics.fmean(deltas)),
        "median_delta": float(statistics.median(deltas)),
        "win_rate": wins / len(deltas),
        "regression_rate": regressions / len(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }


def adjudicate_evolution(
    teacher_manifest: str | Path,
    baseline_scores: str | Path,
    student_scores: str | Path,
    config: AdjudicationConfig | None = None,
) -> dict[str, Any]:
    """Judge whether a distilled automatic TM policy is safe to advance.

    The target slice contains records for which IQA selected a non-baseline
    teacher (``status == 'improved'``). All remaining records are treated as
    baseline anchors. Every manifest scene must have exactly one baseline and
    one student quality score.
    """

    config = config or AdjudicationConfig()
    all_records = load_teacher_manifest(teacher_manifest)
    if not all_records:
        raise ValueError("Teacher manifest is empty")
    allowed_splits = set(config.evaluation_splits)
    records = [record for record in all_records if record.split in allowed_splits]
    if not records:
        raise ValueError(f"Teacher manifest contains no records in evaluation splits {sorted(allowed_splits)}")
    baseline = _load_scores(baseline_scores)
    student = _load_scores(student_scores)
    manifest_ids = [record.scene_id for record in records]
    missing_baseline = sorted(set(manifest_ids) - set(baseline))
    missing_student = sorted(set(manifest_ids) - set(student))
    if missing_baseline or missing_student:
        raise ValueError(
            "Missing quality scores: "
            f"baseline={missing_baseline[:10]}, student={missing_student[:10]}"
        )

    target_ids = [record.scene_id for record in records if record.status == "improved"]
    anchor_ids = [record.scene_id for record in records if record.status != "improved"]
    if not target_ids:
        raise ValueError("Evaluation split contains no improved target scenes")
    if config.require_anchor_scenes and not anchor_ids:
        raise ValueError("Evaluation split contains no baseline anchor scenes")

    target = _summary(target_ids, baseline, student, regression_epsilon=config.regression_epsilon)
    anchors = _summary(anchor_ids, baseline, student, regression_epsilon=config.regression_epsilon)
    overall = _summary(manifest_ids, baseline, student, regression_epsilon=config.regression_epsilon)

    new_hard_failures: list[dict[str, Any]] = []
    for scene_id in manifest_ids:
        baseline_failures = set(baseline[scene_id]["hard_failures"])
        student_failures = set(student[scene_id]["hard_failures"])
        new = sorted(student_failures - baseline_failures)
        if new:
            new_hard_failures.append({"scene_id": scene_id, "failures": new})

    teacher_evaluator_ids = {
        str(record.metadata.get("teacher_evaluator_id"))
        for record in records
        if record.metadata.get("teacher_evaluator_id")
    }
    evaluation_evaluator_ids = {
        str(item["evaluator_id"])
        for item in list(baseline.values()) + list(student.values())
        if item.get("evaluator_id")
    }
    if teacher_evaluator_ids and evaluation_evaluator_ids and teacher_evaluator_ids.isdisjoint(evaluation_evaluator_ids):
        evaluator_independence = "independent"
    elif teacher_evaluator_ids and evaluation_evaluator_ids:
        evaluator_independence = "overlapping"
    else:
        evaluator_independence = "unknown"

    reasons: list[str] = []
    if config.require_independent_evaluator and evaluator_independence != "independent":
        reasons.append(f"independent evaluator requirement not met: {evaluator_independence}")
    if target["mean_delta"] < config.min_target_mean_delta:
        reasons.append(
            f"target mean delta {target['mean_delta']:.6f} is below {config.min_target_mean_delta:.6f}"
        )
    if target["win_rate"] < config.min_target_win_rate:
        reasons.append(
            f"target win rate {target['win_rate']:.6f} is below {config.min_target_win_rate:.6f}"
        )
    if overall["mean_delta"] < config.min_overall_mean_delta:
        reasons.append(
            f"overall mean delta {overall['mean_delta']:.6f} is below {config.min_overall_mean_delta:.6f}"
        )
    anchor_mean_regression = max(0.0, -float(anchors["mean_delta"]))
    if anchor_mean_regression > config.max_anchor_mean_regression:
        reasons.append(
            f"anchor mean regression {anchor_mean_regression:.6f} exceeds {config.max_anchor_mean_regression:.6f}"
        )
    if anchors["count"] and anchors["regression_rate"] > config.max_anchor_regression_rate:
        reasons.append(
            f"anchor regression rate {anchors['regression_rate']:.6f} exceeds {config.max_anchor_regression_rate:.6f}"
        )
    if config.reject_new_hard_failures and new_hard_failures:
        reasons.append(f"new hard_failure count is {len(new_hard_failures)}")

    per_scene = []
    status_by_scene = {record.scene_id: record.status for record in records}
    for scene_id in manifest_ids:
        per_scene.append({
            "scene_id": scene_id,
            "role": "target" if status_by_scene[scene_id] == "improved" else "anchor",
            "baseline_score": baseline[scene_id]["score"],
            "student_score": student[scene_id]["score"],
            "delta": student[scene_id]["score"] - baseline[scene_id]["score"],
            "baseline_hard_failures": list(baseline[scene_id]["hard_failures"]),
            "student_hard_failures": list(student[scene_id]["hard_failures"]),
        })

    if reasons:
        decision = "REJECT"
    elif evaluator_independence == "independent":
        decision = "ACCEPT"
    else:
        decision = "PROVISIONAL_ACCEPT"
    return {
        "version": 1,
        "decision": decision,
        "reasons": reasons,
        "target_slice": target,
        "anchor_slice": anchors,
        "overall": overall,
        "anchor_mean_regression": anchor_mean_regression,
        "new_hard_failure_count": len(new_hard_failures),
        "new_hard_failures": new_hard_failures,
        "config": {
            "min_target_mean_delta": config.min_target_mean_delta,
            "min_target_win_rate": config.min_target_win_rate,
            "min_overall_mean_delta": config.min_overall_mean_delta,
            "max_anchor_mean_regression": config.max_anchor_mean_regression,
            "max_anchor_regression_rate": config.max_anchor_regression_rate,
            "regression_epsilon": config.regression_epsilon,
            "reject_new_hard_failures": config.reject_new_hard_failures,
            "evaluation_splits": list(config.evaluation_splits),
            "require_anchor_scenes": config.require_anchor_scenes,
            "require_independent_evaluator": config.require_independent_evaluator,
        },
        "evaluator_provenance": {
            "teacher_evaluator_ids": sorted(teacher_evaluator_ids),
            "evaluation_evaluator_ids": sorted(evaluation_evaluator_ids),
            "independence": evaluator_independence,
        },
        "all_manifest_scene_count": len(all_records),
        "evaluated_scene_count": len(records),
        "scenes": per_scene,
    }
