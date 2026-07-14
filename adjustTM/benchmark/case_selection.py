from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def _reference_scene_errors(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row.get("method") != method:
            continue
        if row.get("semantic_group") == "real_camera":
            continue
        value = row.get("metrics", {}).get("log_luma_mae")
        if value is not None and math.isfinite(float(value)):
            grouped[str(row["scene_id"])].append(float(value))
    return {scene: _mean(values) for scene, values in grouped.items()}


def _rank_badness(values: Mapping[str, float], *, higher_is_bad: bool) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values, key=lambda scene: (values[scene], scene), reverse=higher_is_bad)
    count = len(ordered)
    if count == 1:
        return {ordered[0]: 0.0}
    return {scene: 1.0 - index / (count - 1) for index, scene in enumerate(ordered)}


def _reference_failure_badness(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    metric_spec = {
        "log_luma_mae": (True, 1.0),
        "gradient_mae": (True, 0.8),
        "chroma_rg_mae_to_gt": (True, 0.8),
        "lpips": (True, 1.0),
        "clip_ratio": (True, 0.3),
        "deep_shadow_ratio": (True, 0.3),
        "rgb_ssim": (False, 0.8),
        "luma_ssim": (False, 0.8),
        "rgb_psnr": (False, 0.5),
        "luma_psnr": (False, 0.5),
    }
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("method") != method or row.get("semantic_group") == "real_camera":
            continue
        scene_id = str(row["scene_id"])
        metrics = row.get("metrics", {})
        for metric in metric_spec:
            value = metrics.get(metric)
            if value is not None and math.isfinite(float(value)):
                grouped[scene_id][metric].append(float(value))
    per_metric: dict[str, dict[str, float]] = defaultdict(dict)
    for scene_id, metrics in grouped.items():
        for metric, values in metrics.items():
            per_metric[metric][scene_id] = _mean(values)
    numerator: dict[str, float] = defaultdict(float)
    denominator: dict[str, float] = defaultdict(float)
    for metric, scene_values in per_metric.items():
        higher_is_bad, weight = metric_spec[metric]
        ranks = _rank_badness(scene_values, higher_is_bad=higher_is_bad)
        for scene_id, rank in ranks.items():
            numerator[scene_id] += weight * rank
            denominator[scene_id] += weight
    return {
        scene_id: numerator[scene_id] / denominator[scene_id]
        for scene_id in numerator
        if denominator[scene_id] > 0
    }


def _control_badness(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    result: dict[str, float] = {}
    weights = {
        "violation_magnitude": 1.0,
        "violation_rate": 1.0,
        "jump_rate": 1.0,
        "dead_zone_rate": 0.5,
        "smoothness": 0.25,
    }
    for row in records:
        if row.get("method") != method:
            continue
        metrics = row.get("metrics", {})
        score = sum(float(metrics.get(key, 0.0)) * weight for key, weight in weights.items())
        result[str(row["scene_id"])] = score
    return result


def _naturalness_scores(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        row_method = row.get("method")
        row_scene = row.get("scene_id")
        task_id = str(row.get("task_id", ""))
        if (row_method is None or row_scene is None) and task_id:
            parts = task_id.split(":", 3)
            if len(parts) == 4:
                _, row_scene, _, row_method = parts
        if row_method != method or row_scene is None:
            continue
        task = str(row.get("kind", row.get("task", row.get("study_type", "")))).lower()
        if task and "natural" not in task:
            continue
        candidates: list[float] = []
        aggregate = row.get("aggregate", {})
        score_root = (
            aggregate.get("scores", row.get("scores", {}))
            if isinstance(aggregate, Mapping)
            else row.get("scores", {})
        )
        if isinstance(score_root, Mapping):
            for key in ("overall_naturalness", "artifact_absence", "exposure_naturalness"):
                value = score_root.get(key)
                if isinstance(value, Mapping):
                    value = value.get("median", value.get("mean"))
                if value is not None:
                    candidates.append(float(value))
        if "overall_naturalness" in row:
            candidates.append(float(row["overall_naturalness"]))
        if candidates:
            grouped[str(row_scene)].append(_mean(candidates))
    return {scene: _mean(values) for scene, values in grouped.items()}


def _take_distinct(order: Sequence[str], count: int, used: set[str]) -> list[str]:
    chosen = [scene for scene in order if scene not in used][: max(0, count)]
    if len(chosen) < count:
        chosen.extend([scene for scene in order if scene not in chosen][: count - len(chosen)])
    used.update(chosen)
    return chosen


def select_case_sets(
    reference_records: Sequence[Mapping[str, Any]],
    control_records: Sequence[Mapping[str, Any]],
    *,
    focus_method: str,
    comparison_baseline: str,
    representative_count: int = 6,
    best_count: int = 6,
    failure_count: int = 6,
    disagreement_count: int = 6,
    vlm_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    focus = _reference_scene_errors(reference_records, focus_method)
    baseline = _reference_scene_errors(reference_records, comparison_baseline)
    scenes = sorted(set(focus) & set(baseline))
    if not scenes:
        raise ValueError("No shared scene-level log_luma_mae records for focus method and comparison baseline")
    improvement = {scene: baseline[scene] - focus[scene] for scene in scenes}
    control_raw = _control_badness(control_records, focus_method)
    control = _rank_badness(control_raw, higher_is_bad=True)
    reference_badness = _reference_failure_badness(reference_records, focus_method)
    natural = _naturalness_scores(vlm_records or [], focus_method)
    natural_badness = _rank_badness(natural, higher_is_bad=False)

    best_order = sorted(scenes, key=lambda scene: (-improvement[scene], focus[scene], scene))

    def failure_score(scene: str) -> float:
        weighted = []
        if scene in reference_badness:
            weighted.append((reference_badness[scene], 0.55))
        if scene in control:
            weighted.append((control[scene], 0.30))
        if scene in natural_badness:
            weighted.append((natural_badness[scene], 0.15))
        if not weighted:
            return focus[scene]
        return sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)

    failure_order = sorted(scenes, key=lambda scene: (-failure_score(scene), improvement[scene], scene))
    median_error = float(np.median([focus[scene] for scene in scenes]))
    representative_order = sorted(
        scenes,
        key=lambda scene: (abs(focus[scene] - median_error), abs(improvement[scene]), scene),
    )

    used: set[str] = set()
    selected = {
        "best_improvements": _take_distinct(best_order, best_count, used),
        "failure_cases": _take_distinct(failure_order, failure_count, used),
        "representative_cases": _take_distinct(representative_order, representative_count, used),
    }

    disagreement_scenes = sorted(set(scenes) & set(natural))
    if disagreement_scenes:
        fidelity_order = sorted(disagreement_scenes, key=lambda scene: (focus[scene], scene))
        natural_order = sorted(disagreement_scenes, key=lambda scene: (-natural[scene], scene))
        fidelity_rank = {scene: index for index, scene in enumerate(fidelity_order)}
        natural_rank = {scene: index for index, scene in enumerate(natural_order)}
        disagreement_order = sorted(
            disagreement_scenes,
            key=lambda scene: (-abs(fidelity_rank[scene] - natural_rank[scene]), scene),
        )
        selected["metric_vlm_disagreement"] = _take_distinct(
            disagreement_order, disagreement_count, used
        )
    else:
        selected["metric_vlm_disagreement"] = []
    return selected
