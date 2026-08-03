from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from .schemas import semantic_group


def reference_group(level_name: str) -> str:
    return semantic_group(level_name)


def control_curve_metrics(
    alphas: Iterable[float],
    mean_log_luminance: Iterable[float],
    *,
    violation_tolerance: float = 1e-7,
    dead_zone_threshold: float = 0.002,
    jump_ratio_threshold: float = 3.0,
) -> dict[str, float]:
    alpha = np.asarray(list(alphas), dtype=np.float64)
    luminance = np.asarray(list(mean_log_luminance), dtype=np.float64)
    if alpha.ndim != 1 or luminance.ndim != 1 or len(alpha) != len(luminance) or len(alpha) < 3:
        raise ValueError("Control curves require equal one-dimensional arrays with at least three points")
    if np.any(np.diff(alpha) <= 0):
        raise ValueError("Alphas must be strictly increasing")
    steps = np.diff(luminance)
    violations = steps < -violation_tolerance
    violation_magnitude = np.maximum(-steps, 0.0)
    dead_zones = np.abs(steps) < dead_zone_threshold
    absolute_steps = np.abs(steps)
    robust_step = float(np.median(absolute_steps)) if len(absolute_steps) else 0.0
    if robust_step <= 0.0:
        jumps = np.zeros_like(steps, dtype=bool)
    else:
        jumps = np.abs(steps) > jump_ratio_threshold * robust_step
    second = np.diff(luminance, n=2)
    total_range = float(luminance[-1] - luminance[0])
    smoothness = float(np.mean(np.abs(second)) / (abs(total_range) + 1e-12))
    zero_index = int(np.argmin(np.abs(alpha)))
    negative_range = float(luminance[zero_index] - luminance[0])
    positive_range = float(luminance[-1] - luminance[zero_index])
    range_balance = float(min(max(negative_range, 0.0), max(positive_range, 0.0)) / (max(negative_range, positive_range, 0.0) + 1e-12))
    longest_dead = 0
    current = 0
    for flag in dead_zones:
        current = current + 1 if flag else 0
        longest_dead = max(longest_dead, current)
    return {
        "violation_rate": float(np.mean(violations)),
        "violation_magnitude": float(np.sum(violation_magnitude)),
        "strict_scene_pass": float(not np.any(violations)),
        "dead_zone_rate": float(np.mean(dead_zones)),
        "longest_dead_zone_length": float(longest_dead),
        "jump_rate": float(np.mean(jumps)),
        "smoothness_second_difference": smoothness,
        "negative_range": negative_range,
        "positive_range": positive_range,
        "total_range": total_range,
        "range_balance": range_balance,
    }


def group_reference_records(records: Iterable[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[reference_group(str(record["level"]))].append(record)
    return dict(grouped)
