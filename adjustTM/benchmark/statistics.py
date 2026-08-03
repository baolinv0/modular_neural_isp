from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np


def aggregate_scene_level_records(records: Iterable[Mapping[str, object]], *, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["scene_id"])].append(float(record[value_key]))
    return {scene_id: float(np.mean(values)) for scene_id, values in sorted(grouped.items())}


def bootstrap_mean_ci(values: Iterable[float], *, samples: int = 10000, seed: int = 42, confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        raise ValueError("Cannot bootstrap empty values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def worst_cvar(values: Iterable[float], *, higher_is_better: bool, fraction: float = 0.05) -> float:
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    if len(array) == 0:
        raise ValueError("Cannot compute CVaR of empty values")
    count = max(1, int(np.ceil(len(array) * fraction)))
    selected = array[:count] if higher_is_better else array[-count:]
    return float(np.mean(selected))


def paired_permutation_pvalue(deltas: np.ndarray, *, samples: int, seed: int) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    if len(deltas) == 0:
        raise ValueError("No paired deltas")
    observed = abs(float(np.mean(deltas)))
    if np.allclose(deltas, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(deltas)))
    permuted = np.abs((signs * deltas).mean(axis=1))
    return float((1 + np.sum(permuted >= observed)) / (samples + 1))


def paired_comparison(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    lower_is_better: bool,
    tie_tolerance: float = 1e-12,
    seed: int = 42,
    bootstrap_samples: int = 10000,
    permutation_samples: int = 10000,
) -> dict[str, float]:
    common = sorted(set(first) & set(second))
    if not common:
        raise ValueError("No common scenes for paired comparison")
    raw_delta = np.asarray([float(first[key]) - float(second[key]) for key in common], dtype=np.float64)
    beneficial = -raw_delta if lower_is_better else raw_delta
    wins = beneficial > tie_tolerance
    ties = np.abs(beneficial) <= tie_tolerance
    losses = beneficial < -tie_tolerance
    low, high = bootstrap_mean_ci(raw_delta, samples=bootstrap_samples, seed=seed)
    return {
        "scene_count": float(len(common)),
        "mean_delta": float(np.mean(raw_delta)),
        "median_delta": float(np.median(raw_delta)),
        "ci_low": low,
        "ci_high": high,
        "win_rate": float(np.mean(wins)),
        "tie_rate": float(np.mean(ties)),
        "loss_rate": float(np.mean(losses)),
        "permutation_p": paired_permutation_pvalue(raw_delta, samples=permutation_samples, seed=seed),
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, float(pvalue) * (count - index))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
