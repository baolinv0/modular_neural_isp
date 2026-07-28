"""Paired incremental-improvement statistics."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def summarize_improvements(
    values: Iterable[float] | np.ndarray,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Summarizes lower-is-better metric improvements.

    Positive values mean the later stage is closer to the non-aligned reference.
    The confidence interval is a percentile bootstrap interval over the paired
    sample-level mean improvement.
    """

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        nan = float("nan")
        return {
            "count": 0,
            "mean": nan,
            "median": nan,
            "std": nan,
            "win_rate": nan,
            "negative_improvement_rate": nan,
            "p10": nan,
            "ci95_low": nan,
            "ci95_high": nan,
        }

    rng = np.random.default_rng(seed)
    if array.size == 1:
        bootstrap_means = np.full(bootstrap_samples, array[0], dtype=np.float64)
    else:
        indices = rng.integers(0, array.size, size=(bootstrap_samples, array.size))
        bootstrap_means = array[indices].mean(axis=1)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "win_rate": float(np.mean(array > 0)),
        "negative_improvement_rate": float(np.mean(array < 0)),
        "p10": float(np.quantile(array, 0.10)),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
    }


def finite_median(values: Iterable[float] | np.ndarray) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def finite_mean(values: Iterable[float] | np.ndarray) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")
