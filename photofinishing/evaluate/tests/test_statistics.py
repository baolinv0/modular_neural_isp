import math

import numpy as np

from photofinishing.evaluate.statistics import summarize_improvements


def test_summary_reports_core_statistics_and_deterministic_ci():
    values = np.array([0.1, 0.2, 0.3, -0.1, 0.4], dtype=np.float64)
    first = summarize_improvements(values, bootstrap_samples=500, seed=7)
    second = summarize_improvements(values, bootstrap_samples=500, seed=7)
    assert first == second
    assert first["count"] == 5
    assert math.isclose(first["mean"], 0.18, abs_tol=1e-12)
    assert math.isclose(first["median"], 0.2, abs_tol=1e-12)
    assert math.isclose(first["win_rate"], 0.8, abs_tol=1e-12)
    assert math.isclose(first["negative_improvement_rate"], 0.2, abs_tol=1e-12)
    assert first["p10"] <= first["median"]
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def test_summary_ignores_nan_and_marks_empty_input():
    result = summarize_improvements(np.array([np.nan, 0.2, np.nan]), bootstrap_samples=100, seed=1)
    assert result["count"] == 1
    assert result["median"] == 0.2
    empty = summarize_improvements(np.array([np.nan]), bootstrap_samples=100, seed=1)
    assert empty["count"] == 0
    assert math.isnan(empty["mean"])
    assert math.isnan(empty["ci95_low"])
