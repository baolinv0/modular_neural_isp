import numpy as np

from photofinishing.evaluate.decision import (
    DecisionThresholds,
    decide_stage1,
    decide_stage2_effectiveness,
    decide_stage2_necessity,
    recommend_data_expansion,
    recommend_stage2_variant,
)
from photofinishing.evaluate.statistics import summarize_improvements


def _summary(values):
    return summarize_improvements(np.asarray(values, np.float64), bootstrap_samples=1000, seed=3)


def test_stage1_requires_brightness_and_tone_evidence():
    summaries = {
        "absolute_ev_error": _summary([0.2, 0.3, 0.1, 0.2, 0.4]),
        "log_luma_quantile_mae": _summary([0.1, 0.2, 0.1, 0.2, 0.3]),
        "tone_shape_mae": _summary([0.05, 0.1, 0.08, 0.03, 0.07]),
        "clipping_ratio_error": _summary([0.0, 0.01, 0.0, 0.0, 0.01]),
    }
    result = decide_stage1(summaries, DecisionThresholds(min_count=5, min_win_rate=0.6))
    assert result["status"] == "effective"
    assert result["answers_bias_problem"] is True


def test_stage2_necessity_uses_reference_repeat_noise_floor():
    stage1 = np.array([0.12, 0.10, 0.15, 0.11, 0.14])
    noise = np.array([0.02, 0.03, 0.02, 0.02, 0.03])
    needed = decide_stage2_necessity(stage1, noise, DecisionThresholds(min_count=5, min_win_rate=0.6))
    assert needed["status"] == "needed"
    unavailable = decide_stage2_necessity(stage1, None, DecisionThresholds(min_count=5))
    assert unavailable["status"] == "undetermined"


def test_stage2_effectiveness_requires_color_gain_and_luma_guard():
    summaries = {
        "luminance_conditioned_cbcr_swd": _summary([0.03, 0.04, 0.02, 0.05, 0.03]),
        "cbcr_swd": _summary([0.02, 0.03, 0.01, 0.02, 0.02]),
        "chroma_mean_error": _summary([0.01, 0.02, 0.01, 0.01, 0.02]),
        "saturation_w1": _summary([0.02, 0.01, 0.03, 0.02, 0.01]),
        "absolute_ev_error": _summary([0.0, 0.0, -0.005, 0.0, 0.0]),
        "tone_shape_mae": _summary([0.0, -0.002, 0.0, 0.0, 0.0]),
        "clipping_ratio_error": _summary([0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    result = decide_stage2_effectiveness(summaries, DecisionThresholds(min_count=5, min_win_rate=0.6))
    assert result["status"] == "effective"
    assert result["luminance_preserved"] is True


def test_variant_selection_prefers_lower_risk_effective_model():
    decisions = {
        "affine": {
            "status": "effective",
            "primary": {"median": 0.03, "negative_improvement_rate": 0.1, "p10": 0.0},
        },
        "full_lut": {
            "status": "effective",
            "primary": {"median": 0.035, "negative_improvement_rate": 0.3, "p10": -0.02},
        },
    }
    result = recommend_stage2_variant(decisions)
    assert result["recommended"] == "affine"
    assert "risk" in result["reason"].lower()


def test_data_expansion_distinguishes_evaluation_and_training_needs():
    result = recommend_data_expansion(
        sample_count=13,
        stage2_necessity={"status": "needed"},
        stage2_decisions={"affine": {"status": "inconclusive"}},
        group_primary_medians={"seed0": 0.03, "seed1": -0.02},
        slice_failures={"night": {"count": 3, "negative_rate": 0.67}},
        thresholds=DecisionThresholds(min_count=20),
    )
    assert result["expand_evaluation_data"] is True
    assert result["expand_training_data"] is True
    assert "night" in result["target_slices"]
