"""Evidence-based decisions for Stage-1 and Stage-2 evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .statistics import summarize_improvements


@dataclass(frozen=True)
class DecisionThresholds:
    min_count: int = 20
    min_win_rate: float = 0.60
    min_auxiliary_color_metrics: int = 2
    ev_regression_tolerance: float = 0.02
    tone_regression_tolerance: float = 0.01
    clipping_regression_tolerance: float = 0.005
    high_slice_negative_rate: float = 0.50


def _has_evidence(summary: Optional[Mapping[str, object]], thresholds: DecisionThresholds) -> bool:
    if not summary:
        return False
    try:
        return (
            int(summary["count"]) >= thresholds.min_count
            and float(summary["median"]) > 0
            and float(summary["win_rate"]) >= thresholds.min_win_rate
            and float(summary["ci95_low"]) >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _not_regressed(
    summary: Optional[Mapping[str, object]],
    tolerance: float,
    thresholds: DecisionThresholds,
) -> bool:
    if not summary:
        return False
    try:
        return int(summary["count"]) >= thresholds.min_count and float(summary["median"]) >= -tolerance
    except (KeyError, TypeError, ValueError):
        return False


def decide_stage1(
    metric_summaries: Mapping[str, Mapping[str, object]],
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> dict[str, object]:
    """Answers whether Stage 1 corrects brightness and global tone."""

    brightness = _has_evidence(metric_summaries.get("absolute_ev_error"), thresholds)
    distribution = _has_evidence(metric_summaries.get("log_luma_quantile_mae"), thresholds)
    tone = _has_evidence(metric_summaries.get("tone_shape_mae"), thresholds)
    clipping = _not_regressed(
        metric_summaries.get("clipping_ratio_error"), thresholds.clipping_regression_tolerance, thresholds
    )
    positives = sum((brightness, distribution, tone))
    effective = brightness and positives >= 2 and clipping
    count = int(metric_summaries.get("absolute_ev_error", {}).get("count", 0))
    if effective:
        status = "effective"
        reason = "Stage 1 reduces brightness error and at least one additional tone-distribution error without clipping regression."
    elif count < thresholds.min_count:
        status = "inconclusive"
        reason = "Stage-1 sample count is below the configured evidence threshold."
    else:
        status = "not_effective"
        reason = "Stage 1 lacks consistent paired evidence for brightness and tone improvement."
    return {
        "status": status,
        "answers_bias_problem": bool(effective),
        "brightness_improved": brightness,
        "distribution_improved": distribution,
        "tone_shape_improved": tone,
        "clipping_preserved": clipping,
        "reason": reason,
    }


def decide_stage2_necessity(
    stage1_color_distances: np.ndarray,
    reference_noise_floor: Optional[np.ndarray],
    thresholds: DecisionThresholds = DecisionThresholds(),
    *,
    bootstrap_samples: int = 5000,
    seed: int = 42,
) -> dict[str, object]:
    """Compares Stage-1 color residual with repeated-reference variability."""

    if reference_noise_floor is None:
        return {
            "status": "undetermined",
            "reason": "No repeated reference is available; Stage-1 color residual cannot be separated from capture/non-alignment noise.",
            "residual_above_noise": None,
        }
    stage1 = np.asarray(stage1_color_distances, dtype=np.float64).reshape(-1)
    noise = np.asarray(reference_noise_floor, dtype=np.float64).reshape(-1)
    if stage1.shape != noise.shape:
        raise ValueError("stage1_color_distances and reference_noise_floor must have identical shape")
    summary = summarize_improvements(stage1 - noise, bootstrap_samples=bootstrap_samples, seed=seed)
    needed = _has_evidence(summary, thresholds)
    if needed:
        status = "needed"
        reason = "Stage-1 color residual is consistently above the repeated-reference noise floor."
    elif int(summary["count"]) < thresholds.min_count:
        status = "inconclusive"
        reason = "Too few repeated-reference pairs to establish whether Stage 2 is necessary."
    else:
        status = "not_needed_or_not_measurable"
        reason = "Stage-1 color residual is not reliably above the repeated-reference noise floor."
    return {
        "status": status,
        "residual_above_noise": needed,
        "residual_minus_noise": summary,
        "reason": reason,
    }


def decide_stage2_effectiveness(
    metric_summaries: Mapping[str, Mapping[str, object]],
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> dict[str, object]:
    """Answers whether Stage 2 adds color gain while preserving Stage-1 tone."""

    primary = metric_summaries.get("luminance_conditioned_cbcr_swd")
    primary_effective = _has_evidence(primary, thresholds)
    auxiliary_names = (
        "cbcr_swd",
        "chroma_mean_error",
        "chroma_covariance_error",
        "saturation_w1",
        "neutral_axis_error",
        "semantic_skin_lab_swd",
        "semantic_sky_lab_swd",
        "semantic_vegetation_lab_swd",
    )
    auxiliary_effective = [name for name in auxiliary_names if _has_evidence(metric_summaries.get(name), thresholds)]
    luminance_preserved = all((
        _not_regressed(metric_summaries.get("absolute_ev_error"), thresholds.ev_regression_tolerance, thresholds),
        _not_regressed(metric_summaries.get("tone_shape_mae"), thresholds.tone_regression_tolerance, thresholds),
        _not_regressed(
            metric_summaries.get("clipping_ratio_error"), thresholds.clipping_regression_tolerance, thresholds
        ),
    ))
    effective = (
        primary_effective
        and len(auxiliary_effective) >= thresholds.min_auxiliary_color_metrics
        and luminance_preserved
    )
    count = int((primary or {}).get("count", 0))
    if effective:
        status = "effective"
        reason = "Stage 2 improves brightness-conditioned chroma plus auxiliary color metrics without luminance regression."
    elif count < thresholds.min_count:
        status = "inconclusive"
        reason = "Stage-2 evidence is underpowered at the configured sample threshold."
    elif not luminance_preserved:
        status = "rejected_luminance_regression"
        reason = "Color distance may improve, but Stage 2 regresses Stage-1 luminance/tone safety gates."
    else:
        status = "not_effective"
        reason = "Stage 2 does not show sufficiently consistent incremental color improvement over Stage 1."
    return {
        "status": status,
        "primary": dict(primary or {}),
        "primary_color_improved": primary_effective,
        "auxiliary_color_metrics_improved": auxiliary_effective,
        "luminance_preserved": luminance_preserved,
        "reason": reason,
    }


def recommend_stage2_variant(variant_decisions: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Prefers the effective Stage-2 variant with the lowest regression risk."""

    candidates = []
    for name, decision in variant_decisions.items():
        if decision.get("status") != "effective":
            continue
        primary = decision.get("primary", {})
        try:
            candidates.append((
                float(primary.get("negative_improvement_rate", 1.0)),
                -float(primary.get("p10", float("-inf"))),
                -float(primary.get("median", float("-inf"))),
                name,
            ))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return {
            "recommended": None,
            "reason": "No Stage-2 variant passed the effectiveness and luminance-preservation gates.",
        }
    candidates.sort()
    selected = candidates[0][3]
    return {
        "recommended": selected,
        "reason": "Selected the effective variant with the lowest negative-improvement risk, then the strongest worst-tail and median gain.",
    }


def recommend_data_expansion(
    *,
    sample_count: int,
    stage2_necessity: Mapping[str, object],
    stage2_decisions: Mapping[str, Mapping[str, object]],
    group_primary_medians: Mapping[str, float],
    slice_failures: Mapping[str, Mapping[str, object]],
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> dict[str, object]:
    """Separates the need for more evaluation evidence from more training coverage."""

    inconclusive = any(decision.get("status") == "inconclusive" for decision in stage2_decisions.values())
    expand_evaluation = sample_count < thresholds.min_count or inconclusive
    evaluation_reasons = []
    if sample_count < thresholds.min_count:
        evaluation_reasons.append(
            f"Only {sample_count} evaluation samples are available; threshold is {thresholds.min_count}."
        )
    if inconclusive:
        evaluation_reasons.append("At least one Stage-2 confidence interval is inconclusive.")

    finite_medians = [float(value) for value in group_primary_medians.values() if np.isfinite(float(value))]
    cross_group_sign_disagreement = bool(finite_medians) and min(finite_medians) < 0 < max(finite_medians)
    target_slices = sorted(
        name for name, payload in slice_failures.items()
        if int(payload.get("count", 0)) > 0
        and float(payload.get("negative_rate", 0.0)) >= thresholds.high_slice_negative_rate
    )
    residual_needed = stage2_necessity.get("status") == "needed"
    expand_training = residual_needed and (cross_group_sign_disagreement or bool(target_slices))
    training_reasons = []
    if cross_group_sign_disagreement:
        training_reasons.append("Stage-2 primary gain changes sign across experiment groups/seeds.")
    if target_slices:
        training_reasons.append("Failures concentrate in uncovered or unstable scene slices: " + ", ".join(target_slices))
    if not residual_needed:
        training_reasons.append("Training expansion is not justified until Stage-2 residual is shown to exceed reference noise.")
    if residual_needed and not expand_training:
        training_reasons.append("No cross-group sign instability or concentrated failure slice currently requires more training data.")

    return {
        "expand_evaluation_data": expand_evaluation,
        "evaluation_reasons": evaluation_reasons,
        "expand_training_data": expand_training,
        "training_reasons": training_reasons,
        "target_slices": target_slices,
    }
