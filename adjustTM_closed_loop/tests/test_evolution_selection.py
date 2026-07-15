from __future__ import annotations

import json
from pathlib import Path

import pytest

from adjustTM_closed_loop.evolution.candidate_selection import (
    CandidateSelector,
    SelectionConfig,
    load_candidate_scores,
)
from adjustTM_closed_loop.evolution.schemas import CandidateScore, DistributionStatus


def candidate(alpha: float, score: float, **kwargs) -> CandidateScore:
    return CandidateScore(
        scene_id="scene.png",
        level=f"a_{alpha:+.2f}",
        alpha=alpha,
        output_path=f"/outputs/{alpha:+.2f}/scene.png",
        overall_score=score,
        confidence=kwargs.pop("confidence", 0.95),
        action=kwargs.pop("action", "KEEP"),
        hard_failures=tuple(kwargs.pop("hard_failures", ())),
        distribution_status=kwargs.pop("distribution_status", DistributionStatus.IN_DOMAIN),
        metrics=kwargs.pop("metrics", {}),
        metadata=kwargs,
    )


def test_selects_best_safe_candidate_above_margin() -> None:
    selector = CandidateSelector(SelectionConfig(min_improvement=0.03))
    result = selector.select_scene([
        candidate(0.0, 0.70),
        candidate(0.25, 0.76),
        candidate(0.50, 0.82),
    ])
    assert result.selected_alpha == 0.50
    assert result.status == "improved"
    assert result.score_delta == pytest.approx(0.12)
    assert result.sample_weight > 0.0


def test_hard_failure_never_wins_even_with_high_score() -> None:
    selector = CandidateSelector(SelectionConfig(min_improvement=0.01))
    result = selector.select_scene([
        candidate(0.0, 0.70),
        candidate(0.25, 0.78),
        candidate(0.75, 0.99, hard_failures=("highlight_clipping",)),
    ])
    assert result.selected_alpha == 0.25
    assert "highlight_clipping" in result.rejected_candidates[0]["reasons"]


def test_low_confidence_ood_and_rejected_candidates_fall_back_to_baseline() -> None:
    selector = CandidateSelector(SelectionConfig(min_improvement=0.02, min_confidence=0.8))
    result = selector.select_scene([
        candidate(0.0, 0.70),
        candidate(0.25, 0.90, confidence=0.4),
        candidate(0.50, 0.95, distribution_status=DistributionStatus.OOD),
        candidate(0.75, 0.97, action="REJECT"),
    ])
    assert result.selected_alpha == 0.0
    assert result.status == "baseline_anchor"
    assert result.reason == "no_safe_candidate"


def test_insufficient_improvement_returns_baseline() -> None:
    selector = CandidateSelector(SelectionConfig(min_improvement=0.05))
    result = selector.select_scene([candidate(0.0, 0.70), candidate(0.25, 0.74)])
    assert result.selected_alpha == 0.0
    assert result.reason == "improvement_below_margin"


def test_tie_break_prefers_smallest_absolute_alpha() -> None:
    selector = CandidateSelector(SelectionConfig(min_improvement=0.01, score_tolerance=1e-6))
    result = selector.select_scene([
        candidate(0.0, 0.70),
        candidate(-0.50, 0.85),
        candidate(0.25, 0.85),
    ])
    assert result.selected_alpha == 0.25


def test_boundary_candidate_is_downweighted() -> None:
    selector = CandidateSelector(SelectionConfig(min_improvement=0.01, allow_boundary=True, boundary_weight=0.25))
    result = selector.select_scene([
        candidate(0.0, 0.70),
        candidate(0.25, 0.85, distribution_status=DistributionStatus.BOUNDARY),
    ])
    assert result.selected_alpha == 0.25
    assert 0.0 < result.sample_weight <= 0.25


def test_load_candidate_scores_accepts_jsonl_aliases(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    rows = [
        {
            "scene_name": "a.png", "level": "a_000", "alpha": 0,
            "output_path": "/a0.png", "score": 0.7, "confidence": 0.9,
            "decision": {"action": "KEEP"},
        },
        {
            "scene_id": "a.png", "level": "a_p025", "alpha": 0.25,
            "image_path": "/a1.png", "overall_score": 0.8,
            "hard_gate": {"passed": True, "reasons": []},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    loaded = load_candidate_scores(path)
    assert len(loaded) == 2
    assert loaded[0].scene_id == "a.png"
    assert loaded[0].action == "KEEP"
    assert loaded[1].output_path == "/a1.png"


def test_requires_exactly_one_baseline_candidate() -> None:
    selector = CandidateSelector(SelectionConfig())
    with pytest.raises(ValueError, match="exactly one baseline"):
        selector.select_scene([candidate(0.25, 0.8)])


def test_review_and_unknown_distribution_are_not_pixel_teachers() -> None:
    selector = CandidateSelector(SelectionConfig(min_improvement=0.01))
    result = selector.select_scene([
        candidate(0.0, 0.70),
        candidate(0.25, 0.90, action="REVIEW"),
        candidate(0.50, 0.95, distribution_status=DistributionStatus.UNKNOWN),
    ])
    assert result.status == "baseline_anchor"
    rejected_reasons = {reason for item in result.rejected_candidates for reason in item["reasons"]}
    assert "action:REVIEW" in rejected_reasons
    assert "unknown_distribution" in rejected_reasons
