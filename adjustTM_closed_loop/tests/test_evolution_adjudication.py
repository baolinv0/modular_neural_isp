from __future__ import annotations

import json
from pathlib import Path

from adjustTM_closed_loop.evolution.adjudication import AdjudicationConfig, adjudicate_evolution
from adjustTM_closed_loop.evolution.schemas import DistributionStatus, TeacherSelection
from adjustTM_closed_loop.evolution.teacher_manifest import build_teacher_manifest


def selection(scene: str, status: str) -> TeacherSelection:
    alpha = 0.25 if status == "improved" else 0.0
    return TeacherSelection(
        scene_id=scene, baseline_path=f"/b/{scene}", selected_path=f"/t/{scene}", baseline_score=0.5,
        selected_score=0.7 if alpha else 0.5, selected_alpha=alpha, selected_level="x", score_delta=0.2 if alpha else 0,
        confidence=1, sample_weight=1, distribution_status=DistributionStatus.IN_DOMAIN, status=status,
        reason="x", rejected_candidates=(), metadata={"teacher_evaluator_id": "teacher-iqa-v1"},
    )


def write_scores(path: Path, values: dict[str, float], failures: dict[str, list[str]] | None = None, evaluator_id: str = "arbiter-iqa-v1") -> None:
    rows = []
    for scene, score in values.items():
        rows.append({"scene_id": scene, "score": score, "hard_failures": (failures or {}).get(scene, []), "evaluator_id": evaluator_id})
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def make_manifest(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"; inputs.mkdir()
    items = [selection("target1.png", "improved"), selection("target2.png", "improved"),
             selection("anchor1.png", "baseline_anchor"), selection("anchor2.png", "baseline_anchor")]
    for item in items:
        (inputs / item.scene_id).write_bytes(b"x")
    path = tmp_path / "teacher.jsonl"
    build_teacher_manifest(items, input_dir=inputs, output_path=path, validation_fraction=0, test_fraction=0)
    return path


def test_accepts_target_improvement_without_anchor_regression(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    baseline, student = tmp_path / "b.jsonl", tmp_path / "s.jsonl"
    write_scores(baseline, {"target1.png": .6, "target2.png": .5, "anchor1.png": .8, "anchor2.png": .75})
    write_scores(student, {"target1.png": .72, "target2.png": .62, "anchor1.png": .8, "anchor2.png": .74})
    report = adjudicate_evolution(manifest, baseline, student, AdjudicationConfig(
        min_target_mean_delta=.05, min_target_win_rate=.5, min_overall_mean_delta=0,
        max_anchor_mean_regression=.02, max_anchor_regression_rate=.5, evaluation_splits=("train",),
    ))
    assert report["decision"] == "ACCEPT"
    assert report["target_slice"]["mean_delta"] > .1


def test_rejects_when_anchor_regresses_even_if_target_improves(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    baseline, student = tmp_path / "b.jsonl", tmp_path / "s.jsonl"
    write_scores(baseline, {"target1.png": .6, "target2.png": .5, "anchor1.png": .8, "anchor2.png": .75})
    write_scores(student, {"target1.png": .8, "target2.png": .7, "anchor1.png": .55, "anchor2.png": .5})
    report = adjudicate_evolution(manifest, baseline, student, AdjudicationConfig(evaluation_splits=("train",)))
    assert report["decision"] == "REJECT"
    assert any("anchor" in reason for reason in report["reasons"])


def test_rejects_new_hard_failure(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    baseline, student = tmp_path / "b.jsonl", tmp_path / "s.jsonl"
    values = {"target1.png": .6, "target2.png": .5, "anchor1.png": .8, "anchor2.png": .75}
    write_scores(baseline, values)
    write_scores(student, {k: v + .1 for k, v in values.items()}, {"target1.png": ["highlight_clipping"]})
    report = adjudicate_evolution(manifest, baseline, student, AdjudicationConfig(evaluation_splits=("train",)))
    assert report["decision"] == "REJECT"
    assert report["new_hard_failure_count"] == 1


def test_fails_for_missing_scene_score(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    baseline, student = tmp_path / "b.jsonl", tmp_path / "s.jsonl"
    write_scores(baseline, {"target1.png": .6})
    write_scores(student, {"target1.png": .7})
    try:
        adjudicate_evolution(manifest, baseline, student, AdjudicationConfig(evaluation_splits=("train",)))
    except ValueError as exc:
        assert "Missing quality scores" in str(exc)
    else:
        raise AssertionError("Expected missing score failure")


def test_same_evaluator_cannot_formally_accept(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    baseline, student = tmp_path / "b.jsonl", tmp_path / "s.jsonl"
    base = {"target1.png": .6, "target2.png": .5, "anchor1.png": .8, "anchor2.png": .75}
    write_scores(baseline, base, evaluator_id="teacher-iqa-v1")
    write_scores(student, {k: v + (.1 if k.startswith("target") else 0) for k, v in base.items()}, evaluator_id="teacher-iqa-v1")
    report = adjudicate_evolution(manifest, baseline, student, AdjudicationConfig(evaluation_splits=("train",)))
    assert report["decision"] == "REJECT"
    assert report["evaluator_provenance"]["independence"] == "overlapping"
