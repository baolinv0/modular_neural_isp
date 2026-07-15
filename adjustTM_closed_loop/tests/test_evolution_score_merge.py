from __future__ import annotations

import json
from pathlib import Path

import pytest

from adjustTM_closed_loop.evolution.score_merge import merge_candidate_scores


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_merge_inference_records_with_iqa_scores(tmp_path: Path) -> None:
    inference = tmp_path / "inference.jsonl"
    scores = tmp_path / "scores.jsonl"
    output = tmp_path / "merged.jsonl"
    write_jsonl(inference, [
        {"scene_id": "a.png", "method": "param", "level": "a_000", "alpha": 0, "output_path": "/a0.png"},
        {"scene_id": "a.png", "method": "param", "level": "a_p025", "alpha": .25, "output_path": "/a1.png"},
    ])
    write_jsonl(scores, [
        {"scene_id": "a.png", "method": "param", "level": "a_000", "score": .7,
         "confidence": .9, "action": "KEEP", "distribution_status": "in_domain"},
        {"scene_id": "a.png", "method": "param", "level": "a_p025", "overall_score": .8,
         "confidence": .95, "action": "KEEP", "distribution_status": "in_domain"},
    ])
    report = merge_candidate_scores(inference, scores, output, method="param")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["merged_count"] == 2
    assert rows[1]["output_path"] == "/a1.png"
    assert rows[1]["overall_score"] == .8


def test_merge_fails_on_missing_iqa_score(tmp_path: Path) -> None:
    inference = tmp_path / "inference.jsonl"
    scores = tmp_path / "scores.jsonl"
    write_jsonl(inference, [{"scene_id": "a", "method": "m", "level": "a_000", "alpha": 0, "output_path": "/x"}])
    write_jsonl(scores, [])
    with pytest.raises(ValueError, match="Missing IQA scores"):
        merge_candidate_scores(inference, scores, tmp_path / "out.jsonl", method="m")


def test_merge_rejects_duplicate_score_key(tmp_path: Path) -> None:
    inference = tmp_path / "inference.jsonl"
    scores = tmp_path / "scores.jsonl"
    write_jsonl(inference, [{"scene_id": "a", "method": "m", "level": "a_000", "alpha": 0, "output_path": "/x"}])
    row = {"scene_id": "a", "method": "m", "level": "a_000", "score": .7}
    write_jsonl(scores, [row, row])
    with pytest.raises(ValueError, match="Duplicate IQA score"):
        merge_candidate_scores(inference, scores, tmp_path / "out.jsonl", method="m")
