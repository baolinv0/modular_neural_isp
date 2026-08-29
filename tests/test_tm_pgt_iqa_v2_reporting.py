from __future__ import annotations

import json

import numpy as np
from PIL import Image


def test_batch_cli_generates_candidates_then_objective_only_report_tree(tmp_path):
    """The production handoff is usable without a live Qwen endpoint."""
    from tm_pgt_iqa.candidate_generation.generate_candidates import main as generate_main
    from tm_pgt_iqa.cli import main as evaluate_main

    source_dir, masks_dir = tmp_path / "source", tmp_path / "masks"
    candidates_dir, results_dir = tmp_path / "candidates", tmp_path / "results"
    source_dir.mkdir()
    masks_dir.mkdir()
    image = np.full((48, 48, 3), 0.55, dtype=np.float32)
    image[14:34, 14:34] = (0.32, 0.25, 0.20)
    labels = np.zeros((48, 48), dtype=np.uint8)
    labels[10:38, 10:38] = 3
    labels[14:34, 14:34] = 1
    labels[16:32, 16:32] = 2
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(source_dir / "scene.png")
    Image.fromarray(labels).save(masks_dir / "scene.png")
    # Keep the integration fixture focused on the report handoff rather than
    # score calibration / severe-guard thresholds.
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "vlm": {"enabled": False},
        "thresholds": {"certified_score": 0.0, "usable_score": 0.0},
        "guards": {"face_clip_reject": 1.0, "face_dark_reject": 1.0, "global_clip_reject": 1.0},
    }), encoding="utf-8")

    assert generate_main([
        "--input", str(source_dir), "--masks", str(masks_dir),
        "--output", str(candidates_dir), "--config", str(config),
    ]) == 0
    assert evaluate_main([
        "--source", str(source_dir), "--candidates", str(candidates_dir),
        "--masks", str(masks_dir), "--output", str(results_dir),
        "--config", str(config), "--no-vlm",
    ]) == 0

    report = json.loads((results_dir / "report.json").read_text(encoding="utf-8"))
    assert report["count"] == 1
    assert report["scenes"][0]["selected"] is not None
    assert (results_dir / "summary.csv").exists()
    assert list((results_dir / "selected").glob("scene.*"))
    assert (results_dir / "candidates" / "scene" / "face_lift_mid.json").exists()
    assert (results_dir / "semantic" / "scene" / "scene.json").exists()
    assert (results_dir / "semantic" / "scene" / "candidate_judgments.json").exists()
    assert (results_dir / "semantic" / "scene" / "pairwise.json").exists()
    assert (results_dir / "viz" / "scene_candidate_grid.jpg").exists()
    assert (results_dir / "viz" / "scene_ranking_grid.jpg").exists()
    assert (results_dir / "viz" / "scene_failure_grid.jpg").exists()


def test_validation_metrics_cover_tau_top2_and_certified_precision():
    from tm_pgt_iqa.validation import evaluate_human_annotations

    report = {"scenes": [
        {"source": "a.png", "selected": "a", "pgt_class": "CERTIFIED_PGT", "ranking": ["a", "b", "c"]},
        {"source": "b.png", "selected": "b", "pgt_class": "USABLE_PGT", "ranking": ["c", "b", "a"]},
    ]}
    annotations = {"scenes": [
        {"source": "a.png", "ranking": ["a", "b", "c"], "accepted": True},
        {"source": "b.png", "ranking": ["a", "b", "c"], "accepted": False},
    ]}
    metrics = evaluate_human_annotations(report, annotations)
    assert metrics["top2_accuracy"] == 1.0
    assert metrics["certified_precision"] == 1.0
    assert metrics["kendall_tau"] == 0.0
