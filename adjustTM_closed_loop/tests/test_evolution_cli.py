from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from adjustTM_closed_loop.evolution.cli import main
from adjustTM_closed_loop.evolution.teacher_manifest import load_teacher_manifest


def write_image(path: Path, value: float, dtype=np.uint16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = np.iinfo(dtype).max
    image = np.full((16, 20, 3), round(value * maximum), dtype=dtype)
    assert cv2.imwrite(str(path), image)


def build_candidate_file(tmp_path: Path) -> tuple[Path, Path]:
    inputs = tmp_path / "inputs"
    candidates = tmp_path / "candidates"
    rows = []
    for index, value in enumerate((0.1, 0.2, 0.6, 0.8, 0.3, 0.5)):
        scene = f"scene_{index}.png"
        write_image(inputs / scene, value)
        baseline = candidates / "a_000" / scene
        positive = candidates / "a_p025" / scene
        write_image(baseline, min(value + 0.1, 0.9))
        write_image(positive, min(value + 0.2, 0.95))
        rows.extend([
            {"scene_id": scene, "level": "a_000", "alpha": 0, "output_path": str(baseline),
             "overall_score": 0.6, "confidence": 1, "action": "KEEP", "distribution_status": "in_domain"},
            {"scene_id": scene, "level": "a_p025", "alpha": 0.25, "output_path": str(positive),
             "overall_score": 0.8 if index < 3 else 0.61, "confidence": 1, "action": "KEEP",
             "distribution_status": "in_domain"},
        ])
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return inputs, path


def test_cli_select_and_fit_policy(tmp_path: Path) -> None:
    inputs, scores = build_candidate_file(tmp_path)
    output = tmp_path / "run"
    assert main([
        "select-teachers", "--scores", str(scores), "--input-dir", str(inputs),
        "--output-dir", str(output), "--validation-fraction", "0", "--test-fraction", "0",
    ]) == 0
    manifest = output / "teacher_manifest.jsonl"
    records = load_teacher_manifest(manifest)
    assert len(records) == 6
    assert sum(item.status == "improved" for item in records) == 3
    assert (output / "selection_summary.json").is_file()

    policy = output / "alpha_policy.json"
    report = output / "policy_report.json"
    assert main([
        "fit-policy", "--teacher-manifest", str(manifest), "--output-policy", str(policy),
        "--output-report", str(report), "--ood-threshold", "1000000",
    ]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["fit_sample_count"] == 6
    assert policy.is_file()


def test_cli_adjudicate_writes_reject_report(tmp_path: Path) -> None:
    inputs, scores = build_candidate_file(tmp_path)
    output = tmp_path / "run"
    main([
        "select-teachers", "--scores", str(scores), "--input-dir", str(inputs),
        "--output-dir", str(output), "--validation-fraction", "0", "--test-fraction", "0",
    ])
    records = load_teacher_manifest(output / "teacher_manifest.jsonl")
    baseline_path, student_path = tmp_path / "baseline.jsonl", tmp_path / "student.jsonl"
    baseline_rows, student_rows = [], []
    for item in records:
        baseline_rows.append({"scene_id": item.scene_id, "score": 0.7})
        student_rows.append({"scene_id": item.scene_id, "score": 0.5 if item.status != "improved" else 0.9})
    baseline_path.write_text("\n".join(json.dumps(row) for row in baseline_rows), encoding="utf-8")
    student_path.write_text("\n".join(json.dumps(row) for row in student_rows), encoding="utf-8")
    report_path = tmp_path / "adjudication.json"
    assert main([
        "adjudicate", "--teacher-manifest", str(output / "teacher_manifest.jsonl"),
        "--baseline-scores", str(baseline_path), "--student-scores", str(student_path),
        "--output", str(report_path), "--evaluation-splits", "train", "--allow-self-evaluation",
    ]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["decision"] == "REJECT"
