import csv
import json

import cv2
import numpy as np

from photofinishing.evaluate.reporting import (
    build_increment_rows,
    build_metric_rows,
    render_comparison_panel,
    write_reports,
)


def test_reporting_writes_machine_and_human_readable_outputs(tmp_path):
    model_metrics = {
        "pretrained": {"signed_ev_error": -0.5, "absolute_ev_error": 0.5, "cbcr_swd": 0.2},
        "stage1": {"signed_ev_error": -0.2, "absolute_ev_error": 0.2, "cbcr_swd": 0.18},
        "stage2/affine": {"signed_ev_error": -0.21, "absolute_ev_error": 0.21, "cbcr_swd": 0.1},
    }
    metric_rows = build_metric_rows("seed0", "scene1", ("night",), model_metrics)
    increment_rows = build_increment_rows(
        "seed0", "scene1", ("night",),
        model_metrics["pretrained"], model_metrics["stage1"],
        {"affine": model_metrics["stage2/affine"]},
    )
    decisions = {
        "cross_group": {
            "stage1": {"status": "effective", "reason": "brightness improved"},
            "stage2_necessity": {"status": "needed", "reason": "above noise"},
            "stage2": {
                "affine": {
                    "status": "effective",
                    "reason": "color improved",
                    "primary": {"median": 0.03, "win_rate": 0.8, "ci95_low": 0.01, "ci95_high": 0.05},
                }
            },
            "recommended_stage2": {"recommended": "affine", "reason": "lower risk"},
        },
        "data_expansion": {
            "expand_evaluation_data": False,
            "evaluation_reasons": [],
            "expand_training_data": False,
            "training_reasons": ["not required"],
            "target_slices": [],
        },
    }
    write_reports(
        tmp_path,
        metric_rows=metric_rows,
        increment_rows=increment_rows,
        summary={"groups": {"seed0": {}}},
        decisions=decisions,
    )
    for name in [
        "per_sample_metrics.csv", "incremental_improvements.csv", "summary.json",
        "summary.csv", "decisions.json", "report.md",
    ]:
        assert (tmp_path / name).is_file()
    with (tmp_path / "incremental_improvements.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stage1_row = next(row for row in rows if row["transition"] == "pretrained_to_stage1")
    stage2_row = next(row for row in rows if row["transition"] == "stage1_to_affine")
    assert float(stage1_row["absolute_ev_error"]) == 0.3
    assert float(stage2_row["cbcr_swd"]) == 0.08
    assert "signed_ev_error" not in stage1_row
    assert json.loads((tmp_path / "decisions.json").read_text())["cross_group"]["stage1"]["status"] == "effective"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Stage 1" in report and "affine" in report


def test_comparison_panel_contains_all_named_images(tmp_path):
    images = {
        "Input": np.zeros((16, 16, 3), np.float32),
        "Pretrained": np.full((16, 16, 3), 0.25, np.float32),
        "Stage1": np.full((16, 16, 3), 0.5, np.float32),
        "Stage2/affine": np.full((16, 16, 3), 0.75, np.float32),
        "Reference": np.ones((16, 16, 3), np.float32),
    }
    output = tmp_path / "panel.jpg"
    render_comparison_panel(images, output, tile_size=64)
    panel = cv2.imread(str(output))
    assert panel is not None
    assert panel.shape[1] == 64 * len(images)
    assert panel.shape[0] > 64
