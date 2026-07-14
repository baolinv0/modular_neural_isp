from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _write_image(
    path: Path,
    value: int,
    *,
    patch: tuple[slice, slice] | None = None,
    patch_value: int | None = None,
    shape: tuple[int, int] = (40, 60),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((shape[0], shape[1], 3), value, dtype=np.uint16)
    if patch is not None and patch_value is not None:
        image[patch[0], patch[1]] = patch_value
    assert cv2.imwrite(str(path), image)


def test_select_case_sets_separates_best_representative_and_failure() -> None:
    from adjustTM.benchmark.case_gallery import select_case_sets

    reference = []
    for scene, baseline, focus in [
        ("best", 4.0, 1.0),
        ("representative", 2.0, 2.0),
        ("failure", 1.0, 5.0),
        ("other", 3.0, 2.5),
    ]:
        for method, value in [("gamma_global", baseline), ("film", focus)]:
            reference.append(
                {
                    "scene_id": scene,
                    "method": method,
                    "semantic_group": "flux_endpoint",
                    "level": "a_p100",
                    "metrics": {"log_luma_mae": value},
                }
            )
    control = [
        {
            "scene_id": "failure",
            "method": "film",
            "metrics": {"violation_magnitude": 3.0, "jump_rate": 1.0},
        },
        {
            "scene_id": "best",
            "method": "film",
            "metrics": {"violation_magnitude": 0.0, "jump_rate": 0.0},
        },
    ]
    selected = select_case_sets(
        reference,
        control,
        focus_method="film",
        comparison_baseline="gamma_global",
        representative_count=1,
        best_count=1,
        failure_count=1,
    )
    assert selected["best_improvements"] == ["best"]
    assert selected["failure_cases"] == ["failure"]
    assert selected["representative_cases"] == ["representative"]


def test_automatic_crop_boxes_are_inside_image_and_cover_extremes(
    tmp_path: Path,
) -> None:
    from adjustTM.benchmark.case_gallery import automatic_crop_boxes

    path = tmp_path / "scene.png"
    _write_image(
        path,
        20000,
        patch=(slice(0, 10), slice(0, 10)),
        patch_value=65000,
    )
    image = cv2.cvtColor(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
    )
    boxes = automatic_crop_boxes(image, crop_fraction=0.25)
    assert set(boxes) == {"highlight", "shadow", "texture"}
    height, width = image.shape[:2]
    for x, y, crop_width, crop_height in boxes.values():
        assert 0 <= x < width and 0 <= y < height
        assert x + crop_width <= width and y + crop_height <= height
    highlight_x, highlight_y, _, _ = boxes["highlight"]
    assert highlight_x < 20 and highlight_y < 20


def test_build_case_gallery_creates_pages_browser_assets_and_curves(
    tmp_path: Path,
) -> None:
    from adjustTM.benchmark.case_gallery import build_case_gallery

    scene_id = "scene.png"
    gt = {}
    for level, value in [
        ("a_m100", 10000),
        ("a_000", 30000),
        ("a_p100", 50000),
    ]:
        path = tmp_path / "gt" / level / scene_id
        _write_image(path, value)
        gt[level] = {"path": str(path)}
    manifest = {
        "scenes": [{"scene_id": scene_id, "gt": gt, "tags": ["hdr"]}]
    }
    output_root = tmp_path / "outputs"
    methods = ["gamma_global", "film"]
    levels = ["a_m100", "a_000", "a_p100"]
    alpha_map = {"a_m100": -1.0, "a_000": 0.0, "a_p100": 1.0}
    for method_index, method in enumerate(methods):
        for level_index, level in enumerate(levels):
            _write_image(
                output_root / method / level / scene_id,
                12000 + method_index * 4000 + level_index * 12000,
            )
    reference = []
    for method in methods:
        for level in levels:
            reference.append(
                {
                    "scene_id": scene_id,
                    "method": method,
                    "level": level,
                    "semantic_group": (
                        "real_camera" if level == "a_000" else "flux_endpoint"
                    ),
                    "alpha": alpha_map[level],
                    "metrics": {
                        "log_luma_mae": 0.1 if method == "film" else 0.3,
                        "rgb_ssim": 0.9,
                        "target_mean_log_luma": alpha_map[level],
                    },
                }
            )
    dense = []
    for method_index, method in enumerate(methods):
        for alpha in [-1.0, 0.0, 1.0]:
            dense.append(
                {
                    "scene_id": scene_id,
                    "method": method,
                    "alpha": alpha,
                    "mean_log_luma": alpha + method_index * 0.1,
                    "clip_ratio": max(alpha, 0) * 0.05,
                    "deep_shadow_ratio": max(-alpha, 0) * 0.05,
                    "chroma_rg_drift_from_zero": abs(alpha) * 0.01,
                }
            )
    outputs = build_case_gallery(
        manifest=manifest,
        output_root=output_root,
        reference_records=reference,
        dense_records=dense,
        control_records=[],
        methods=methods,
        levels=levels,
        focus_method="film",
        comparison_baseline="gamma_global",
        output_dir=tmp_path / "gallery",
        representative_count=1,
        best_count=1,
        failure_count=1,
        asset_mode="copy",
    )
    assert Path(outputs["index"]).is_file()
    assert Path(outputs["browser"]).is_file()
    assert Path(outputs["case_index"]).is_file()
    page = Path(outputs["representative_cases"]).read_text(encoding="utf-8")
    assert "Cross-method comparison" in page
    assert "Nine-level trajectory" in page
    assert "Mean log luminance" in page
    assert "Target (nine-level)" in page
    assert "highlight" in page
    index = json.loads(Path(outputs["case_index"]).read_text(encoding="utf-8"))
    assert index["case_sets"]["representative_cases"] == [scene_id]
    assert any((tmp_path / "gallery" / "assets").rglob("*.png"))


def test_report_builder_links_visual_evidence_gallery(tmp_path: Path) -> None:
    from adjustTM.benchmark.report import build_report

    gallery = tmp_path / "qualitative" / "index.html"
    gallery.parent.mkdir(parents=True)
    gallery.write_text("gallery", encoding="utf-8")
    outputs = build_report(
        {"main_methods": {}, "diagnostic_methods": {}, "protocol": {}},
        tmp_path / "report",
        visual_gallery=gallery,
    )
    report = Path(outputs["html"]).read_text(encoding="utf-8")
    assert "Visual evidence" in report
    assert "../qualitative/index.html" in report


def test_visual_gallery_cli_supports_help() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "adjustTM.benchmark.build_case_gallery", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--focus-method" in completed.stdout


def test_gallery_scales_crop_coordinates_between_gt_and_resized_outputs(
    tmp_path: Path,
) -> None:
    from adjustTM.benchmark.case_gallery import materialize_gallery_assets

    scene_id = "scene.png"
    gt = {}
    for level in ["a_m100", "a_000", "a_p100"]:
        path = tmp_path / "gt" / level / scene_id
        _write_image(
            path,
            10000,
            patch=(slice(0, 10), slice(0, 15)),
            patch_value=65000,
            shape=(40, 60),
        )
        gt[level] = {"path": str(path)}
        _write_image(
            tmp_path / "outputs" / "film" / level / scene_id,
            10000,
            patch=(slice(0, 5), slice(0, 8)),
            patch_value=65000,
            shape=(20, 30),
        )
    index = materialize_gallery_assets(
        manifest={"scenes": [{"scene_id": scene_id, "gt": gt}]},
        output_root=tmp_path / "outputs",
        methods=["film"],
        levels=["a_m100", "a_000", "a_p100"],
        output_dir=tmp_path / "gallery",
        selected_scenes=[scene_id],
        crop_fraction=0.25,
        asset_mode="copy",
    )
    highlight = index["scenes"][scene_id]["crops"]["highlight"]
    target_crop = cv2.imread(
        str(tmp_path / "gallery" / highlight["target"]["a_000"]),
        cv2.IMREAD_UNCHANGED,
    )
    method_crop = cv2.imread(
        str(
            tmp_path
            / "gallery"
            / highlight["methods"]["film"]["a_000"]
        ),
        cv2.IMREAD_UNCHANGED,
    )
    assert target_crop.shape[:2] == (10, 15)
    assert method_crop.shape[:2] == (5, 8)
    assert method_crop.mean() > 50000


def test_vlm_disagreement_parses_aggregate_task_ids() -> None:
    from adjustTM.benchmark.case_gallery import select_case_sets

    reference = []
    for scene, error in [("s1.png", 0.1), ("s2.png", 0.5)]:
        for method, value in [("film", error), ("gamma_global", 0.4)]:
            reference.append(
                {
                    "scene_id": scene,
                    "method": method,
                    "level": "a_p100",
                    "semantic_group": "flux_endpoint",
                    "metrics": {"log_luma_mae": value},
                }
            )
    vlm = [
        {
            "task_id": "naturalness:s1.png:a_p100:film",
            "kind": "naturalness",
            "scores": {"overall_naturalness": {"median": 1.0}},
        },
        {
            "task_id": "naturalness:s2.png:a_p100:film",
            "kind": "naturalness",
            "scores": {"overall_naturalness": {"median": 5.0}},
        },
    ]
    selected = select_case_sets(
        reference,
        [],
        focus_method="film",
        comparison_baseline="gamma_global",
        representative_count=0,
        best_count=0,
        failure_count=0,
        disagreement_count=2,
        vlm_records=vlm,
    )
    assert set(selected["metric_vlm_disagreement"]) == {"s1.png", "s2.png"}
