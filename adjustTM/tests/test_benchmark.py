from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from adjustTM.constants import LEVELS


def _write_rgb(path: Path, value: int, dtype=np.uint16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10, 3), value, dtype=dtype)
    assert cv2.imwrite(str(path), image[..., ::-1])


def test_semantic_group_separates_real_flux_and_retinex() -> None:
    from adjustTM.benchmark.schemas import semantic_group

    assert semantic_group("a_000") == "real_camera"
    assert semantic_group("a_m100") == "flux_endpoint"
    assert semantic_group("a_p100") == "flux_endpoint"
    assert semantic_group("a_m050") == "retinex_intermediate"


def test_build_manifest_validates_and_hashes_complete_dataset(tmp_path: Path) -> None:
    from adjustTM.benchmark.schemas import build_manifest, canonical_hash

    input_dir = tmp_path / "input"
    gt_root = tmp_path / "gt"
    _write_rgb(input_dir / "scene.png", 1234)
    for index, (level, _) in enumerate(LEVELS):
        _write_rgb(gt_root / level / "scene.png", 1000 + index)

    manifest = build_manifest(input_dir, gt_root)
    assert manifest["scene_count"] == 1
    assert manifest["scenes"][0]["input"]["dtype"] == "uint16"
    assert set(manifest["scenes"][0]["gt"]) == {name for name, _ in LEVELS}
    assert canonical_hash(manifest) == canonical_hash(json.loads(json.dumps(manifest)))


def test_build_manifest_fails_on_missing_level(tmp_path: Path) -> None:
    from adjustTM.benchmark.schemas import build_manifest

    input_dir = tmp_path / "input"
    gt_root = tmp_path / "gt"
    _write_rgb(input_dir / "scene.png", 1234)
    for level, _ in LEVELS[:-1]:
        _write_rgb(gt_root / level / "scene.png", 1000)
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        build_manifest(input_dir, gt_root)


def test_exposure_transform_identity_and_ev_gain() -> None:
    from adjustTM.benchmark.transforms import exposure_transform

    image = torch.full((1, 3, 4, 4), 0.25)
    assert torch.equal(exposure_transform(image, 0.0), image)
    brighter = exposure_transform(image, 1.0)
    assert torch.all(brighter > image)


def test_luminance_gamma_preserves_linear_chromaticity() -> None:
    from adjustTM.benchmark.transforms import luminance_gamma_transform
    from adjustTM.transfer import srgb_to_linear

    image = torch.tensor([[[[0.2]], [[0.4]], [[0.6]]]], dtype=torch.float32)
    output = luminance_gamma_transform(image, 0.6)
    before = srgb_to_linear(image)
    after = srgb_to_linear(output)
    before_rg = before[:, :2] / before.sum(dim=1, keepdim=True)
    after_rg = after[:, :2] / after.sum(dim=1, keepdim=True)
    assert torch.allclose(before_rg, after_rg, atol=1e-5)


def test_project_global_parameters_enforces_anchor_and_monotonicity() -> None:
    from adjustTM.benchmark.baselines import project_level_parameters

    exposure = {name: value for (name, _), value in zip(LEVELS, [-1.0, -0.5, -0.7, -0.1, 0.2, 0.8, 0.4, 1.2, 1.0])}
    projected = project_level_parameters(exposure, kind="exposure")
    seq = [projected[name] for name, _ in LEVELS]
    assert seq[4] == 0.0
    assert seq == sorted(seq)

    gamma = {name: value for (name, _), value in zip(LEVELS, [2.0, 1.4, 1.6, 1.1, 0.8, 0.7, 0.9, 0.5, 0.6])}
    projected_g = project_level_parameters(gamma, kind="gamma")
    seq_g = [projected_g[name] for name, _ in LEVELS]
    assert seq_g[4] == 1.0
    assert seq_g == sorted(seq_g, reverse=True)


def test_search_parameter_recovers_synthetic_exposure() -> None:
    from adjustTM.benchmark.baselines import search_best_parameter
    from adjustTM.benchmark.transforms import exposure_transform

    base = torch.full((1, 3, 8, 8), 0.2)
    target = exposure_transform(base, 0.75)
    result = search_best_parameter(base, target, kind="exposure", minimum=-2, maximum=2, steps=81)
    assert result.parameter == pytest.approx(0.75, abs=0.06)


def test_control_curve_metrics_detect_violation_dead_zone_and_jump() -> None:
    from adjustTM.benchmark.metrics import control_curve_metrics

    alphas = np.linspace(-1, 1, 7)
    luminance = np.array([-2.0, -1.5, -1.5, -1.0, -1.1, -0.2, 0.0])
    metrics = control_curve_metrics(alphas, luminance, dead_zone_threshold=1e-6, jump_ratio_threshold=2.0)
    assert metrics["violation_rate"] > 0
    assert metrics["dead_zone_rate"] > 0
    assert metrics["jump_rate"] > 0
    assert metrics["strict_scene_pass"] == 0.0


def test_scene_first_aggregation_does_not_treat_levels_as_independent() -> None:
    from adjustTM.benchmark.statistics import aggregate_scene_level_records

    records = [
        {"scene_id": "a", "value": 0.0},
        {"scene_id": "a", "value": 2.0},
        {"scene_id": "b", "value": 10.0},
    ]
    scene_values = aggregate_scene_level_records(records, value_key="value")
    assert scene_values == {"a": 1.0, "b": 10.0}


def test_paired_statistics_and_holm_are_deterministic() -> None:
    from adjustTM.benchmark.statistics import paired_comparison, holm_adjust

    first = {f"s{i}": float(i) for i in range(10)}
    second = {f"s{i}": float(i) + 1.0 for i in range(10)}
    result = paired_comparison(first, second, lower_is_better=True, seed=7, bootstrap_samples=300, permutation_samples=500)
    assert result["win_rate"] == 1.0
    assert result["mean_delta"] == pytest.approx(-1.0)
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_vlm_prompts_are_separate_and_scores_validate() -> None:
    from adjustTM.benchmark.vlm import build_prompt, aggregate_repeats

    intent = build_prompt("intent", alpha=0.5)
    natural = build_prompt("naturalness", alpha=0.5)
    assert "目标" in intent["system"]
    assert "不要评价它是否接近" in natural["system"]
    records = [
        {"global_brightness_match": 4, "confidence": 0.8},
        {"global_brightness_match": 5, "confidence": 0.9},
        {"global_brightness_match": 3, "confidence": 0.7},
    ]
    aggregate = aggregate_repeats(records, required_scores=["global_brightness_match"])
    assert aggregate["scores"]["global_brightness_match"]["median"] == 4.0


def test_human_trials_are_deterministically_blinded_and_balanced() -> None:
    from adjustTM.benchmark.human_study import build_balanced_trials

    methods = ["a", "b", "c", "d", "e"]
    trials_1, mapping_1 = build_balanced_trials(
        scene_ids=["s1", "s2"], levels=["a_m100", "a_p100"], methods=methods,
        candidates_per_trial=3, blocks_per_scene_level=2, seed=42, study_type="intent_match"
    )
    trials_2, mapping_2 = build_balanced_trials(
        scene_ids=["s1", "s2"], levels=["a_m100", "a_p100"], methods=methods,
        candidates_per_trial=3, blocks_per_scene_level=2, seed=42, study_type="intent_match"
    )
    assert trials_1 == trials_2
    assert mapping_1 == mapping_2
    counts = {method: 0 for method in methods}
    for trial in trials_1:
        for candidate in trial["candidates"]:
            counts[mapping_1[trial["trial_id"]][candidate["candidate_id"]]] += 1
    assert max(counts.values()) - min(counts.values()) <= 2


def test_fit_pad_round_trip_geometry_preserves_aspect_ratio() -> None:
    from adjustTM.benchmark.image_io import fit_pad_tensor, unpad_tensor

    image = torch.rand(1, 3, 20, 40)
    padded, geometry = fit_pad_tensor(image, max_side=32, multiple=8)
    assert padded.shape[-2:] == (16, 32)
    restored = unpad_tensor(padded, geometry)
    assert restored.shape[-2:] == (16, 32)
    assert geometry["scale"] == pytest.approx(0.8)


def test_generate_outputs_uses_cache_identity_and_dense_records(tmp_path: Path) -> None:
    from adjustTM.benchmark.generate_outputs import generate_cached_outputs

    input_path = tmp_path / "input.png"
    _write_rgb(input_path, 1000)
    manifest = {
        "content_sha256": "manifest-hash",
        "scenes": [{"scene_id": "input.png", "input": {"path": str(input_path)}}],
    }

    class Runner:
        def predict(self, image: torch.Tensor, alpha: float):
            return {"output": torch.full_like(image, (alpha + 1.0) / 2.0)}

        def metadata(self):
            return {"kind": "synthetic"}

    result = generate_cached_outputs(
        manifest=manifest,
        runners={"toy": Runner()},
        output_root=tmp_path / "out",
        protocol_hash="protocol-hash",
        levels=[("a_m100", -1.0), ("a_000", 0.0), ("a_p100", 1.0)],
        dense_alphas=[-1.0, 0.0, 1.0],
        max_side=None,
        device="cpu",
    )
    assert len(result["inference_records"]) == 3
    assert len(result["dense_records"]) == 3
    cache = json.loads((tmp_path / "out" / "cache_identity.json").read_text())
    assert cache["protocol_hash"] == "protocol-hash"
    with pytest.raises(RuntimeError):
        generate_cached_outputs(
            manifest=manifest, runners={"toy": Runner()}, output_root=tmp_path / "out",
            protocol_hash="different", levels=[("a_000", 0.0)], dense_alphas=[], max_side=None, device="cpu"
        )


def test_reference_metric_records_keep_semantic_groups(tmp_path: Path) -> None:
    from adjustTM.benchmark.evaluate_reference import evaluate_cached_outputs

    gt_root = tmp_path / "gt"
    output_root = tmp_path / "outputs"
    scene = "s.png"
    manifest_scene = {"scene_id": scene, "gt": {}}
    for level, _ in LEVELS:
        path = gt_root / level / scene
        _write_rgb(path, 20000)
        manifest_scene["gt"][level] = {"path": str(path)}
        _write_rgb(output_root / "method" / level / scene, 20000)
    records = evaluate_cached_outputs(
        manifest={"scenes": [manifest_scene]}, output_root=output_root, methods=["method"],
        levels=[name for name, _ in LEVELS], device="cpu", lpips_model=None
    )
    groups = {record["semantic_group"] for record in records}
    assert groups == {"real_camera", "flux_endpoint", "retinex_intermediate"}
    assert all(record["metrics"]["rgb_psnr"] > 100 for record in records)


def test_bradley_terry_orders_consistent_winner() -> None:
    from adjustTM.benchmark.human_study import bradley_terry_scores

    scores = bradley_terry_scores([("a", "b"), ("a", "b"), ("a", "c"), ("b", "c")])
    assert scores["a"] > scores["b"] > scores["c"]


def test_report_builder_keeps_main_and_oracle_tables_separate(tmp_path: Path) -> None:
    from adjustTM.benchmark.report import build_report

    summary = {
        "main_methods": {"film": {"control": {"strict_scene_pass": 0.9}}},
        "diagnostic_methods": {"gamma_oracle": {"reference": {"log_luma_mae": 0.01}}},
        "protocol": {"scene_count": 10},
    }
    outputs = build_report(summary, tmp_path)
    html = Path(outputs["html"]).read_text(encoding="utf-8")
    assert "Main methods" in html
    assert "Diagnostic oracle methods" in html
    assert "gamma_oracle" in html


def test_cli_modules_support_help() -> None:
    import subprocess
    import sys

    modules = [
        "build_manifest", "calibrate_baselines", "generate_outputs", "generate_oracles",
        "evaluate_reference", "evaluate_control", "evaluate_vlm", "build_human_study",
        "analyze_human_study", "compare_methods", "report", "run",
    ]
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", f"adjustTM.benchmark.{module}", "--help"],
            check=False, capture_output=True, text=True,
        )
        assert completed.returncode == 0, (module, completed.stderr)


def test_global_simple_runner_interpolates_parameters_for_dense_alpha() -> None:
    from adjustTM.benchmark.methods import SimpleTransformRunner

    class Baseline:
        def predict(self, image, alpha):
            return {"output": image}
        def metadata(self):
            return {"type": "baseline"}

    parameters = {name: alpha for name, alpha in LEVELS}
    runner = SimpleTransformRunner(Baseline(), parameters, "exposure")
    assert runner.parameter_for_alpha(0.125) == pytest.approx(0.125)
    assert runner.parameter_for_alpha(-0.875) == pytest.approx(-0.875)


def test_compare_summary_bootstraps_scene_aggregates_not_raw_levels() -> None:
    from adjustTM.benchmark.compare_methods import summarize

    reference = [
        {"scene_id": "a", "method": "m", "semantic_group": "flux_endpoint", "metrics": {"x": 0.0}},
        {"scene_id": "a", "method": "m", "semantic_group": "flux_endpoint", "metrics": {"x": 2.0}},
        {"scene_id": "b", "method": "m", "semantic_group": "flux_endpoint", "metrics": {"x": 10.0}},
    ]
    result = summarize(reference, [], ["m"], [])
    metric = result["main_methods"]["m"]["reference"]["flux_endpoint.x"]
    assert metric["mean"] == pytest.approx(5.5)
    assert metric["scene_count"] == 2


def test_generate_outputs_records_zero_drift_and_dense_chroma(tmp_path: Path) -> None:
    from adjustTM.benchmark.generate_outputs import generate_cached_outputs

    input_path = tmp_path / "input.png"
    _write_rgb(input_path, 5000)
    manifest = {"content_sha256": "m", "scenes": [{"scene_id": "input.png", "input": {"path": str(input_path)}}]}

    class Runner:
        def predict(self, image, alpha):
            return {"output": image.clamp(0, 1) * (1 + 0.1 * alpha)}
        def zero_reference(self, image):
            return image.clamp(0, 1)
        def metadata(self):
            return {"kind": "controlled"}

    result = generate_cached_outputs(
        manifest=manifest, runners={"m": Runner()}, output_root=tmp_path / "out", protocol_hash="p",
        levels=[("a_000", 0.0)], dense_alphas=[-1.0, 0.0, 1.0], max_side=None, device="cpu"
    )
    assert result["inference_records"][0]["alpha_zero_max_drift"] == 0.0
    assert all("chroma_rg_drift_from_zero" in row for row in result["dense_records"])


def test_streaming_global_calibration_recovers_exposure_without_loading_dataset(tmp_path: Path) -> None:
    from adjustTM.benchmark.calibrate_baselines import calibrate_from_manifest
    from adjustTM.benchmark.image_io import write_srgb_png16
    from adjustTM.benchmark.transforms import exposure_transform

    baseline_root = tmp_path / "baseline"
    scenes = []
    for index in range(2):
        scene_id = f"s{index}.png"
        base = torch.full((1, 3, 8, 8), 0.2 + 0.02 * index)
        write_srgb_png16(baseline_root / "a_000" / scene_id, base)
        gt = {}
        for level, alpha in LEVELS:
            target = exposure_transform(base, alpha * 0.5)
            path = tmp_path / "gt" / level / scene_id
            write_srgb_png16(path, target)
            gt[level] = {"path": str(path)}
        scenes.append({"scene_id": scene_id, "gt": gt})
    manifest = {"scenes": scenes}
    result = calibrate_from_manifest(
        manifest, baseline_root, kind="exposure", minimum=-1.0, maximum=1.0, steps=41, device="cpu"
    )
    assert result["a_p100"] == pytest.approx(0.5, abs=0.06)
    assert result["a_m100"] == pytest.approx(-0.5, abs=0.06)


def test_trajectory_records_use_all_nine_levels_per_scene() -> None:
    from adjustTM.benchmark.evaluate_reference import build_trajectory_records

    records = []
    for method in ["m"]:
        for scene in ["s"]:
            for level, alpha in LEVELS:
                records.append({
                    "method": method, "scene_id": scene, "level": level,
                    "metrics": {"pred_mean_log_luma": alpha, "target_mean_log_luma": alpha}
                })
    trajectories = build_trajectory_records(records)
    assert len(trajectories) == 1
    assert trajectories[0]["metrics"]["curve_mae"] == 0.0
    assert trajectories[0]["metrics"]["spearman"] == pytest.approx(1.0)


def test_vlm_validation_rejects_out_of_range_and_flags_variance() -> None:
    from adjustTM.benchmark.vlm import aggregate_repeats, validate_response

    with pytest.raises(ValueError):
        validate_response({"x": 6}, required_scores=["x"])
    aggregate = aggregate_repeats([{"x": 1}, {"x": 5}, {"x": 1}], required_scores=["x"], unstable_std_threshold=1.0)
    assert aggregate["unstable_judgment"] is True


def test_compare_summary_contains_paired_deltas_against_baselines() -> None:
    from adjustTM.benchmark.compare_methods import summarize

    reference = []
    for scene, base, film in [("s1", 2.0, 1.0), ("s2", 4.0, 2.0)]:
        reference.append({"scene_id": scene, "method": "frozen_baseline", "semantic_group": "flux_endpoint", "metrics": {"log_luma_mae": base}})
        reference.append({"scene_id": scene, "method": "film", "semantic_group": "flux_endpoint", "metrics": {"log_luma_mae": film}})
    result = summarize(reference, [], ["frozen_baseline", "film"], [], comparison_baselines=["frozen_baseline"])
    comparison = result["comparisons"]["film"]["vs_frozen_baseline"]["reference.flux_endpoint.log_luma_mae"]
    assert comparison["mean_delta"] == pytest.approx(-1.5)
    assert comparison["win_rate"] == 1.0


def test_human_study_html_is_blinded_and_collects_best_worst() -> None:
    from adjustTM.benchmark.human_study import render_study_html

    trials = [{
        "trial_id": "t1", "study_type": "intent_match", "scene_id": "s", "level": "a_p100",
        "center_image": "center.png", "target_image": "target.png",
        "candidates": [
            {"candidate_id": "C1", "blind_asset_id": "x", "image": "one.png"},
            {"candidate_id": "C2", "blind_asset_id": "y", "image": "two.png"},
        ],
    }]
    html = render_study_html(trials, title="Blind Study")
    assert "best_candidate_id" in html
    assert "worst_candidate_id" in html
    assert "one.png" in html
    assert "film" not in html


def test_human_qc_parses_csv_boolean_strings() -> None:
    from adjustTM.benchmark.human_study import response_quality_control

    qc = response_quality_control([
        {"rater_id": "r", "attention_correct": "false", "repeat_consistent": "true", "duration_seconds": "2"},
        {"rater_id": "r", "attention_correct": "false", "repeat_consistent": "true", "duration_seconds": "2"},
    ])
    assert qc["r"]["attention_rate"] == 0.0
    assert qc["r"]["included"] is False


def test_reference_subset_does_not_require_nine_level_trajectory(tmp_path: Path) -> None:
    from adjustTM.benchmark.evaluate_reference import build_trajectory_records

    records = [{
        "method": "m", "scene_id": "s", "level": "a_000",
        "metrics": {"pred_mean_log_luma": 0.0, "target_mean_log_luma": 0.0},
    }]
    with pytest.raises(RuntimeError):
        build_trajectory_records(records)


def test_materialize_blinded_assets_removes_method_names(tmp_path: Path) -> None:
    from adjustTM.benchmark.human_study import materialize_blinded_assets

    source = tmp_path / "outputs" / "film" / "a_p100" / "s.png"
    _write_rgb(source, 1000)
    trials = [{
        "trial_id": "t", "scene_id": "s.png", "level": "a_p100",
        "center_image": str(source),
        "candidates": [{"candidate_id": "C1", "blind_asset_id": "abc123", "image": str(source)}],
    }]
    blinded = materialize_blinded_assets(trials, tmp_path / "study", mode="copy")
    candidate_path = blinded[0]["candidates"][0]["image"]
    assert "film" not in candidate_path
    assert Path(tmp_path / "study" / candidate_path).is_file()


def test_select_stratified_scenes_covers_available_tags() -> None:
    from adjustTM.benchmark.human_study import select_stratified_scenes

    scenes = [
        {"scene_id": "a", "tags": ["portrait"]},
        {"scene_id": "b", "tags": ["hdr"]},
        {"scene_id": "c", "tags": ["portrait"]},
        {"scene_id": "d", "tags": ["low_light"]},
    ]
    selected = select_stratified_scenes(scenes, count=3, seed=4)
    assert len(selected) == 3
    selected_tags = {tag for scene in scenes if scene["scene_id"] in selected for tag in scene["tags"]}
    assert len(selected_tags) >= 3
