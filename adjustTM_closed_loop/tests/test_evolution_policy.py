from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from adjustTM_closed_loop.evolution.features import FEATURE_NAMES, extract_image_features
from adjustTM_closed_loop.evolution.policy import RidgeAlphaPolicy, fit_policy_from_manifest
from adjustTM_closed_loop.evolution.teacher_manifest import build_teacher_manifest, load_teacher_manifest
from adjustTM_closed_loop.evolution.schemas import DistributionStatus, TeacherSelection


def write_image(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 32, 3), round(value * 65535), dtype=np.uint16)
    assert cv2.imwrite(str(path), image)


def selection(scene: str, alpha: float, *, weight: float = 1.0, status: str = "improved") -> TeacherSelection:
    return TeacherSelection(
        scene_id=scene,
        baseline_path=f"/candidates/a_000/{scene}", selected_path=f"/candidates/{alpha}/{scene}",
        baseline_score=0.6, selected_score=0.8 if alpha else 0.6,
        selected_alpha=alpha, selected_level="a_000" if alpha == 0 else f"alpha_{alpha}",
        score_delta=0.2 if alpha else 0.0, confidence=0.95, sample_weight=weight,
        distribution_status=DistributionStatus.IN_DOMAIN, status=status,
        reason="test", rejected_candidates=(), metadata={},
    )


def test_feature_extractor_is_finite_and_named(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    write_image(path, 0.25)
    features = extract_image_features(path)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_teacher_manifest_binds_input_and_selection(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    write_image(input_dir / "a.png", 0.2)
    output = tmp_path / "teacher.jsonl"
    build_teacher_manifest([selection("a.png", 0.25)], input_dir=input_dir, output_path=output)
    records = load_teacher_manifest(output)
    assert records[0].input_path == str(input_dir / "a.png")
    assert records[0].selected_alpha == 0.25
    assert records[0].split in {"train", "validation", "test"}


def test_manifest_fails_for_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.png"):
        build_teacher_manifest([selection("missing.png", 0.25)], input_dir=tmp_path, output_path=tmp_path / "x.jsonl")


def test_ridge_policy_learns_scene_conditioned_alpha(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    selections = []
    for index, value in enumerate(np.linspace(0.08, 0.8, 20)):
        scene = f"scene_{index:02d}.png"
        write_image(input_dir / scene, float(value))
        alpha = float(np.clip(0.5 - value, -0.5, 0.5))
        selections.append(selection(scene, alpha))
    manifest = tmp_path / "teacher.jsonl"
    build_teacher_manifest(selections, input_dir=input_dir, output_path=manifest, split_seed=7)
    model, report = fit_policy_from_manifest(manifest, ridge=1e-3, ood_threshold=20.0)
    assert report["sample_count"] == 20
    dark_prediction = model.predict_path(input_dir / "scene_00.png")
    bright_prediction = model.predict_path(input_dir / "scene_19.png")
    assert dark_prediction.alpha > bright_prediction.alpha
    assert dark_prediction.in_domain


def test_policy_fails_closed_on_ood_feature() -> None:
    model = RidgeAlphaPolicy(
        feature_names=FEATURE_NAMES,
        feature_mean=np.zeros(len(FEATURE_NAMES)), feature_scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.ones(len(FEATURE_NAMES)) * 0.1, intercept=0.2,
        alpha_min=-1.0, alpha_max=1.0, ood_threshold=3.0,
    )
    prediction = model.predict_features(np.ones(len(FEATURE_NAMES)) * 100)
    assert prediction.alpha == 0.0
    assert not prediction.in_domain
    assert prediction.reason == "ood_fallback"


def test_policy_round_trip_json(tmp_path: Path) -> None:
    model = RidgeAlphaPolicy(
        feature_names=FEATURE_NAMES,
        feature_mean=np.zeros(len(FEATURE_NAMES)), feature_scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.arange(len(FEATURE_NAMES)) * 0.001, intercept=0.1,
        alpha_min=-1.0, alpha_max=1.0, ood_threshold=10.0,
    )
    path = tmp_path / "policy.json"
    model.save(path)
    loaded = RidgeAlphaPolicy.load(path)
    x = np.linspace(0, 1, len(FEATURE_NAMES))
    assert loaded.predict_features(x).alpha == pytest.approx(model.predict_features(x).alpha)


def test_policy_rejects_feature_contract_mismatch() -> None:
    with pytest.raises(ValueError, match="feature_names"):
        RidgeAlphaPolicy(
            feature_names=["wrong"] * len(FEATURE_NAMES),
            feature_mean=np.zeros(len(FEATURE_NAMES)), feature_scale=np.ones(len(FEATURE_NAMES)),
            coefficients=np.zeros(len(FEATURE_NAMES)), intercept=0.0,
        )


def test_teacher_manifest_stratifies_improved_and_anchor_scenes(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    selections = []
    for index in range(40):
        scene = f"stratified_{index:02d}.png"
        write_image(input_dir / scene, 0.1 + index * 0.01)
        is_improved = index % 2 == 0
        selections.append(selection(
            scene,
            0.25 if is_improved else 0.0,
            status="improved" if is_improved else "baseline_anchor",
        ))
    manifest = tmp_path / "teacher.jsonl"
    build_teacher_manifest(
        selections, input_dir=input_dir, output_path=manifest,
        split_seed=42, validation_fraction=0.2, test_fraction=0.2,
    )
    records = load_teacher_manifest(manifest)
    for split in ("validation", "test"):
        statuses = {record.status for record in records if record.split == split}
        assert statuses == {"improved", "baseline_anchor"}
