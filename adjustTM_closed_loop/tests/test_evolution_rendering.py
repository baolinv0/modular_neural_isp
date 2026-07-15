from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from adjustTM_closed_loop.evolution.features import FEATURE_NAMES
from adjustTM_closed_loop.evolution.policy import RidgeAlphaPolicy
from adjustTM_closed_loop.evolution.rendering import render_policy_with_runner
from adjustTM_closed_loop.evolution.schemas import DistributionStatus, TeacherRecord


def write_linear(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 3), round(value * 65535), dtype=np.uint16)
    assert cv2.imwrite(str(path), image)


def read_linear(path, device="cpu"):
    del device
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0


def fit_pad(image, *, max_side, multiple):
    del max_side, multiple
    return image, {"height": image.shape[0], "width": image.shape[1]}


def unpad(image, geometry):
    return image[:geometry["height"], :geometry["width"]]


def write_srgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.round(np.clip(image, 0, 1) * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(path), bgr)


def record(tmp_path: Path, scene: str) -> TeacherRecord:
    input_path = tmp_path / "inputs" / scene
    write_linear(input_path, 0.2)
    return TeacherRecord(
        scene_id=scene,
        input_path=str(input_path),
        baseline_path="unused",
        target_path="unused",
        selected_alpha=0.5,
        selected_level="a_p050",
        baseline_score=0.5,
        selected_score=0.7,
        score_delta=0.2,
        confidence=1.0,
        sample_weight=1.0,
        status="improved",
        split="test",
        distribution_status=DistributionStatus.IN_DOMAIN,
        reason="teacher",
        metadata={},
    )


class FakeRunner:
    def zero_reference(self, image):
        return np.clip(image, 0, 1)

    def predict(self, image, alpha):
        return {"output": np.clip(image + float(alpha) * 0.1, 0, 1)}


def test_render_policy_writes_baseline_student_and_records(tmp_path: Path) -> None:
    item = record(tmp_path, "scene.png")
    policy = RidgeAlphaPolicy(
        feature_names=FEATURE_NAMES,
        feature_mean=np.zeros(len(FEATURE_NAMES)),
        feature_scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.zeros(len(FEATURE_NAMES)),
        intercept=0.5,
        ood_threshold=1e9,
    )
    output = tmp_path / "render"
    rows = render_policy_with_runner(
        [item], policy=policy, runner=FakeRunner(), output_dir=output,
        read_linear=read_linear, fit_pad=fit_pad, unpad=unpad, write_srgb=write_srgb,
        max_side=16, multiple=4,
    )
    assert len(rows) == 1
    assert rows[0]["predicted_alpha"] == pytest.approx(0.5)
    assert (output / "baseline" / "scene.png").is_file()
    assert (output / "student" / "scene.png").is_file()
    assert (output / "render_records.jsonl").is_file()


def test_ood_policy_renders_exact_baseline(tmp_path: Path) -> None:
    item = record(tmp_path, "ood.png")
    policy = RidgeAlphaPolicy(
        feature_names=FEATURE_NAMES,
        feature_mean=np.full(len(FEATURE_NAMES), 1000.0),
        feature_scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.ones(len(FEATURE_NAMES)),
        intercept=0.8,
        ood_threshold=0.1,
    )
    output = tmp_path / "render"
    rows = render_policy_with_runner(
        [item], policy=policy, runner=FakeRunner(), output_dir=output,
        read_linear=read_linear, fit_pad=fit_pad, unpad=unpad, write_srgb=write_srgb,
        max_side=16, multiple=4,
    )
    assert rows[0]["predicted_alpha"] == 0.0
    assert rows[0]["policy_reason"] == "ood_fallback"
    baseline = cv2.imread(str(output / "baseline" / "ood.png"), cv2.IMREAD_UNCHANGED)
    student = cv2.imread(str(output / "student" / "ood.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(baseline, student)
