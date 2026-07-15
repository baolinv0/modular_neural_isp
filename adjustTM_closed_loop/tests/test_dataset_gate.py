from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from adjustTM_closed_loop.dataset_gate import DatasetGate, DatasetGateConfig
from adjustTM_closed_loop.schemas import DatasetClass


LEVELS = (
    "a_m100", "a_m075", "a_m050", "a_m025", "a_000",
    "a_p025", "a_p050", "a_p075", "a_p100",
)


def _write_rgb(path: Path, value: float, dtype=np.uint8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = np.iinfo(dtype).max
    image = np.full((12, 16, 3), round(value * maximum), dtype=dtype)
    assert cv2.imwrite(str(path), image)


def _make_dataset(root: Path, values: list[float], *, input_dtype=np.uint16) -> tuple[Path, Path]:
    input_dir = root / "input_linear"
    gt_root = root / "gt_levels"
    _write_rgb(input_dir / "scene.png", 0.2, dtype=input_dtype)
    for level, value in zip(LEVELS, values, strict=True):
        _write_rgb(gt_root / level / "scene.png", value)
    return input_dir, gt_root


def test_gate_classifies_clean_monotonic_sequence(tmp_path: Path) -> None:
    input_dir, gt_root = _make_dataset(tmp_path, [0.08, 0.11, 0.15, 0.20, 0.26, 0.34, 0.44, 0.56, 0.70])
    report = DatasetGate(DatasetGateConfig()).run(input_dir, gt_root)
    assert report.summary.total_scenes == 1
    assert report.summary.counts[DatasetClass.CLEAN] == 1
    assert report.scenes[0].classification is DatasetClass.CLEAN
    assert report.scenes[0].monotonic_violation_rate == 0.0


def test_gate_marks_non_monotonic_sequence_invalid(tmp_path: Path) -> None:
    input_dir, gt_root = _make_dataset(tmp_path, [0.08, 0.11, 0.15, 0.20, 0.30, 0.27, 0.44, 0.56, 0.70])
    report = DatasetGate(DatasetGateConfig()).run(input_dir, gt_root)
    assert report.scenes[0].classification is DatasetClass.INVALID
    assert "non_monotonic" in report.scenes[0].reasons


def test_gate_marks_high_clip_endpoint_boundary(tmp_path: Path) -> None:
    input_dir, gt_root = _make_dataset(tmp_path, [0.08, 0.11, 0.15, 0.20, 0.26, 0.34, 0.44, 0.56, 1.0])
    report = DatasetGate(DatasetGateConfig(max_endpoint_clip_ratio=0.1)).run(input_dir, gt_root)
    assert report.scenes[0].classification is DatasetClass.BOUNDARY
    assert "endpoint_clipping" in report.scenes[0].reasons


def test_gate_fails_when_level_file_is_missing(tmp_path: Path) -> None:
    input_dir, gt_root = _make_dataset(tmp_path, [0.08, 0.11, 0.15, 0.20, 0.26, 0.34, 0.44, 0.56, 0.70])
    (gt_root / "a_p100" / "scene.png").unlink()
    with pytest.raises(FileNotFoundError, match="a_p100"):
        DatasetGate(DatasetGateConfig()).run(input_dir, gt_root)


def test_gate_rejects_non_uint16_linear_input(tmp_path: Path) -> None:
    input_dir, gt_root = _make_dataset(
        tmp_path, [0.08, 0.11, 0.15, 0.20, 0.26, 0.34, 0.44, 0.56, 0.70], input_dtype=np.uint8
    )
    with pytest.raises(TypeError, match="uint16"):
        DatasetGate(DatasetGateConfig()).run(input_dir, gt_root)
