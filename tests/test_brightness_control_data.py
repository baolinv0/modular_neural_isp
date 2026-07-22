from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from photofinishing.control_data import (
    LEVELS,
    BrightnessPairDataset,
    build_balanced_pair_schedule,
    read_linear_png,
)


def _write_rgb16(path: Path, value: int):
    arr = np.full((4, 5, 3), value, dtype=np.uint16)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _write_rgb8(path: Path, value: int):
    arr = np.full((4, 5, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def test_read_linear_png_normalizes_uint16(tmp_path):
    path = tmp_path / "x.png"
    _write_rgb16(path, 32768)
    image = read_linear_png(path)
    assert image.shape == (3, 4, 5)
    assert abs(float(image.mean()) - 32768 / 65535.0) < 1e-6


def test_balanced_pair_schedule_uses_all_pairs_and_equal_level_frequency():
    schedule = build_balanced_pair_schedule(num_scenes=3, seed=7)
    assert len(schedule) == 3 * 36
    counts = Counter()
    for _, low_idx, high_idx in schedule:
        assert low_idx < high_idx
        counts[low_idx] += 1
        counts[high_idx] += 1
    assert set(counts.values()) == {3 * 8}


def test_dataset_requires_identical_filenames_across_levels(tmp_path):
    inputs = tmp_path / "inputs"
    gt_root = tmp_path / "gt"
    inputs.mkdir()
    _write_rgb16(inputs / "scene.png", 12000)
    for level_name, _ in LEVELS:
        (gt_root / level_name).mkdir(parents=True)
        _write_rgb8(gt_root / level_name / "scene.png", 100)

    dataset = BrightnessPairDataset(inputs, gt_root, image_size=None, seed=0)
    sample = dataset[0]
    assert sample["input"].shape == (3, 4, 5)
    assert sample["target_low"].shape == (3, 4, 5)
    assert sample["alpha_low"] < sample["alpha_high"]
