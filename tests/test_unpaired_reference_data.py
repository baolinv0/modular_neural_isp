from pathlib import Path
import csv

import cv2
import numpy as np
import pytest

from photofinishing.unpaired_reference_data import ReferenceStyleDataset, load_reference_manifest


def _write_rgb(path: Path, shape=(8, 12, 3), value=128):
    image = np.full(shape, value, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _write_manifest(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "input_path", "reference_path", "split"])
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_accepts_non_aligned_shapes_and_resizes_independently(tmp_path):
    _write_rgb(tmp_path / "b.png", shape=(9, 13, 3), value=80)
    _write_rgb(tmp_path / "a.png", shape=(15, 7, 3), value=160)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"sample_id": "scene1", "input_path": "b.png", "reference_path": "a.png", "split": "train"}
    ])
    dataset = ReferenceStyleDataset(manifest, split="train", image_size=16)
    item = dataset[0]
    assert item["input_image"].shape == (3, 16, 16)
    assert item["reference_image"].shape == (3, 16, 16)
    assert item["input_image"].mean() < item["reference_image"].mean()


def test_manifest_rejects_duplicate_sample_ids_across_splits(tmp_path):
    _write_rgb(tmp_path / "b.png")
    _write_rgb(tmp_path / "a.png")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"sample_id": "same", "input_path": "b.png", "reference_path": "a.png", "split": "train"},
        {"sample_id": "same", "input_path": "b.png", "reference_path": "a.png", "split": "val"},
    ])
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_reference_manifest(manifest, split="train")
