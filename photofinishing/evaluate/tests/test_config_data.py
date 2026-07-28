import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from photofinishing.evaluate.config import load_experiment_config
from photofinishing.evaluate.data import load_evaluation_manifest, load_mask


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_config_supports_multiple_groups_and_stage2_variants(tmp_path):
    for name in ["pre0.pth", "s10.pth", "a0.pth", "f0.pth", "pre1.pth", "s11.pth", "a1.pth"]:
        _touch(tmp_path / name)
    (tmp_path / "a0.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "experiments.json"
    config_path.write_text(json.dumps({
        "schema_version": 1,
        "groups": [
            {
                "name": "seed0",
                "pretrained": {"checkpoint": "pre0.pth"},
                "stage1": {"checkpoint": "s10.pth"},
                "stage2": [
                    {"name": "affine", "checkpoint": "a0.pth", "run_config": "a0.json"},
                    {"name": "full_lut", "checkpoint": "f0.pth"},
                ],
            },
            {
                "name": "seed1",
                "pretrained": {"checkpoint": "pre1.pth"},
                "stage1": {"checkpoint": "s11.pth"},
                "stage2": [{"name": "affine", "checkpoint": "a1.pth"}],
            },
        ],
    }), encoding="utf-8")

    config = load_experiment_config(config_path)
    assert [group.name for group in config.groups] == ["seed0", "seed1"]
    assert [model.name for model in config.groups[0].stage2] == ["affine", "full_lut"]
    assert config.groups[0].pretrained.checkpoint == (tmp_path / "pre0.pth").resolve()
    assert config.groups[0].stage2[0].run_config == (tmp_path / "a0.json").resolve()


def test_config_rejects_duplicate_group_or_stage2_names(tmp_path):
    checkpoint = _touch(tmp_path / "model.pth")
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({
        "schema_version": 1,
        "groups": [
            {
                "name": "dup",
                "pretrained": {"checkpoint": str(checkpoint)},
                "stage1": {"checkpoint": str(checkpoint)},
                "stage2": [
                    {"name": "same", "checkpoint": str(checkpoint)},
                    {"name": "same", "checkpoint": str(checkpoint)},
                ],
            }
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate stage2"):
        load_experiment_config(config_path)


def test_manifest_loads_repeat_reference_masks_and_scene_tags(tmp_path):
    image = np.full((8, 8, 3), 127, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    mask[:, :4] = 255
    for name in ["input.png", "reference.png", "repeat.png"]:
        cv2.imwrite(str(tmp_path / name), image)
    for name in ["input_valid.png", "reference_valid.png", "input_skin.png", "reference_skin.png"]:
        cv2.imwrite(str(tmp_path / name), mask)

    manifest = tmp_path / "manifest.csv"
    fields = [
        "sample_id", "input_path", "reference_path", "split", "reference_repeat_path",
        "input_valid_mask_path", "reference_valid_mask_path",
        "input_skin_mask_path", "reference_skin_mask_path", "scene_tags",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "sample_id": "scene_1", "input_path": "input.png", "reference_path": "reference.png",
            "split": "test", "reference_repeat_path": "repeat.png",
            "input_valid_mask_path": "input_valid.png", "reference_valid_mask_path": "reference_valid.png",
            "input_skin_mask_path": "input_skin.png", "reference_skin_mask_path": "reference_skin.png",
            "scene_tags": "portrait;indoor",
        })

    records = load_evaluation_manifest(manifest, split="test")
    assert len(records) == 1
    record = records[0]
    assert record.reference_repeat_path == (tmp_path / "repeat.png").resolve()
    assert record.scene_tags == ("portrait", "indoor")
    assert record.input_semantic_masks["skin"] == (tmp_path / "input_skin.png").resolve()
    loaded = load_mask(record.input_valid_mask_path, (4, 4))
    assert loaded.dtype == np.bool_
    assert 0.4 < loaded.mean() < 0.6
