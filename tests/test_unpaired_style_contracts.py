import json
from pathlib import Path

import numpy as np
import pytest

from photofinishing.unpaired_style.contracts import (
    Stage1LossWeights,
    StylePairRecord,
    TwoStageTrainingConfig,
    load_manifest,
    validate_disjoint_manifests,
)


def _write_array(path: Path, value: float = 0.5):
  np.save(path, np.full((8, 8, 3), value, dtype=np.float32))


def test_manifest_resolves_paths_and_rejects_scene_leakage(tmp_path):
  _write_array(tmp_path / "input.npy")
  _write_array(tmp_path / "reference.npy")
  row = {
    "sample_id": "a",
    "scene_group": "scene-1",
    "input_path": "input.npy",
    "reference_path": "reference.npy",
    "split": "train",
  }
  manifest = tmp_path / "train.jsonl"
  manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
  records = load_manifest(str(manifest))
  assert Path(records[0].input_path).is_absolute()
  validation = [StylePairRecord(
    sample_id="b", scene_group="scene-1", input_path=records[0].input_path,
    reference_path=records[0].reference_path, split="validation")]
  with pytest.raises(ValueError, match="scene groups"):
    validate_disjoint_manifests(records, validation)


def test_split_validation_rejects_reused_input_with_renamed_scene():
  train = [StylePairRecord(
    sample_id="train", scene_group="train-scene", input_path="same-input.npy",
    reference_path="train-reference.npy", split="train")]
  validation = [StylePairRecord(
    sample_id="validation", scene_group="validation-scene", input_path="same-input.npy",
    reference_path="validation-reference.npy", split="validation")]
  with pytest.raises(ValueError, match="input images"):
    validate_disjoint_manifests(train, validation)


def test_config_rejects_non_positive_learning_rate():
  with pytest.raises(ValueError, match="stage1_learning_rate"):
    TwoStageTrainingConfig(stage1_learning_rate=0.0)


def test_config_rejects_negative_loss_weight():
  with pytest.raises(ValueError, match="stage1.exposure"):
    TwoStageTrainingConfig(stage1=Stage1LossWeights(exposure=-1.0))
