import json

import cv2
import numpy as np
import pytest

from photofinishing.reference_style_data import (
  ReferenceStyleDataset,
  load_reference_style_manifest,
)


def _write_rgb(path, image):
  cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def test_dataset_loads_unaligned_uint16_input_and_uint8_reference(tmp_path):
  input_path = tmp_path / "bin.png"
  reference_path = tmp_path / "agt.png"
  _write_rgb(input_path, np.full((12, 20, 3), 32768, dtype=np.uint16))
  _write_rgb(reference_path, np.full((25, 9, 3), 128, dtype=np.uint8))
  manifest = tmp_path / "manifest.jsonl"
  manifest.write_text(json.dumps({
    "sample_id": "scene-1",
    "input_path": input_path.name,
    "reference_path": reference_path.name,
    "split": "train",
  }) + "\n", encoding="utf-8")
  dataset = ReferenceStyleDataset(str(manifest), "train", image_size=16)
  sample = dataset[0]
  assert sample["input"].shape == (3, 16, 16)
  assert sample["reference"].shape == (3, 16, 16)
  assert float(sample["input"].mean()) == pytest.approx(32768 / 65535, rel=1e-4)
  assert float(sample["reference"].mean()) == pytest.approx(128 / 255, rel=1e-4)


def test_manifest_rejects_duplicate_ids(tmp_path):
  input_path = tmp_path / "bin.png"
  reference_path = tmp_path / "agt.png"
  _write_rgb(input_path, np.zeros((4, 4, 3), dtype=np.uint8))
  _write_rgb(reference_path, np.ones((4, 4, 3), dtype=np.uint8))
  row = {
    "sample_id": "same",
    "input_path": input_path.name,
    "reference_path": reference_path.name,
    "split": "train",
  }
  manifest = tmp_path / "manifest.jsonl"
  manifest.write_text(
    json.dumps(row) + "\n" + json.dumps(row) + "\n",
    encoding="utf-8",
  )
  with pytest.raises(ValueError, match="Duplicate sample_id"):
    load_reference_style_manifest(str(manifest))
