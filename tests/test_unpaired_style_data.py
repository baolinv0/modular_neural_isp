import numpy as np
from pathlib import Path

from photofinishing.unpaired_style.contracts import RegionPair, StylePairRecord
from photofinishing.unpaired_style.data import UnpairedStyleDataset


def test_dataset_supports_non_aligned_shapes_and_independent_masks(tmp_path: Path):
  np.save(tmp_path / "input.npy", np.full((10, 18, 3), 0.3, np.float32))
  np.save(tmp_path / "reference.npy", np.full((16, 9, 3), 0.6, np.float32))
  np.save(tmp_path / "input_mask.npy", np.ones((10, 18), np.float32))
  np.save(tmp_path / "reference_mask.npy", np.ones((16, 9), np.float32))
  record = StylePairRecord(
    sample_id="s", scene_group="g", input_path=str(tmp_path / "input.npy"),
    reference_path=str(tmp_path / "reference.npy"), split="train",
    regions={"face": RegionPair(
      input_mask=str(tmp_path / "input_mask.npy"),
      reference_mask=str(tmp_path / "reference_mask.npy"), weight=2.0)},
  )
  item = UnpairedStyleDataset([record], image_size=12)[0]
  assert item["input"].shape == (3, 12, 12)
  assert item["reference"].shape == (3, 12, 12)
  assert item["input_masks"].shape == (1, 1, 12, 12)
  assert item["region_weights"].tolist() == [2.0]
