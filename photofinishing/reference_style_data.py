"""Dataset utilities for non-pixel-aligned reference-style fine-tuning."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ReferenceStyleRecord:
  """One same-scene pair without a pixel-alignment requirement."""

  sample_id: str
  input_path: str
  reference_path: str
  split: str
  weight: float = 1.0

  @classmethod
  def from_dict(cls, payload: Dict[str, object], base_dir: str) -> "ReferenceStyleRecord":
    required = {"sample_id", "input_path", "reference_path", "split"}
    missing = sorted(required - set(payload))
    if missing:
      raise ValueError(f"Missing manifest fields: {missing}")
    sample_id = str(payload["sample_id"]).strip()
    split = str(payload["split"]).strip().lower()
    if not sample_id:
      raise ValueError("sample_id must be non-empty")
    if split not in _VALID_SPLITS:
      raise ValueError(f"Unsupported split '{split}'")
    weight = float(payload.get("weight", 1.0))
    if not np.isfinite(weight) or weight <= 0:
      raise ValueError("weight must be finite and positive")

    def resolve(value: object) -> str:
      path = os.path.expanduser(str(value))
      if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
      return os.path.abspath(path)

    input_path = resolve(payload["input_path"])
    reference_path = resolve(payload["reference_path"])
    if input_path == reference_path:
      raise ValueError(f"{sample_id}: input and reference must be different files")
    for role, path in (("input", input_path), ("reference", reference_path)):
      if not os.path.isfile(path):
        raise FileNotFoundError(f"{sample_id}: missing {role} file: {path}")
    return cls(sample_id, input_path, reference_path, split, weight)


def load_reference_style_manifest(path: str, split: Optional[str] = None) -> List[ReferenceStyleRecord]:
  """Loads and validates a JSONL manifest."""
  manifest_path = os.path.abspath(os.path.expanduser(path))
  base_dir = os.path.dirname(manifest_path)
  records: List[ReferenceStyleRecord] = []
  seen_ids = set()
  seen_pairs = set()
  with open(manifest_path, "r", encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
      line = raw_line.strip()
      if not line:
        continue
      try:
        payload = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
      record = ReferenceStyleRecord.from_dict(payload, base_dir)
      if record.sample_id in seen_ids:
        raise ValueError(f"Duplicate sample_id: {record.sample_id}")
      pair_key = (record.input_path, record.reference_path)
      if pair_key in seen_pairs:
        raise ValueError(f"Duplicate input/reference pair: {pair_key}")
      seen_ids.add(record.sample_id)
      seen_pairs.add(pair_key)
      if split is None or record.split == split:
        records.append(record)
  if not records:
    suffix = f" for split '{split}'" if split else ""
    raise ValueError(f"Manifest contains no records{suffix}")
  return records


def read_rgb_image(path: str) -> np.ndarray:
  """Reads uint8/uint16/float RGB files and returns float32 in [0, 1]."""
  suffix = Path(path).suffix.lower()
  if suffix == ".npy":
    image = np.load(path)
  else:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
      raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 2:
      image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
      image = image[..., :3]
    if image.ndim != 3 or image.shape[2] != 3:
      raise ValueError(f"Expected 3-channel RGB image, got {image.shape}: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if image.ndim != 3 or image.shape[2] != 3:
    raise ValueError(f"Expected HxWx3 array, got {image.shape}: {path}")
  if np.issubdtype(image.dtype, np.integer):
    max_value = float(np.iinfo(image.dtype).max)
    image = image.astype(np.float32) / max_value
  else:
    image = image.astype(np.float32)
  if not np.isfinite(image).all():
    raise ValueError(f"Non-finite image values: {path}")
  if image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
    raise ValueError(f"Image values must be normalized to [0, 1]: {path}")
  return np.clip(image, 0.0, 1.0)


def image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
  if image_size <= 0:
    raise ValueError("image_size must be positive")
  resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
  return torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float()


class ReferenceStyleDataset(Dataset):
  """Loads Bin and same-scene AGT independently; no geometric pairing is assumed."""

  def __init__(self, manifest_path: str, split: str, image_size: int = 512):
    self.records = load_reference_style_manifest(manifest_path, split=split)
    self.image_size = int(image_size)

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, index: int) -> Dict[str, object]:
    record = self.records[index]
    input_image = image_to_tensor(read_rgb_image(record.input_path), self.image_size)
    reference_image = image_to_tensor(read_rgb_image(record.reference_path), self.image_size)
    return {
      "sample_id": record.sample_id,
      "input": input_image,
      "reference": reference_image,
      "weight": torch.tensor(record.weight, dtype=torch.float32),
    }
