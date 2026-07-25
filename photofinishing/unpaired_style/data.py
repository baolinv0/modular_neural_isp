"""Dataset for same-scene non-pixel-aligned photofinishing adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .contracts import StylePairRecord


def _normalize_array(array: np.ndarray) -> np.ndarray:
  if array.dtype == np.uint8:
    array = array.astype(np.float32) / 255.0
  elif array.dtype == np.uint16:
    array = array.astype(np.float32) / 65535.0
  else:
    array = array.astype(np.float32)
  if not np.isfinite(array).all():
    raise ValueError("image contains NaN or Inf")
  if array.min() < 0:
    raise ValueError("image contains negative values")
  return np.clip(array, 0.0, 1.0)


def read_tensor(path: str, *, grayscale: bool = False) -> torch.Tensor:
  """Read .npy/.pt or a conventional image into CHW float32 [0, 1]."""
  suffix = Path(path).suffix.lower()
  if suffix == ".npy":
    array = np.load(path, allow_pickle=False)
  elif suffix in {".pt", ".pth"}:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
      for key in ("image", "tensor", "data"):
        if key in value:
          value = value[key]
          break
    if not torch.is_tensor(value):
      raise ValueError(f"unsupported tensor payload in {path}")
    array = value.detach().cpu().numpy()
  else:
    try:
      import cv2
      flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
      array = cv2.imread(path, flag)
      if array is None:
        raise ValueError(f"failed to read image: {path}")
      if not grayscale and array.ndim == 3:
        if array.shape[2] == 4:
          array = array[:, :, :3]
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    except ImportError:
      from PIL import Image
      image = Image.open(path)
      image = image.convert("L" if grayscale else "RGB")
      array = np.asarray(image)
  array = _normalize_array(array)
  if array.ndim == 2:
    array = array[..., None]
  if array.ndim != 3:
    raise ValueError(f"expected HWC or CHW image, got {array.shape}")
  if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
    array = np.transpose(array, (1, 2, 0))
  if grayscale:
    if array.shape[-1] > 1:
      array = array.mean(axis=-1, keepdims=True)
  elif array.shape[-1] != 3:
    raise ValueError(f"expected RGB image, got {array.shape}")
  return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def _resize(tensor: torch.Tensor, size: int, *, mask: bool = False) -> torch.Tensor:
  mode = "nearest" if mask else "bilinear"
  kwargs = {} if mask else {"align_corners": False}
  return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode=mode, **kwargs).squeeze(0)


class UnpairedStyleDataset(Dataset):
  """Loads two independently resized images and optional independent region masks."""

  def __init__(self, records: Sequence[StylePairRecord], image_size: int = 512):
    if image_size <= 0:
      raise ValueError("image_size must be positive")
    self.records = list(records)
    if not self.records:
      raise ValueError("records must not be empty")
    self.image_size = image_size
    self.region_names = sorted({name for record in self.records for name in record.regions})

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, index: int) -> Dict[str, object]:
    record = self.records[index]
    input_image = _resize(read_tensor(record.input_path), self.image_size)
    reference_image = _resize(read_tensor(record.reference_path), self.image_size)
    input_masks: List[torch.Tensor] = []
    reference_masks: List[torch.Tensor] = []
    region_valid: List[float] = []
    region_weights: List[float] = []
    for name in self.region_names:
      region = record.regions.get(name)
      if region is None:
        zero = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        input_masks.append(zero)
        reference_masks.append(zero.clone())
        region_valid.append(0.0)
        region_weights.append(0.0)
      else:
        input_mask = _resize(read_tensor(region.input_mask, grayscale=True), self.image_size, mask=True)
        reference_mask = _resize(read_tensor(region.reference_mask, grayscale=True), self.image_size, mask=True)
        input_masks.append((input_mask > 0.5).float())
        reference_masks.append((reference_mask > 0.5).float())
        valid = float(input_mask.sum() > 0 and reference_mask.sum() > 0)
        region_valid.append(valid)
        region_weights.append(region.weight if valid else 0.0)
    if input_masks:
      input_masks_tensor = torch.stack(input_masks, dim=0)
      reference_masks_tensor = torch.stack(reference_masks, dim=0)
    else:
      input_masks_tensor = torch.zeros((0, 1, self.image_size, self.image_size))
      reference_masks_tensor = torch.zeros((0, 1, self.image_size, self.image_size))
    return {
      "sample_id": record.sample_id,
      "scene_group": record.scene_group,
      "input": input_image,
      "reference": reference_image,
      "confidence": torch.tensor(record.confidence, dtype=torch.float32),
      "input_masks": input_masks_tensor,
      "reference_masks": reference_masks_tensor,
      "region_valid": torch.tensor(region_valid, dtype=torch.float32),
      "region_weights": torch.tensor(region_weights, dtype=torch.float32),
      "region_names": self.region_names,
    }
