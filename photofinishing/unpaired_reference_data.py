"""Manifest-driven same-scene, non-pixel-aligned reference dataset.

The input image is the camera-B photofinishing input. The reference image is
an A-camera product rendering of the same scene. The two images are resized
independently; no pixel correspondence is assumed or created.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_REQUIRED_COLUMNS = {"sample_id", "input_path", "reference_path", "split"}
_ALLOWED_INPUT_MODES = {"linear_srgb", "raw_metadata"}


@dataclass(frozen=True)
class ReferenceRecord:
    sample_id: str
    input_path: Path
    reference_path: Path
    split: str
    metadata_path: Optional[Path] = None


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_reference_manifest(manifest_path: str | Path, split: Optional[str] = None) -> List[ReferenceRecord]:
    """Loads and validates the experiment manifest.

    CSV columns: sample_id,input_path,reference_path,split[,metadata_path].
    All sample IDs must be unique across the complete manifest so that a scene
    cannot silently enter two splits under the same identity.
    """
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)

    records: List[ReferenceRecord] = []
    seen_ids: set[str] = set()
    root = manifest.parent
    for row_index, row in enumerate(rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        row_split = (row.get("split") or "").strip()
        if not sample_id or not row_split:
            raise ValueError(f"Empty sample_id/split at CSV row {row_index}")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        seen_ids.add(sample_id)

        input_path = _resolve_path(root, (row.get("input_path") or "").strip())
        reference_path = _resolve_path(root, (row.get("reference_path") or "").strip())
        metadata_value = (row.get("metadata_path") or "").strip()
        metadata_path = _resolve_path(root, metadata_value) if metadata_value else None

        for label, path in (("input", input_path), ("reference", reference_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} image for {sample_id} not found: {path}")
        if metadata_path is not None and not metadata_path.is_file():
            raise FileNotFoundError(f"metadata for {sample_id} not found: {metadata_path}")

        if split is None or row_split == split:
            records.append(
                ReferenceRecord(
                    sample_id=sample_id,
                    input_path=input_path,
                    reference_path=reference_path,
                    split=row_split,
                    metadata_path=metadata_path,
                )
            )

    if not records:
        raise ValueError(f"No samples found for split={split!r}")
    return records


def _read_normalized_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    elif np.issubdtype(image.dtype, np.floating):
        scale = 1.0
    else:
        raise TypeError(f"Unsupported image dtype {image.dtype} for {path}")
    image = image.astype(np.float32) / scale
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite image values: {path}")
    return np.clip(image, 0.0, 1.0)


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > image_size else cv2.INTER_LINEAR
    return cv2.resize(image, (image_size, image_size), interpolation=interpolation).astype(np.float32)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _raw_metadata_to_lsrgb(image: np.ndarray, metadata_path: Path) -> np.ndarray:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata: Dict[str, object] = json.load(handle)
    try:
        illuminant = np.asarray(metadata["cam_illum"], dtype=np.float32)
        ccm = np.asarray(metadata["ccm"], dtype=np.float32).reshape(3, 3)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid cam_illum/ccm metadata: {metadata_path}") from exc
    if illuminant.shape != (3,) or not np.isfinite(illuminant).all() or not np.isfinite(ccm).all():
        raise ValueError(f"Invalid finite metadata values: {metadata_path}")
    wb_gain = illuminant[1] / np.maximum(illuminant, 1e-8)
    color_matrix = ccm @ np.diag(wb_gain)
    converted = image.reshape(-1, 3) @ color_matrix.T
    return np.clip(converted.reshape(image.shape), 0.0, 1.0).astype(np.float32)


class ReferenceStyleDataset(Dataset):
    """Returns independently resized B-input/A-reference pairs.

    No crop, warp, or shared geometric augmentation is performed. The losses
    consume only position-independent statistics, so resizing is used solely
    to bound memory and batch tensors.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int = 512,
        input_mode: str = "linear_srgb",
    ) -> None:
        if input_mode not in _ALLOWED_INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(_ALLOWED_INPUT_MODES)}")
        self.records = load_reference_manifest(manifest_path, split=split)
        self.image_size = int(image_size)
        self.input_mode = input_mode
        if self.input_mode == "raw_metadata":
            missing = [r.sample_id for r in self.records if r.metadata_path is None]
            if missing:
                raise ValueError(f"raw_metadata mode requires metadata_path: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        input_image = _read_normalized_rgb(record.input_path)
        if self.input_mode == "raw_metadata":
            assert record.metadata_path is not None
            input_image = _raw_metadata_to_lsrgb(input_image, record.metadata_path)
        reference = _read_normalized_rgb(record.reference_path)
        return {
            "sample_id": record.sample_id,
            "input_image": _to_tensor(_resize_rgb(input_image, self.image_size)),
            "reference_image": _to_tensor(_resize_rgb(reference, self.image_size)),
        }
