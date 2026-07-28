"""Data loading for same-scene non-pixel-aligned evaluation."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from ..unpaired_reference_data import _raw_metadata_to_lsrgb, _read_normalized_rgb, _resize_rgb
except ImportError:  # direct execution from photofinishing/evaluate
    from photofinishing.unpaired_reference_data import _raw_metadata_to_lsrgb, _read_normalized_rgb, _resize_rgb


_REQUIRED_COLUMNS = {"sample_id", "input_path", "reference_path", "split"}
_SEMANTIC_NAMES = ("skin", "sky", "vegetation")


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    input_path: Path
    reference_path: Path
    split: str
    metadata_path: Optional[Path]
    reference_repeat_path: Optional[Path]
    input_valid_mask_path: Optional[Path]
    reference_valid_mask_path: Optional[Path]
    input_ignore_mask_path: Optional[Path]
    reference_ignore_mask_path: Optional[Path]
    input_semantic_masks: Dict[str, Path]
    reference_semantic_masks: Dict[str, Path]
    scene_tags: Tuple[str, ...]


def _resolve_optional(root: Path, value: object, label: str) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    raw = Path(text)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_evaluation_manifest(path: str | Path, split: Optional[str] = None) -> list[EvaluationRecord]:
    """Loads an extended evaluation manifest.

    Required columns are compatible with the training manifest. Optional
    columns include repeat references, independent valid/ignore masks, paired
    semantic masks, metadata, and semicolon-separated scene tags.
    """

    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Evaluation manifest missing columns: {sorted(missing)}")
        rows = list(reader)

    root = manifest.parent
    seen: set[str] = set()
    records: list[EvaluationRecord] = []
    for row_index, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id") or "").strip()
        row_split = str(row.get("split") or "").strip()
        if not sample_id or not row_split:
            raise ValueError(f"Empty sample_id/split at CSV row {row_index}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in evaluation manifest: {sample_id}")
        seen.add(sample_id)
        input_path = _resolve_optional(root, row.get("input_path"), f"input image for {sample_id}")
        reference_path = _resolve_optional(root, row.get("reference_path"), f"reference image for {sample_id}")
        assert input_path is not None and reference_path is not None
        if split is not None and row_split != split:
            continue

        input_semantic: Dict[str, Path] = {}
        reference_semantic: Dict[str, Path] = {}
        for semantic in _SEMANTIC_NAMES:
            input_mask = _resolve_optional(
                root, row.get(f"input_{semantic}_mask_path"), f"input {semantic} mask for {sample_id}"
            )
            reference_mask = _resolve_optional(
                root, row.get(f"reference_{semantic}_mask_path"), f"reference {semantic} mask for {sample_id}"
            )
            if (input_mask is None) != (reference_mask is None):
                raise ValueError(
                    f"Semantic masks must be supplied for both input and reference: {sample_id}/{semantic}"
                )
            if input_mask is not None and reference_mask is not None:
                input_semantic[semantic] = input_mask
                reference_semantic[semantic] = reference_mask

        tags = tuple(
            dict.fromkeys(tag.strip() for tag in str(row.get("scene_tags") or "").split(";") if tag.strip())
        )
        records.append(EvaluationRecord(
            sample_id=sample_id,
            input_path=input_path,
            reference_path=reference_path,
            split=row_split,
            metadata_path=_resolve_optional(root, row.get("metadata_path"), f"metadata for {sample_id}"),
            reference_repeat_path=_resolve_optional(
                root, row.get("reference_repeat_path"), f"repeat reference for {sample_id}"
            ),
            input_valid_mask_path=_resolve_optional(
                root, row.get("input_valid_mask_path"), f"input valid mask for {sample_id}"
            ),
            reference_valid_mask_path=_resolve_optional(
                root, row.get("reference_valid_mask_path"), f"reference valid mask for {sample_id}"
            ),
            input_ignore_mask_path=_resolve_optional(
                root, row.get("input_ignore_mask_path"), f"input ignore mask for {sample_id}"
            ),
            reference_ignore_mask_path=_resolve_optional(
                root, row.get("reference_ignore_mask_path"), f"reference ignore mask for {sample_id}"
            ),
            input_semantic_masks=input_semantic,
            reference_semantic_masks=reference_semantic,
            scene_tags=tags,
        ))
    if not records:
        raise ValueError(f"No evaluation samples found for split={split!r}")
    return records


def load_mask(path: Optional[Path], target_shape: tuple[int, int]) -> np.ndarray:
    """Loads a binary mask with nearest-neighbor resizing.

    A missing valid mask means all pixels are valid. Ignore masks are handled
    by callers by inverting the returned binary mask.
    """

    height, width = target_shape
    if height <= 0 or width <= 0:
        raise ValueError("target_shape must be positive")
    if path is None:
        return np.ones((height, width), dtype=bool)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if np.issubdtype(resized.dtype, np.floating):
        return resized > 0.5
    return resized > 127


def _combined_valid_mask(
    valid_path: Optional[Path],
    ignore_path: Optional[Path],
    shape: tuple[int, int],
) -> np.ndarray:
    valid = load_mask(valid_path, shape)
    if ignore_path is not None:
        valid &= ~load_mask(ignore_path, shape)
    if not np.any(valid):
        raise ValueError("Valid/ignore masks remove every pixel")
    return valid


def load_record_images(
    record: EvaluationRecord,
    *,
    image_size: int,
    input_mode: str,
) -> dict[str, object]:
    """Loads independently resized images and masks for one evaluation record."""

    input_image = _read_normalized_rgb(record.input_path)
    if input_mode == "raw_metadata":
        if record.metadata_path is None:
            raise ValueError(f"raw_metadata mode requires metadata for {record.sample_id}")
        input_image = _raw_metadata_to_lsrgb(input_image, record.metadata_path)
    elif input_mode != "linear_srgb":
        raise ValueError("input_mode must be linear_srgb or raw_metadata")
    reference = _read_normalized_rgb(record.reference_path)
    repeat = _read_normalized_rgb(record.reference_repeat_path) if record.reference_repeat_path else None

    input_image = _resize_rgb(input_image, image_size)
    reference = _resize_rgb(reference, image_size)
    repeat = _resize_rgb(repeat, image_size) if repeat is not None else None
    input_shape = input_image.shape[:2]
    reference_shape = reference.shape[:2]
    input_valid = _combined_valid_mask(record.input_valid_mask_path, record.input_ignore_mask_path, input_shape)
    reference_valid = _combined_valid_mask(
        record.reference_valid_mask_path, record.reference_ignore_mask_path, reference_shape
    )

    input_semantic = {
        name: load_mask(path, input_shape) & input_valid for name, path in record.input_semantic_masks.items()
    }
    reference_semantic = {
        name: load_mask(path, reference_shape) & reference_valid
        for name, path in record.reference_semantic_masks.items()
    }
    return {
        "input": input_image,
        "reference": reference,
        "reference_repeat": repeat,
        "input_valid_mask": input_valid,
        "reference_valid_mask": reference_valid,
        "input_semantic_masks": input_semantic,
        "reference_semantic_masks": reference_semantic,
    }
