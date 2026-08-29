from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

from .config import LabelConfig


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Small dependency-free square dilation used only for semantic regions."""
    if radius <= 0:
        return mask.astype(bool, copy=True)
    padded = np.pad(mask.astype(bool), radius, mode="constant")
    h, w = mask.shape
    result = np.zeros((h, w), dtype=bool)
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            result |= padded[y:y + h, x:x + w]
    return result


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    return ~_dilate(~mask.astype(bool), radius)


def _soft_array(value: np.ndarray | None, shape: tuple[int, int], fallback: np.ndarray) -> np.ndarray:
    if value is None:
        return fallback.astype(np.float32)
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"soft mask shape {value.shape} does not match expected {shape}")
    if value.max(initial=0.0) > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _derived_binary(value: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if value is None:
        return None
    value = np.asarray(value, dtype=bool)
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} does not match base mask shape {shape}")
    return value


@dataclass
class SemanticMasks:
    """Binary semantic regions plus optional soft maps used for TM generation.

    The first three fields intentionally retain the V1 constructor contract.
    Additional regions are derived when they are not supplied.
    """

    face: np.ndarray
    skin: np.ndarray
    background: np.ndarray
    human: np.ndarray | None = None
    face_core: np.ndarray | None = None
    face_inner_ring: np.ndarray | None = None
    face_outer_ring: np.ndarray | None = None
    soft_face: np.ndarray | None = None
    soft_skin: np.ndarray | None = None
    soft_human: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = np.asarray(self.face).shape
        if len(shape) != 2:
            raise ValueError("semantic masks must be 2D")
        self.face = np.asarray(self.face, dtype=bool)
        self.skin = np.asarray(self.skin, dtype=bool)
        background = np.asarray(self.background, dtype=bool)
        if self.skin.shape != shape or background.shape != shape:
            raise ValueError("semantic masks must have matching shapes")
        self.human = (np.asarray(self.human, dtype=bool) if self.human is not None else np.zeros(shape, dtype=bool))
        if self.human.shape != shape:
            raise ValueError("human mask must match face mask shape")
        self.soft_face = _soft_array(self.soft_face, shape, self.face)
        self.soft_skin = _soft_array(self.soft_skin, shape, self.skin)
        self.soft_human = _soft_array(self.soft_human, shape, self.human)
        # Operational regions form a partition. Skin remains a face subregion
        # for existing IQA calls, while ``human`` means body outside the face.
        self.skin &= self.face
        self.human &= ~self.face
        self.background = ~(self.face | self.human)
        self.soft_skin *= self.face
        self.soft_human *= ~self.face
        supplied_core = _derived_binary(self.face_core, shape, "face_core")
        supplied_inner = _derived_binary(self.face_inner_ring, shape, "face_inner_ring")
        supplied_outer = _derived_binary(self.face_outer_ring, shape, "face_outer_ring")
        self.face_core = (
            supplied_core & self.face
            if supplied_core is not None
            else _erode(self.face, radius=1)
        )
        if not self.face_core.any():
            self.face_core = self.face.copy()
        inner_extent = _dilate(self.face, radius=2)
        outer_extent = _dilate(self.face, radius=5)
        self.face_inner_ring = (
            supplied_inner & ~self.face
            if supplied_inner is not None
            else inner_extent & ~self.face
        )
        self.face_outer_ring = (
            supplied_outer & ~self.face & ~self.face_inner_ring
            if supplied_outer is not None
            else outer_extent & ~inner_extent
        )

    @classmethod
    def from_label_map(
        cls,
        label: np.ndarray,
        labels: LabelConfig,
        *,
        soft_face: np.ndarray | None = None,
        soft_skin: np.ndarray | None = None,
        soft_human: np.ndarray | None = None,
    ) -> "SemanticMasks":
        label = np.asarray(label)
        if label.ndim != 2:
            raise ValueError("label map must be a single-channel image")
        skin = (_soft_array(soft_skin, label.shape, label == labels.skin) >= 0.5) if soft_skin is not None else (label == labels.skin)
        face = (_soft_array(soft_face, label.shape, (label == labels.face) | skin) >= 0.5) if soft_face is not None else (label == labels.face)
        face |= skin
        human = (_soft_array(soft_human, label.shape, label == labels.human) >= 0.5) if soft_human is not None else (label == labels.human)
        human &= ~face
        background = ~(face | human)
        # Soft maps are authoritative for their associated region when available.
        if not face.any():
            raise ValueError("semantic mask has no face pixels")
        if not background.any():
            raise ValueError("semantic mask has no background pixels")
        if not skin.any():
            skin = face.copy()
        return cls(
            face=face,
            skin=skin,
            human=human,
            background=background,
            soft_face=soft_face,
            soft_skin=soft_skin,
            soft_human=soft_human,
        )


def load_soft_mask(path: str | Path) -> np.ndarray:
    """Load a one-channel soft segmentation map normalized to [0, 1]."""
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return array / 255.0


def load_label_map(
    path: str | Path,
    labels: LabelConfig,
    *,
    soft_face_path: str | Path | None = None,
    soft_skin_path: str | Path | None = None,
    soft_human_path: str | Path | None = None,
) -> SemanticMasks:
    label = np.asarray(Image.open(path).convert("L"))
    return SemanticMasks.from_label_map(
        label,
        labels,
        soft_face=load_soft_mask(soft_face_path) if soft_face_path else None,
        soft_skin=load_soft_mask(soft_skin_path) if soft_skin_path else None,
        soft_human=load_soft_mask(soft_human_path) if soft_human_path else None,
    )


def resize_masks(masks: SemanticMasks, size_hw: tuple[int, int]) -> SemanticMasks:
    h, w = size_hw

    def binary(mask: np.ndarray) -> np.ndarray:
        im = Image.fromarray(mask.astype(np.uint8) * 255)
        return np.asarray(im.resize((w, h), Image.Resampling.NEAREST)) > 127

    def soft(mask: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.clip(mask, 0.0, 1.0) * 255.0).convert("L")
        return np.asarray(im.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    return SemanticMasks(
        face=binary(masks.face), skin=binary(masks.skin), background=binary(masks.background),
        human=binary(masks.human), face_core=binary(masks.face_core),
        face_inner_ring=binary(masks.face_inner_ring), face_outer_ring=binary(masks.face_outer_ring),
        soft_face=soft(masks.soft_face), soft_skin=soft(masks.soft_skin), soft_human=soft(masks.soft_human),
    )
