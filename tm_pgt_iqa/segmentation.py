from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

from .config import LabelConfig


@dataclass
class SemanticMasks:
    face: np.ndarray
    skin: np.ndarray
    background: np.ndarray


def load_label_map(path: str | Path, labels: LabelConfig) -> SemanticMasks:
    label = np.asarray(Image.open(path).convert("L"))
    skin = label == labels.skin
    face = (label == labels.face) | skin
    background = label == labels.background
    if not face.any():
        raise ValueError(f"semantic mask has no face pixels: {path}")
    if not background.any():
        raise ValueError(f"semantic mask has no background pixels: {path}")
    if not skin.any():
        skin = face.copy()
    return SemanticMasks(face=face, skin=skin, background=background)


def resize_masks(masks: SemanticMasks, size_hw: tuple[int, int]) -> SemanticMasks:
    h, w = size_hw
    def r(mask: np.ndarray) -> np.ndarray:
        im = Image.fromarray((mask.astype(np.uint8) * 255))
        return np.asarray(im.resize((w, h), Image.Resampling.NEAREST)) > 127
    return SemanticMasks(r(masks.face), r(masks.skin), r(masks.background))
