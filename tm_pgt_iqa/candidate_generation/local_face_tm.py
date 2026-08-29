from __future__ import annotations

import numpy as np

from ..config import LocalFaceTMConfig
from ..segmentation import SemanticMasks
from .retinex import linear_luminance, scale_luminance, srgb_to_linear


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    padded = np.pad(mask.astype(bool), radius)
    h, w = mask.shape
    output = np.zeros_like(mask, dtype=bool)
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            output |= padded[y:y + h, x:x + w]
    return output


def _smooth(image: np.ndarray, radius: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=max(radius, 0.1), mode="reflect")
    except ImportError:  # pragma: no cover
        return image


def face_adjustment_map(masks: SemanticMasks, cfg: LocalFaceTMConfig) -> np.ndarray:
    """Soft, human-bounded face adjustment map with no hard background edge."""
    human_support = masks.face | masks.human
    expanded = _dilate(masks.face, cfg.dilation_radius) & human_support
    seed = np.where(expanded, masks.soft_face, 0.0)
    # Include an expanded low-valued support before smoothing for a natural rim.
    seed = np.maximum(seed, expanded.astype(np.float32) * 0.15)
    smooth = _smooth(seed, cfg.smoothing_radius)
    smooth *= human_support.astype(np.float32)
    peak = float(smooth[masks.face_core].max(initial=0.0))
    if peak > 0:
        smooth = smooth / peak
    return np.clip(smooth, 0.0, 1.0).astype(np.float32)


def generate_local_face_tm(rgb: np.ndarray, masks: SemanticMasks, cfg: LocalFaceTMConfig) -> dict[str, np.ndarray]:
    adjustment = face_adjustment_map(masks, cfg)
    luminance = linear_luminance(srgb_to_linear(rgb))
    result: dict[str, np.ndarray] = {}
    for level, ev in cfg.lift_ev.items():
        target = luminance * (2.0 ** (adjustment * float(ev)))
        result[f"face_lift_{level}"] = scale_luminance(rgb, target)
    return result
