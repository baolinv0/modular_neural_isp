from __future__ import annotations

import numpy as np

from ..config import ToneShapeConfig
from ..segmentation import SemanticMasks
from .local_face_tm import _smooth
from .retinex import linear_luminance, scale_luminance, srgb_to_linear

_EPS = 1e-6


def _shape_target(luminance: np.ndarray, face: np.ndarray, gamma: float) -> np.ndarray:
    median = float(np.median(luminance[face]))
    shaped = median * np.power(np.maximum(luminance, _EPS) / max(median, _EPS), gamma)
    # Re-normalise over face pixels to preserve median exposure in linear space.
    shaped *= median / max(float(np.median(shaped[face])), _EPS)
    return shaped


def _face_weight(masks: SemanticMasks, smoothing_radius: float) -> np.ndarray:
    """Face-first soft support, optionally feathered through adjacent body."""
    weight = _smooth(masks.soft_face, smoothing_radius)
    human_support = masks.face | masks.human
    return np.clip(weight * human_support, 0.0, 1.0)


def generate_tone_shape(rgb: np.ndarray, masks: SemanticMasks, cfg: ToneShapeConfig) -> dict[str, np.ndarray]:
    """Alter face shadow/highlight distribution while retaining face median EV."""
    luminance = linear_luminance(srgb_to_linear(rgb))
    weight = _face_weight(masks, cfg.smoothing_radius)
    result: dict[str, np.ndarray] = {}
    for candidate_id, gamma in (
        ("tone_shadow_preserve", cfg.shadow_preserve_gamma),
        ("tone_soft_highlight", cfg.soft_highlight_gamma),
    ):
        shaped = _shape_target(luminance, masks.face, float(gamma))
        target = luminance * (1.0 - weight) + shaped * weight
        # One correction after the soft transition preserves the specified face
        # median without affecting the background.
        before = float(np.median(luminance[masks.face]))
        after = float(np.median(target[masks.face]))
        target[masks.face] *= before / max(after, _EPS)
        result[candidate_id] = scale_luminance(rgb, target)
    return result
