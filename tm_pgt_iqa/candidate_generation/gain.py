from __future__ import annotations

import numpy as np

from ..config import GainConfig
from .retinex import hue_preserving_gamut_compress, linear_to_srgb, srgb_to_linear


def generate_gain_max(rgb: np.ndarray, cfg: GainConfig) -> np.ndarray:
    """Maximum global exposure gain satisfying the configured pre-compression clip budget."""
    linear = srgb_to_linear(rgb)
    low, high = 1.0, max(float(cfg.max_gain), 1.0)
    for _ in range(max(int(cfg.search_steps), 1)):
        mid = (low + high) / 2.0
        clip_ratio = float(np.mean(np.max(linear * mid, axis=-1) > 1.0))
        if clip_ratio <= cfg.max_clip_ratio:
            low = mid
        else:
            high = mid
    return np.clip(linear_to_srgb(hue_preserving_gamut_compress(linear * low)), 0.0, 1.0).astype(np.float32)
