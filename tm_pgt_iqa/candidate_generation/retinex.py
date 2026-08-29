from __future__ import annotations

import numpy as np

from ..config import RetinexConfig

_EPS = 1e-6


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(np.asarray(rgb, dtype=np.float32), 0.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def linear_luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def hue_preserving_gamut_compress(linear_rgb: np.ndarray) -> np.ndarray:
    """Compress out-of-gamut RGB together, preserving chromatic direction."""
    linear_rgb = np.maximum(linear_rgb, 0.0)
    peak = linear_rgb.max(axis=-1, keepdims=True)
    scale = np.where(peak > 1.0, 1.0 / peak, 1.0)
    return linear_rgb * scale


def scale_luminance(rgb: np.ndarray, target_luminance: np.ndarray) -> np.ndarray:
    """Apply a luminance target by scaling linear RGB, not RGB channels independently."""
    linear = srgb_to_linear(rgb)
    current = linear_luminance(linear)
    scale = np.asarray(target_luminance, dtype=np.float32) / np.maximum(current, _EPS)
    transformed = hue_preserving_gamut_compress(linear * scale[..., None])
    return np.clip(linear_to_srgb(transformed), 0.0, 1.0).astype(np.float32)


def _gaussian_blur(image: np.ndarray, radius: float) -> np.ndarray:
    # The exact blur implementation is intentionally isolated: illumination is
    # an audit-only Retinex component and reconstruction remains deterministic.
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=max(float(radius) / 6.0, 0.1), mode="reflect")
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return image.copy()


def retinex_illumination(rgb: np.ndarray, cfg: RetinexConfig) -> tuple[np.ndarray, np.ndarray]:
    linear = srgb_to_linear(rgb)
    illumination = _gaussian_blur(linear_luminance(linear), cfg.gaussian_radius)
    reflectance = linear / np.maximum(illumination[..., None], cfg.eps)
    return reflectance, illumination


def generate_retinex(rgb: np.ndarray, cfg: RetinexConfig) -> dict[str, np.ndarray]:
    """Generate the ordered, deterministic primary illumination search axis."""
    reflectance, illumination = retinex_illumination(rgb, cfg)
    result: dict[str, np.ndarray] = {}
    for candidate_id, ev in cfg.levels_ev.items():
        target_illumination = illumination * (2.0 ** float(ev))
        rebuilt = reflectance * target_illumination[..., None]
        rebuilt = hue_preserving_gamut_compress(rebuilt)
        result[candidate_id] = np.clip(linear_to_srgb(rebuilt), 0.0, 1.0).astype(np.float32)
    return result
