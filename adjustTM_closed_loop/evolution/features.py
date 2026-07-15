from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FEATURE_NAMES = (
    "log_luma_mean", "luma_p01", "luma_p05", "luma_p10", "luma_p25", "luma_p50", "luma_p75",
    "luma_p90", "luma_p95", "luma_p99", "shadow_ratio", "highlight_ratio", "robust_contrast",
    "mean_r", "mean_g", "mean_b", "chroma_rg_std", "chroma_bg_std", "gradient_mean",
)


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected readable three-channel image: {path}")
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"Unsupported image dtype {image.dtype}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / scale


def extract_rgb_features(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("rgb must be finite HxWx3")
    rgb = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    percentiles = np.percentile(luma, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    total = rgb.sum(axis=-1) + 1e-8
    rg = rgb[..., 0] / total
    bg = rgb[..., 2] / total
    gy, gx = np.gradient(luma)
    values = [
        float(np.mean(np.log2(luma + 1e-6))), *[float(v) for v in percentiles],
        float(np.mean(luma <= 0.01)), float(np.mean(luma >= 0.99)), float(percentiles[6] - percentiles[2]),
        float(rgb[..., 0].mean()), float(rgb[..., 1].mean()), float(rgb[..., 2].mean()),
        float(rg.std()), float(bg.std()), float(np.sqrt(gx * gx + gy * gy).mean()),
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise FloatingPointError("Feature extraction produced invalid values")
    return result


def extract_image_features(path: str | Path) -> np.ndarray:
    return extract_rgb_features(read_rgb(path))
