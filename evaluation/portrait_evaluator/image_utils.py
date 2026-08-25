from __future__ import annotations

from pathlib import Path
import math
import numpy as np

try:
    import cv2
except ImportError as exc:
    raise RuntimeError("portrait_evaluator requires opencv-python or opencv-python-headless") from exc

from portrait_evaluator.models import ImageFeatures


def srgb_to_linear(rgb01: np.ndarray) -> np.ndarray:
    rgb01 = np.asarray(rgb01, dtype=np.float32)
    return np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)


def luminance_from_rgb(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb.astype(np.float32) / 255.0)
    return (0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb.astype(np.float32) / 255.0)
    x = 0.4124564 * linear[..., 0] + 0.3575761 * linear[..., 1] + 0.1804375 * linear[..., 2]
    y = 0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1] + 0.0721750 * linear[..., 2]
    z = 0.0193339 * linear[..., 0] + 0.1191920 * linear[..., 1] + 0.9503041 * linear[..., 2]
    xyz = np.stack([x / 0.95047, y / 1.0, z / 1.08883], axis=-1)
    delta = 6.0 / 29.0
    threshold = delta ** 3
    f = np.where(xyz > threshold, np.cbrt(xyz), xyz / (3 * delta * delta) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def load_image(path: Path) -> ImageFeatures:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    return features_from_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def features_from_rgb(rgb: np.ndarray) -> ImageFeatures:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected RGB HxWx3 image")
    rgb = np.asarray(rgb, dtype=np.uint8)
    return ImageFeatures(rgb=rgb, luminance=luminance_from_rgb(rgb), lab=rgb_to_lab(rgb))


def robust_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float("nan") if values.size == 0 else float(np.percentile(values, q))


def robust_median(values: np.ndarray) -> float:
    return robust_percentile(values, 50)


def delta_e00(lab1: np.ndarray, lab2: np.ndarray) -> float:
    L1, a1, b1 = map(float, lab1); L2, a2, b2 = map(float, lab2)
    C1 = math.hypot(a1, b1); C2 = math.hypot(a2, b2); Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt((Cbar ** 7) / (Cbar ** 7 + 25 ** 7))) if Cbar > 0 else 0.5
    ap1 = (1 + G) * a1; ap2 = (1 + G) * a2
    Cp1 = math.hypot(ap1, b1); Cp2 = math.hypot(ap2, b2)
    hp1 = math.degrees(math.atan2(b1, ap1)) % 360; hp2 = math.degrees(math.atan2(b2, ap2)) % 360
    dLp = L2 - L1; dCp = Cp2 - Cp1; dh = hp2 - hp1
    if Cp1 * Cp2 == 0: dhp = 0.0
    elif abs(dh) <= 180: dhp = dh
    elif dh > 180: dhp = dh - 360
    else: dhp = dh + 360
    dHp = 2 * math.sqrt(Cp1 * Cp2) * math.sin(math.radians(dhp / 2))
    Lbarp = (L1 + L2) / 2; Cbarp = (Cp1 + Cp2) / 2
    if Cp1 * Cp2 == 0: hbarp = hp1 + hp2
    elif abs(hp1 - hp2) <= 180: hbarp = (hp1 + hp2) / 2
    elif hp1 + hp2 < 360: hbarp = (hp1 + hp2 + 360) / 2
    else: hbarp = (hp1 + hp2 - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbarp - 30)) + 0.24 * math.cos(math.radians(2 * hbarp)) + 0.32 * math.cos(math.radians(3 * hbarp + 6)) - 0.20 * math.cos(math.radians(4 * hbarp - 63)))
    dtheta = 30 * math.exp(-((hbarp - 275) / 25) ** 2)
    Rc = 2 * math.sqrt((Cbarp ** 7) / (Cbarp ** 7 + 25 ** 7)) if Cbarp > 0 else 0.0
    Sl = 1 + 0.015 * ((Lbarp - 50) ** 2) / math.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp; Sh = 1 + 0.015 * Cbarp * T; Rt = -math.sin(math.radians(2 * dtheta)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def hue_degrees(a: float, b: float) -> float:
    return math.degrees(math.atan2(b, a)) % 360


def circular_distance_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def resize_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]; scale = min(1.0, float(max_side) / max(h, w))
    if scale == 1.0: return rgb
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
