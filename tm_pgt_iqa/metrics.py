from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
from PIL import Image

from .config import GuardConfig, ScoreWeights
from .segmentation import SemanticMasks

EPS = 1e-6


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def luminance(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _pct(x: np.ndarray, qs: list[float]) -> np.ndarray:
    return np.percentile(x, qs).astype(np.float64)


def _band_score(value: float, low_bad: float, low_good: float, high_good: float, high_bad: float) -> float:
    if low_good <= value <= high_good:
        return 100.0
    if value <= low_bad or value >= high_bad:
        return 0.0
    if value < low_good:
        return 100.0 * (value - low_bad) / (low_good - low_bad)
    return 100.0 * (high_bad - value) / (high_bad - high_good)


def _gaussian_score(value: float, center: float, sigma: float) -> float:
    z = (value - center) / max(sigma, EPS)
    return float(100.0 * math.exp(-0.5 * z * z))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    x = 0.4124564 * lin[..., 0] + 0.3575761 * lin[..., 1] + 0.1804375 * lin[..., 2]
    y = 0.2126729 * lin[..., 0] + 0.7151522 * lin[..., 1] + 0.0721750 * lin[..., 2]
    z = 0.0193339 * lin[..., 0] + 0.1191920 * lin[..., 1] + 0.9503041 * lin[..., 2]
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883
    d = 6 / 29
    def f(t):
        return np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def _binary_dilate(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), radius)
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy+h, dx:dx+w] > 0
    return out


@dataclass
class TMFeatures:
    face_y10: float
    face_y25: float
    face_y50: float
    face_y75: float
    face_y90: float
    bg_y50: float
    global_p01: float
    global_p10: float
    global_p25: float
    global_p50: float
    global_p75: float
    global_p90: float
    global_p99: float
    face_clip_ratio: float
    face_dark_ratio: float
    global_clip_ratio: float
    face_bg_ev: float
    face_tone_span_ev: float
    face_shadow_sep_ev: float
    face_highlight_sep_ev: float
    global_log_std: float
    global_log_skew: float
    halo_strength: float
    skin_chroma: float
    skin_hue_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TMQuality:
    exposure: float
    dynamic_range: float
    face_tone: float
    face_background: float
    naturalness: float
    overall: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GuardResult:
    passed: bool
    failures: list[str]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "failures": self.failures}


def extract_features(rgb: np.ndarray, masks: SemanticMasks) -> TMFeatures:
    y = luminance(rgb)
    fy = y[masks.face]
    by = y[masks.background]
    gy = y.reshape(-1)
    f10, f25, f50, f75, f90 = _pct(fy, [10, 25, 50, 75, 90])
    g01, g10, g25, g50, g75, g90, g99 = _pct(gy, [1, 10, 25, 50, 75, 90, 99])
    b50 = float(np.median(by))
    logf = np.log2(np.clip(fy, EPS, None))
    lf10, lf50, lf90 = _pct(logf, [10, 50, 90])
    logg = np.log2(np.clip(gy, EPS, None))
    mean = float(logg.mean())
    std = float(logg.std())
    skew = float(np.mean(((logg - mean) / max(std, EPS)) ** 3))

    ring = _binary_dilate(masks.face, radius=3) & ~masks.face
    ring &= ~masks.skin
    if ring.any():
        boundary_gap = abs(float(np.median(fy)) - float(np.median(y[ring])))
        halo_strength = boundary_gap
    else:
        halo_strength = 0.0

    lab = _rgb_to_lab(rgb)
    skin_ab = lab[masks.skin][:, 1:3]
    med_ab = np.median(skin_ab, axis=0)
    chroma = float(np.linalg.norm(med_ab))
    hue = float(np.degrees(np.arctan2(med_ab[1], med_ab[0])))

    return TMFeatures(
        float(f10), float(f25), float(f50), float(f75), float(f90), b50,
        float(g01), float(g10), float(g25), float(g50), float(g75), float(g90), float(g99),
        float(np.mean(fy > 0.98)), float(np.mean(fy < 0.02)), float(np.mean(gy > 0.98)),
        float(math.log2((f50 + EPS) / (b50 + EPS))),
        float(lf90 - lf10), float(lf50 - lf10), float(lf90 - lf50),
        std, skew, halo_strength, chroma, hue,
    )


def score_features(f: TMFeatures, weights: ScoreWeights) -> TMQuality:
    face_ev = math.log2(f.face_y50 + EPS)
    exposure = _band_score(face_ev, -5.0, -3.1, -1.4, -0.35)

    highlight_headroom = f.global_p99 - f.global_p90
    shadow_sep = f.global_p10 - f.global_p01
    mid_span = f.global_p75 - f.global_p25
    dr = (
        0.35 * _band_score(highlight_headroom, 0.0, 0.015, 0.18, 0.35)
        + 0.25 * _band_score(shadow_sep, 0.0, 0.005, 0.12, 0.25)
        + 0.40 * _band_score(mid_span, 0.02, 0.08, 0.45, 0.70)
    )
    dr *= max(0.0, 1.0 - min(1.0, f.global_clip_ratio / 0.20))

    face_tone = (
        0.45 * _band_score(f.face_tone_span_ev, 0.15, 0.55, 2.8, 4.5)
        + 0.275 * _band_score(f.face_shadow_sep_ev, 0.05, 0.20, 1.5, 2.5)
        + 0.275 * _band_score(f.face_highlight_sep_ev, 0.05, 0.20, 1.5, 2.5)
    )

    fb = _band_score(f.face_bg_ev, -3.0, -1.2, 2.0, 3.5)

    natural = (
        0.60 * _band_score(f.global_log_std, 0.15, 0.45, 2.0, 3.2)
        + 0.40 * _gaussian_score(f.global_log_skew, center=-0.2, sigma=1.6)
    )

    overall = (
        weights.exposure * exposure
        + weights.dynamic_range * dr
        + weights.face_tone * face_tone
        + weights.face_background * fb
        + weights.naturalness * natural
    )
    return TMQuality(float(exposure), float(dr), float(face_tone), float(fb), float(natural), float(overall))


def evaluate_guards(f: TMFeatures, cfg: GuardConfig) -> GuardResult:
    failures: list[str] = []
    if f.face_clip_ratio > cfg.face_clip_reject:
        failures.append("FACE_HIGHLIGHT_CLIP")
    if f.face_dark_ratio > cfg.face_dark_reject:
        failures.append("FACE_UNDEREXPOSURE")
    if f.global_clip_ratio > cfg.global_clip_reject:
        failures.append("GLOBAL_HIGHLIGHT_CLIP")
    if f.halo_strength > cfg.halo_reject:
        failures.append("LOCAL_TM_HALO")
    if not (cfg.skin_chroma_min <= f.skin_chroma <= cfg.skin_chroma_max):
        failures.append("SKIN_COLOR_ABNORMAL")
    return GuardResult(not failures, failures)
