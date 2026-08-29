from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
from PIL import Image

from .config import GuardConfig, ObjectiveMetricConfig, ScoreWeights
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
    halo_inner_luminance: float
    halo_outer_luminance: float
    halo_face_luminance: float
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
    warnings: list[str] | None = None

    def to_dict(self) -> dict:
        return {"passed": self.passed, "failures": self.failures, "warnings": self.warnings or []}


@dataclass
class SourcePreservation:
    """Objective evidence that an edit still looks like tone mapping.

    This is deliberately conservative: only unmistakable structural evidence
    produces ``FAIL``.  Hue changes, a weak edge mismatch, and other ambiguous
    evidence remain ``SUSPICIOUS`` for the semantic judge to resolve.
    """

    status: str
    structural_failure: bool
    low_frequency_error: float
    edge_position_agreement: float
    face_structure_correlation: float | None
    hue_shift_degrees: dict[str, float]
    issues: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def extract_features(
    rgb: np.ndarray,
    masks: SemanticMasks,
    guards: GuardConfig | None = None,
) -> TMFeatures:
    """Extract objective evidence using configurable clip/dark thresholds.

    ``guards`` is optional to retain the V1 two-argument public call form.
    """
    guards = guards or GuardConfig()
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

    # A face/background contrast is not a halo.  A halo is a local overshoot at
    # the boundary, therefore compare the inner ring against both neighbours.
    face_interior = masks.face_core if masks.face_core.any() else masks.face
    inner_ring = masks.face_inner_ring
    outer_ring = masks.face_outer_ring
    y_face = float(np.median(y[face_interior]))
    y_inner = float(np.median(y[inner_ring])) if inner_ring.any() else y_face
    y_outer = float(np.median(y[outer_ring])) if outer_ring.any() else y_inner
    bright_overshoot = max(0.0, y_inner - max(y_face, y_outer))
    dark_overshoot = max(0.0, min(y_face, y_outer) - y_inner)
    halo_strength = float(max(bright_overshoot, dark_overshoot))

    lab = _rgb_to_lab(rgb)
    skin_ab = lab[masks.skin][:, 1:3]
    med_ab = np.median(skin_ab, axis=0)
    chroma = float(np.linalg.norm(med_ab))
    hue = float(np.degrees(np.arctan2(med_ab[1], med_ab[0])))

    return TMFeatures(
        float(f10), float(f25), float(f50), float(f75), float(f90), b50,
        float(g01), float(g10), float(g25), float(g50), float(g75), float(g90), float(g99),
        float(np.mean(fy > guards.face_highlight_threshold)),
        float(np.mean(fy < guards.face_dark_threshold)),
        float(np.mean(gy > guards.face_highlight_threshold)),
        float(math.log2((f50 + EPS) / (b50 + EPS))),
        float(lf90 - lf10), float(lf50 - lf10), float(lf90 - lf50),
        std, skew, halo_strength, y_inner, y_outer, y_face, chroma, hue,
    )


def score_features(
    f: TMFeatures,
    weights: ScoreWeights,
    objective: ObjectiveMetricConfig | None = None,
) -> TMQuality:
    objective = objective or ObjectiveMetricConfig()
    face_ev = math.log2(f.face_y50 + EPS)
    exposure = _band_score(face_ev, *objective.exposure_band)

    highlight_headroom = f.global_p99 - f.global_p90
    shadow_sep = f.global_p10 - f.global_p01
    mid_span = f.global_p75 - f.global_p25
    dr_weights = objective.dynamic_range_components
    dr = (
        dr_weights[0] * _band_score(highlight_headroom, *objective.highlight_headroom_band)
        + dr_weights[1] * _band_score(shadow_sep, *objective.shadow_separation_band)
        + dr_weights[2] * _band_score(mid_span, *objective.midtone_span_band)
    )
    dr *= max(0.0, 1.0 - min(1.0, f.global_clip_ratio / objective.dynamic_range_clip_scale))

    tone_weights = objective.face_tone_components
    face_tone = (
        tone_weights[0] * _band_score(f.face_tone_span_ev, *objective.face_tone_span_band)
        + tone_weights[1] * _band_score(f.face_shadow_sep_ev, *objective.face_shadow_band)
        + tone_weights[2] * _band_score(f.face_highlight_sep_ev, *objective.face_highlight_band)
    )

    fb = _band_score(f.face_bg_ev, *objective.face_background_band)

    natural_weights = objective.naturalness_components
    natural = (
        natural_weights[0] * _band_score(f.global_log_std, *objective.global_log_std_band)
        + natural_weights[1] * _gaussian_score(
            f.global_log_skew,
            center=objective.global_log_skew_center,
            sigma=objective.global_log_skew_sigma,
        )
    )

    overall = (
        weights.exposure * exposure
        + weights.dynamic_range * dr
        + weights.face_tone * face_tone
        + weights.face_background * fb
        + weights.naturalness * natural
    )
    return TMQuality(float(exposure), float(dr), float(face_tone), float(fb), float(natural), float(overall))


def evaluate_guards(
    f: TMFeatures,
    cfg: GuardConfig,
    preservation: SourcePreservation | None = None,
) -> GuardResult:
    failures: list[str] = []
    warnings: list[str] = []
    if f.face_clip_ratio > cfg.face_clip_reject:
        failures.append("FACE_HIGHLIGHT_CLIP")
    if f.face_dark_ratio > cfg.face_dark_reject:
        failures.append("FACE_SEVERE_UNDEREXPOSURE")
    if f.global_clip_ratio > cfg.global_clip_reject:
        failures.append("GLOBAL_CATASTROPHIC_CLIP")
    halo_warning = cfg.halo_warning if cfg.halo_warning is not None else cfg.halo_reject
    if f.halo_strength > halo_warning:
        warnings.append("LOCAL_TM_HALO")
    if not (cfg.skin_chroma_min <= f.skin_chroma <= cfg.skin_chroma_max):
        warnings.append("SKIN_COLOR_ABNORMAL")
    if preservation is not None and preservation.structural_failure:
        failures.append("SOURCE_PRESERVATION_FAIL")
    return GuardResult(not failures, failures, warnings)


def _resize_rgb(rgb: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == size_hw:
        return rgb
    h, w = size_hw
    image = Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0), mode="RGB")
    return np.asarray(image.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _normalised_log_luminance(rgb: np.ndarray) -> np.ndarray:
    value = np.log2(np.clip(luminance(rgb), EPS, None))
    return (value - float(value.mean())) / max(float(value.std()), EPS)


def _low_frequency(value: np.ndarray) -> np.ndarray:
    """Blur via a small down/up sample without adding a SciPy dependency."""
    h, w = value.shape
    small_w, small_h = max(2, w // 8), max(2, h // 8)
    image = Image.fromarray(value.astype(np.float32), mode="F")
    small = image.resize((small_w, small_h), Image.Resampling.BILINEAR)
    return np.asarray(small.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)


def _edge_map(value: np.ndarray, percentile: float) -> np.ndarray:
    gy, gx = np.gradient(value)
    magnitude = np.hypot(gx, gy)
    threshold = float(np.percentile(magnitude, percentile))
    if threshold <= EPS:
        return np.zeros_like(value, dtype=bool)
    return magnitude >= threshold


def _edge_agreement(source_edges: np.ndarray, candidate_edges: np.ndarray) -> float:
    if not source_edges.any() and not candidate_edges.any():
        return 1.0
    # One-pixel tolerance makes this check robust to ordinary resampling while
    # still detecting moved/repainted geometry.
    source_near = _binary_dilate(source_edges, radius=1)
    candidate_near = _binary_dilate(candidate_edges, radius=1)
    matched = int(np.count_nonzero(candidate_edges & source_near)) + int(np.count_nonzero(source_edges & candidate_near))
    total = int(np.count_nonzero(source_edges) + np.count_nonzero(candidate_edges))
    return float(matched / max(total, 1))


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 16 or b.size < 16:
        return None
    a_std, b_std = float(a.std()), float(b.std())
    if a_std < EPS or b_std < EPS:
        return None
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def _regional_hue_shift(source: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    source_ab = np.median(_rgb_to_lab(source)[mask][:, 1:3], axis=0)
    candidate_ab = np.median(_rgb_to_lab(candidate)[mask][:, 1:3], axis=0)
    source_angle = math.degrees(math.atan2(float(source_ab[1]), float(source_ab[0])))
    candidate_angle = math.degrees(math.atan2(float(candidate_ab[1]), float(candidate_ab[0])))
    return float(abs((candidate_angle - source_angle + 180.0) % 360.0 - 180.0))


def evaluate_source_preservation(
    source_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    masks: SemanticMasks,
    cfg: GuardConfig,
) -> SourcePreservation:
    """Assess whether source/candidate differences remain plausibly TM-only."""
    candidate_rgb = np.asarray(candidate_rgb, dtype=np.float32)
    source_rgb = _resize_rgb(np.asarray(source_rgb, dtype=np.float32), candidate_rgb.shape[:2])
    source_log = _normalised_log_luminance(source_rgb)
    candidate_log = _normalised_log_luminance(candidate_rgb)
    low_frequency_error = float(np.mean(np.abs(_low_frequency(source_log) - _low_frequency(candidate_log))))
    edge_agreement = _edge_agreement(
        _edge_map(source_log, cfg.preservation_edge_percentile),
        _edge_map(candidate_log, cfg.preservation_edge_percentile),
    )
    face_support = masks.face_core if masks.face_core.any() else masks.face
    face_correlation = _correlation(source_log[face_support], candidate_log[face_support])
    hue_shift_degrees = {
        "skin": _regional_hue_shift(source_rgb, candidate_rgb, masks.skin),
        "face": _regional_hue_shift(source_rgb, candidate_rgb, masks.face),
        "background": _regional_hue_shift(source_rgb, candidate_rgb, masks.background),
    }

    low_frequency_fail = low_frequency_error > cfg.preservation_low_frequency_fail
    edge_failure = edge_agreement < cfg.preservation_edge_agreement_fail
    face_structure_failure = (
        face_correlation is not None
        and face_correlation < cfg.preservation_face_correlation_fail
    )
    # Illumination edits can legitimately alter broad low-frequency luma.  Do
    # not turn that signal into a hard failure unless a separate geometry/edge
    # proxy agrees that source structure was changed.
    structural_failure = (
        (low_frequency_fail and (edge_failure or face_structure_failure))
        or (edge_failure and face_structure_failure)
    )
    structural_suspicious = (
        low_frequency_error > cfg.preservation_low_frequency_suspicious
        or edge_agreement < cfg.preservation_edge_agreement_suspicious
        or (
            face_correlation is not None
            and face_correlation < cfg.preservation_face_correlation_suspicious
        )
    )
    max_hue_shift = max(hue_shift_degrees.values(), default=0.0)
    issues: list[str] = []
    if structural_failure:
        issues.append("STRUCTURE_CHANGED")
    elif structural_suspicious:
        issues.append("STRUCTURE_SUSPICIOUS")
    if max_hue_shift > cfg.preservation_hue_shift_fail_deg:
        issues.append("LARGE_HUE_SHIFT")
    elif max_hue_shift > cfg.preservation_hue_shift_suspicious_deg:
        issues.append("HUE_SHIFT_SUSPICIOUS")
    status = "FAIL" if structural_failure else ("SUSPICIOUS" if issues else "PASS")
    return SourcePreservation(
        status=status,
        structural_failure=structural_failure,
        low_frequency_error=low_frequency_error,
        edge_position_agreement=edge_agreement,
        face_structure_correlation=face_correlation,
        hue_shift_degrees=hue_shift_degrees,
        issues=issues,
    )
