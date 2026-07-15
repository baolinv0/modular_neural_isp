from __future__ import annotations

from collections.abc import Iterable

from .schemas import FailureCode


_REASON_MAP: tuple[tuple[tuple[str, ...], FailureCode], ...] = (
    (("underexposure", "brightness_low", "too_dark"), FailureCode.BRIGHTNESS_UNDER),
    (("overexposure", "brightness_high", "too_bright"), FailureCode.BRIGHTNESS_OVER),
    (("clip", "highlight"), FailureCode.HIGHLIGHT_CLIPPING),
    (("shadow", "black_crush", "deep_shadow"), FailureCode.SHADOW_CRUSH),
    (("color", "chroma", "chromaticity", "white_balance"), FailureCode.CHROMA_DRIFT),
    (("monotonic", "dead_zone", "endpoint_range", "saturation_score", "smooth"), FailureCode.CONTROL_CURVE),
    (("regional", "coherence", "semantic"), FailureCode.REGIONAL_INCONSISTENCY),
    (("edge", "ssim", "halo", "banding", "structure", "geometry", "alignment"), FailureCode.STRUCTURAL_ARTIFACT),
)


def classify_failure_reasons(reasons: Iterable[str]) -> set[FailureCode]:
    codes: set[FailureCode] = set()
    for reason in reasons:
        normalized = str(reason).lower()
        matched = False
        for keywords, code in _REASON_MAP:
            if any(keyword in normalized for keyword in keywords):
                codes.add(code)
                matched = True
                break
        if not matched:
            codes.add(FailureCode.STRUCTURAL_ARTIFACT)
    return codes
