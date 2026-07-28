"""Position-independent luminance and chroma metrics for non-aligned images."""
from __future__ import annotations

import math
from typing import Mapping, Optional

import cv2
import numpy as np


_EPS = 1e-6
_LUMA_QUANTILES = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
_WASSERSTEIN_QUANTILES = np.linspace(0.0, 1.0, 257)
_SEMANTIC_NAMES = ("skin", "sky", "vegetation")


def _validate_rgb(image: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{label} must have shape [H, W, 3], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.clip(array, 0.0, 1.0)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = _validate_rgb(image, "sRGB image")
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = _validate_rgb(image, "linear image")
    return np.where(image <= 0.0031308, 12.92 * image, 1.055 * np.power(image, 1 / 2.4) - 0.055).astype(np.float32)


def _mask_or_all(mask: Optional[np.ndarray], shape: tuple[int, int], label: str) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{label} shape {result.shape} does not match image shape {shape}")
    if not np.any(result):
        raise ValueError(f"{label} selects no pixels")
    return result


def _linear_luma(image: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(image)
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


def _quantile(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.full(probabilities.shape, np.nan, dtype=np.float64)
    return np.quantile(finite, probabilities)


def _wasserstein_1d(first: np.ndarray, second: np.ndarray) -> float:
    first_q = _quantile(first, _WASSERSTEIN_QUANTILES)
    second_q = _quantile(second, _WASSERSTEIN_QUANTILES)
    if not np.isfinite(first_q).all() or not np.isfinite(second_q).all():
        return float("nan")
    return float(np.mean(np.abs(first_q - second_q)))


def _subsample_rows(values: np.ndarray, max_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] <= max_points:
        return values
    indices = np.linspace(0, values.shape[0] - 1, max_points, dtype=np.int64)
    return values[indices]


def _sliced_wasserstein_2d(
    first: np.ndarray,
    second: np.ndarray,
    *,
    directions: int = 32,
    max_points: int = 16384,
) -> float:
    first = _subsample_rows(first, max_points)
    second = _subsample_rows(second, max_points)
    if first.shape[0] < 2 or second.shape[0] < 2:
        return float("nan")
    angles = np.linspace(0.0, math.pi, directions, endpoint=False, dtype=np.float64)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    distances = []
    for direction in unit:
        distances.append(_wasserstein_1d(first @ direction, second @ direction))
    return float(np.nanmean(distances))


def _rgb_to_cbcr(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = srgb_to_linear(image)
    y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    cb = -0.114572 * linear[..., 0] - 0.385428 * linear[..., 1] + 0.5 * linear[..., 2]
    cr = 0.5 * linear[..., 0] - 0.454153 * linear[..., 1] - 0.045847 * linear[..., 2]
    return y.astype(np.float32), np.stack([cb, cr], axis=-1).astype(np.float32)


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_validate_rgb(image, "RGB image").astype(np.float32), cv2.COLOR_RGB2LAB)


def compute_luminance_metrics(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    output_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Computes non-aligned global exposure and tone-shape distances."""

    output = _validate_rgb(output, "output")
    reference = _validate_rgb(reference, "reference")
    output_valid = _mask_or_all(output_mask, output.shape[:2], "output_mask")
    reference_valid = _mask_or_all(reference_mask, reference.shape[:2], "reference_mask")
    output_y = _linear_luma(output)[output_valid]
    reference_y = _linear_luma(reference)[reference_valid]
    output_log = np.log2(output_y + _EPS)
    reference_log = np.log2(reference_y + _EPS)
    output_q = _quantile(output_log, _LUMA_QUANTILES)
    reference_q = _quantile(reference_log, _LUMA_QUANTILES)
    signed_ev = float(output_q[4] - reference_q[4])
    tone_output = output_q - output_q[4]
    tone_reference = reference_q - reference_q[4]

    output_shadow = float(np.mean(output_y < 0.03))
    reference_shadow = float(np.mean(reference_y < 0.03))
    output_highlight = float(np.mean(output_y > 0.95))
    reference_highlight = float(np.mean(reference_y > 0.95))
    output_pixels = output[output_valid]
    reference_pixels = reference[reference_valid]
    output_clipping = float(np.mean(np.any((output_pixels <= 0.001) | (output_pixels >= 0.999), axis=1)))
    reference_clipping = float(
        np.mean(np.any((reference_pixels <= 0.001) | (reference_pixels >= 0.999), axis=1))
    )
    return {
        "signed_ev_error": signed_ev,
        "absolute_ev_error": abs(signed_ev),
        "log_luma_quantile_mae": float(np.mean(np.abs(output_q - reference_q))),
        "tone_shape_mae": float(np.mean(np.abs(tone_output - tone_reference))),
        "log_luma_w1": _wasserstein_1d(output_log, reference_log),
        "shadow_ratio_error": abs(output_shadow - reference_shadow),
        "highlight_ratio_error": abs(output_highlight - reference_highlight),
        "clipping_ratio_error": abs(output_clipping - reference_clipping),
    }


def _masked_chroma(cbcr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cbcr[mask].reshape(-1, 2)


def _conditioned_chroma_swd(
    output_y: np.ndarray,
    output_cbcr: np.ndarray,
    reference_y: np.ndarray,
    reference_cbcr: np.ndarray,
    output_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> float:
    bins = ((0.0, 0.18), (0.18, 0.72), (0.72, 1.01))
    values = []
    for lower, upper in bins:
        out_band = output_mask & (output_y >= lower) & (output_y < upper)
        ref_band = reference_mask & (reference_y >= lower) & (reference_y < upper)
        if np.count_nonzero(out_band) < 8 or np.count_nonzero(ref_band) < 8:
            continue
        values.append(_sliced_wasserstein_2d(_masked_chroma(output_cbcr, out_band), _masked_chroma(reference_cbcr, ref_band)))
    return float(np.mean(values)) if values else float("nan")


def _covariance_error(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape[0] < 2 or second.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(np.cov(first, rowvar=False) - np.cov(second, rowvar=False), ord="fro"))


def _neutral_axis_error(output_lab: np.ndarray, reference_lab: np.ndarray) -> float:
    output_c = np.linalg.norm(output_lab[:, 1:3], axis=1)
    reference_c = np.linalg.norm(reference_lab[:, 1:3], axis=1)
    output_neutral = output_lab[output_c <= 12.0, 1:3]
    reference_neutral = reference_lab[reference_c <= 12.0, 1:3]
    if output_neutral.shape[0] < 8 or reference_neutral.shape[0] < 8:
        return float("nan")
    return float(np.linalg.norm(output_neutral.mean(axis=0) - reference_neutral.mean(axis=0)))


def compute_color_metrics(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    output_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    input_semantic_masks: Optional[Mapping[str, np.ndarray]] = None,
    reference_semantic_masks: Optional[Mapping[str, np.ndarray]] = None,
) -> dict[str, float]:
    """Computes chroma distances without matching pixel positions."""

    output = _validate_rgb(output, "output")
    reference = _validate_rgb(reference, "reference")
    output_valid = _mask_or_all(output_mask, output.shape[:2], "output_mask")
    reference_valid = _mask_or_all(reference_mask, reference.shape[:2], "reference_mask")
    output_y, output_cbcr_image = _rgb_to_cbcr(output)
    reference_y, reference_cbcr_image = _rgb_to_cbcr(reference)
    output_cbcr = _masked_chroma(output_cbcr_image, output_valid)
    reference_cbcr = _masked_chroma(reference_cbcr_image, reference_valid)

    output_lab_image = _rgb_to_lab(output)
    reference_lab_image = _rgb_to_lab(reference)
    output_lab = output_lab_image[output_valid].reshape(-1, 3)
    reference_lab = reference_lab_image[reference_valid].reshape(-1, 3)
    output_saturation = np.linalg.norm(output_lab[:, 1:3], axis=1)
    reference_saturation = np.linalg.norm(reference_lab[:, 1:3], axis=1)

    metrics: dict[str, float] = {
        "cbcr_swd": _sliced_wasserstein_2d(output_cbcr, reference_cbcr),
        "luminance_conditioned_cbcr_swd": _conditioned_chroma_swd(
            output_y, output_cbcr_image, reference_y, reference_cbcr_image, output_valid, reference_valid
        ),
        "chroma_mean_error": float(np.linalg.norm(output_cbcr.mean(axis=0) - reference_cbcr.mean(axis=0))),
        "chroma_covariance_error": _covariance_error(output_cbcr, reference_cbcr),
        "saturation_w1": _wasserstein_1d(output_saturation, reference_saturation),
        "neutral_axis_error": _neutral_axis_error(output_lab, reference_lab),
    }

    input_semantic_masks = dict(input_semantic_masks or {})
    reference_semantic_masks = dict(reference_semantic_masks or {})
    for semantic in _SEMANTIC_NAMES:
        output_semantic = input_semantic_masks.get(semantic)
        reference_semantic = reference_semantic_masks.get(semantic)
        if output_semantic is None or reference_semantic is None:
            metrics[f"semantic_{semantic}_lab_swd"] = float("nan")
            continue
        output_semantic = _mask_or_all(output_semantic, output.shape[:2], f"input_{semantic}_mask") & output_valid
        reference_semantic = _mask_or_all(
            reference_semantic, reference.shape[:2], f"reference_{semantic}_mask"
        ) & reference_valid
        if np.count_nonzero(output_semantic) < 8 or np.count_nonzero(reference_semantic) < 8:
            metrics[f"semantic_{semantic}_lab_swd"] = float("nan")
            continue
        metrics[f"semantic_{semantic}_lab_swd"] = _sliced_wasserstein_2d(
            output_lab_image[output_semantic, 1:3].reshape(-1, 2),
            reference_lab_image[reference_semantic, 1:3].reshape(-1, 2),
        )
    return metrics


def compute_all_metrics(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    output_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    input_semantic_masks: Optional[Mapping[str, np.ndarray]] = None,
    reference_semantic_masks: Optional[Mapping[str, np.ndarray]] = None,
) -> dict[str, float]:
    metrics = compute_luminance_metrics(
        output, reference, output_mask=output_mask, reference_mask=reference_mask
    )
    metrics.update(compute_color_metrics(
        output,
        reference,
        output_mask=output_mask,
        reference_mask=reference_mask,
        input_semantic_masks=input_semantic_masks,
        reference_semantic_masks=reference_semantic_masks,
    ))
    return metrics
