"""Stage-wise analysis wrapper for the Modular Neural ISP pipeline.

The wrapper intentionally does not reimplement AWB, AE, or tone mapping. It
runs the existing ``PipeLine`` twice so the post-AE/pre-TM linear image becomes
an explicit, inspectable boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
import torch


_PROCESSING_ORDER = [
    "raw",
    "denoised_raw",
    "linear_awb",
    "linear_ae",
    "gain",
    "gtm",
    "ltm",
    "chroma",
    "gamma",
    "final",
]

_FIRST_PASS_REQUIRED = (
    "raw",
    "denoised_raw",
    "lsrgb",
    "srgb",
    "ev",
    "illum",
    "ccm",
)

_SECOND_PASS_REQUIRED = (
    "srgb",
    "gain_param",
    "gtm_param",
    "ltm_param",
)

_RESERVED_PIPELINE_KWARGS = {
    "raw",
    "denoised_raw",
    "lsrgb",
    "illum",
    "ccm",
    "img_metadata",
    "auto_exposure",
    "photofinishing",
    "return_intermediate",
    "enhancement_strength",
    "sharpening_amount",
    "log_messages",
    "report_time",
    "always_return_np",
}


def _as_bhwc_float64(image: Any) -> np.ndarray:
    """Convert BCHW/CHW/BHWC/HWC RGB data to finite BHWC float64."""
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(image).__name__}.")

    if array.ndim == 3:
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))[None, ...]
        elif array.shape[-1] == 3:
            array = array[None, ...]
        else:
            raise ValueError(f"Expected a 3-channel CHW or HWC image, got shape {array.shape}.")
    elif array.ndim == 4:
        if array.shape[1] == 3:
            array = np.transpose(array, (0, 2, 3, 1))
        elif array.shape[-1] != 3:
            raise ValueError(f"Expected a 3-channel BCHW or BHWC image, got shape {array.shape}.")
    else:
        raise ValueError(f"Expected a 3D or 4D RGB image, got shape {array.shape}.")

    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite values.")
    return array


def compute_luminance_stats(
    image: Any,
    *,
    low_clip_threshold: float = 1e-4,
    high_clip_threshold: float = 1.0 - 1e-4,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute robust Rec.709 luminance statistics for a stage image."""
    if not 0.0 <= low_clip_threshold < high_clip_threshold <= 1.0:
        raise ValueError("Clip thresholds must satisfy 0 <= low < high <= 1.")

    rgb = _as_bhwc_float64(image)
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )

    p01, p50, p99 = np.percentile(luminance, [1.0, 50.0, 99.0])
    dynamic_range = max(0.0, float(np.log2((p99 + eps) / (p01 + eps))))
    return {
        "mean": float(np.mean(luminance)),
        "std": float(np.std(luminance)),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "low_clip_ratio": float(np.mean(luminance <= low_clip_threshold)),
        "high_clip_ratio": float(np.mean(luminance >= high_clip_threshold)),
        "robust_dynamic_range_stops": dynamic_range,
    }


def to_jsonable(value: Any) -> Any:
    """Recursively convert tensors and NumPy values to JSON-compatible types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"Cannot convert {type(value).__name__} to a JSON-compatible value.")


def summarize_parameter(value: Any, *, max_inline_values: int = 32) -> Any:
    """Summarize a numeric parameter without expanding large tensors in JSON."""
    if value is None:
        return None
    if max_inline_values < 0:
        raise ValueError("max_inline_values must be non-negative.")
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    elif isinstance(value, (np.ndarray, np.generic, int, float)):
        array = np.asarray(value)
    else:
        raise TypeError(f"Expected a numeric parameter, got {type(value).__name__}.")
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected a numeric parameter, got dtype {array.dtype}.")
    array64 = array.astype(np.float64, copy=False)
    if not np.isfinite(array64).all():
        raise ValueError("Parameter contains non-finite values.")
    summary = {
        "shape": list(array.shape),
        "numel": int(array.size),
        "min": float(array64.min()) if array.size else None,
        "max": float(array64.max()) if array.size else None,
        "mean": float(array64.mean()) if array.size else None,
        "std": float(array64.std()) if array.size else None,
    }
    if array.size <= max_inline_values:
        summary["values"] = to_jsonable(value)
    return summary


def _validate_keys(output: Mapping[str, Any], required: tuple[str, ...], pass_name: str) -> None:
    missing = [key for key in required if key not in output]
    if missing:
        raise KeyError(f"{pass_name} output is missing required key(s): {', '.join(missing)}")


def _without_reserved(extra: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if extra is None:
        return {}
    conflicts = sorted(_RESERVED_PIPELINE_KWARGS.intersection(extra))
    if conflicts:
        raise ValueError(
            "extra_pipeline_kwargs cannot override analyzer-controlled argument(s): "
            + ", ".join(conflicts)
        )
    return dict(extra)


class FullPipelineAnalyzer:
    """Expose AWB/AE and tone rendering as two observable pipeline passes."""

    def __init__(self, pipeline: Any):
        if not callable(pipeline):
            raise TypeError("pipeline must be callable.")
        self.pipeline = pipeline

    def run(
        self,
        raw: torch.Tensor,
        metadata: Optional[dict],
        illum: Any = None,
        ccm: Any = None,
        *,
        auto_exposure: bool = True,
        denoising_strength: Optional[float] = None,
        chroma_denoising_strength: Optional[float] = None,
        luma_denoising_strength: Optional[float] = None,
        enhancement_strength: Optional[float] = None,
        sharpening_amount: float = 0.0,
        use_cc_awb: bool = False,
        awb_user_pref: bool = False,
        adjust_ccm: bool = False,
        target_cct: Optional[float] = None,
        target_tint: Optional[float] = None,
        style_id: int = 0,
        downscale_ps: bool = True,
        post_process_ltm: bool = False,
        solver_iter: Optional[int] = None,
        contrast_amount: float = 0.0,
        vibrance_amount: float = 0.0,
        saturation_amount: float = 0.0,
        highlight_amount: float = 0.0,
        shadow_amount: float = 0.0,
        ev_scale: float = 0.0,
        extra_pipeline_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not torch.is_tensor(raw):
            raise TypeError("raw must be a BCHW torch.Tensor so both passes preserve tensor outputs.")
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError(f"raw must have shape [B, 3, H, W], got {tuple(raw.shape)}.")

        extra = _without_reserved(extra_pipeline_kwargs)

        first_kwargs: MutableMapping[str, Any] = dict(
            illum=illum,
            ccm=ccm,
            img_metadata=metadata,
            denoising_strength=denoising_strength,
            chroma_denoising_strength=None,
            luma_denoising_strength=None,
            enhancement_strength=0.0,
            sharpening_amount=0.0,
            use_cc_awb=use_cc_awb,
            awb_user_pref=awb_user_pref,
            adjust_ccm=adjust_ccm,
            target_cct=target_cct,
            target_tint=target_tint,
            style_id=style_id,
            auto_exposure=auto_exposure,
            ev_scale=ev_scale,
            photofinishing=False,
            return_intermediate=False,
            log_messages=True,
            report_time=True,
            always_return_np=False,
        )
        first_start = perf_counter()
        first_output = self.pipeline(raw, **first_kwargs)
        first_seconds = perf_counter() - first_start
        _validate_keys(first_output, _FIRST_PASS_REQUIRED, "capture/color/exposure pass")

        second_kwargs: MutableMapping[str, Any] = dict(
            denoised_raw=first_output["denoised_raw"],
            lsrgb=first_output["srgb"],
            illum=first_output["illum"],
            ccm=first_output["ccm"],
            img_metadata=metadata,
            denoising_strength=None,
            chroma_denoising_strength=chroma_denoising_strength,
            luma_denoising_strength=luma_denoising_strength,
            enhancement_strength=enhancement_strength,
            sharpening_amount=sharpening_amount,
            use_cc_awb=use_cc_awb,
            awb_user_pref=False,
            adjust_ccm=False,
            target_cct=None,
            target_tint=None,
            style_id=style_id,
            downscale_ps=downscale_ps,
            post_process_ltm=post_process_ltm,
            contrast_amount=contrast_amount,
            vibrance_amount=vibrance_amount,
            saturation_amount=saturation_amount,
            highlight_amount=highlight_amount,
            shadow_amount=shadow_amount,
            auto_exposure=False,
            ev_scale=0.0,
            photofinishing=True,
            return_intermediate=True,
            log_messages=True,
            report_time=True,
            always_return_np=False,
        )
        if solver_iter is not None:
            second_kwargs["solver_iter"] = solver_iter
        second_kwargs.update(extra)

        second_start = perf_counter()
        second_output = self.pipeline(raw, **second_kwargs)
        second_seconds = perf_counter() - second_start
        _validate_keys(second_output, _SECOND_PASS_REQUIRED, "tone-rendering pass")

        stages = OrderedDict(
            raw=first_output.get("raw"),
            denoised_raw=first_output.get("denoised_raw"),
            linear_awb=first_output.get("lsrgb"),
            linear_ae=first_output.get("srgb"),
            gain=second_output.get("lsrgb_gain"),
            gtm=second_output.get("lsrgb_gtm"),
            ltm=second_output.get("lsrgb_ltm"),
            chroma=second_output.get("processed_lsrgb"),
            gamma=second_output.get("gamma"),
            final=second_output.get("srgb"),
        )

        stage_statistics: Dict[str, Dict[str, float]] = {}
        missing_optional_stages = []
        for name, image in stages.items():
            if image is None:
                missing_optional_stages.append(name)
                continue
            stage_statistics[name] = compute_luminance_stats(image)

        report = {
            "processing_order": list(_PROCESSING_ORDER),
            "configuration": {
                "auto_exposure": bool(auto_exposure),
                "recompute_awb": illum is None,
                "use_cc_awb": bool(use_cc_awb),
                "awb_user_pref": bool(awb_user_pref),
                "style_id": int(style_id),
                "downscale_photofinishing": bool(downscale_ps),
                "post_process_ltm": bool(post_process_ltm),
                "ev_scale": float(ev_scale),
            },
            "exposure": {
                "enabled": bool(auto_exposure),
                "ev": to_jsonable(first_output.get("ev")),
            },
            "white_balance": {
                "illum": to_jsonable(first_output.get("illum")),
                "ccm": to_jsonable(first_output.get("ccm")),
                "cct": to_jsonable(first_output.get("cct")),
                "tint": to_jsonable(first_output.get("tint")),
            },
            "tone_parameters": {
                "gain": summarize_parameter(second_output.get("gain_param")),
                "gtm": summarize_parameter(second_output.get("gtm_param")),
                "ltm": summarize_parameter(second_output.get("ltm_param")),
                "chroma_lut": summarize_parameter(second_output.get("chroma_lut_param")),
                "gamma": summarize_parameter(second_output.get("gamma_param")),
            },
            "stage_statistics": stage_statistics,
            "missing_optional_stages": missing_optional_stages,
            "timing_seconds": {
                "capture_color_exposure": float(first_seconds),
                "tone_rendering": float(second_seconds),
                "total": float(first_seconds + second_seconds),
            },
        }

        return {
            "stages": stages,
            "first_pass": first_output,
            "second_pass": second_output,
            "report": to_jsonable(report),
        }
