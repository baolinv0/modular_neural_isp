from __future__ import annotations

import statistics
from typing import Iterable, Mapping


INTENT_SCORES = (
    "global_brightness_match", "spatial_brightness_match", "highlight_target_match",
    "shadow_target_match", "local_contrast_match", "color_target_match", "structural_consistency",
)
NATURALNESS_SCORES = (
    "exposure_naturalness", "highlight_rolloff", "shadow_quality", "local_contrast_quality",
    "color_wb_stability", "skin_tone_naturalness", "artifact_absence", "structural_fidelity",
    "overall_naturalness",
)


def build_prompt(kind: str, *, alpha: float) -> dict[str, str]:
    if kind == "intent":
        system = (
            "你是一名手机 ISP 与 Tone Mapping 图像质量专家。请评价候选图像是否准确实现目标图像所表达的亮度调节意图。"
            "重点比较整体与空间亮度、高光、阴影、局部对比度、颜色和结构。不要猜测模型，只输出严格 JSON。"
        )
        user = f"Image 1 是真实相机中心图，Image 2 是 alpha={alpha:.3f} 的固定目标图，Image 3 是候选输出。"
    elif kind == "naturalness":
        system = (
            "你是一名手机摄影与 Tone Mapping 图像质量专家。评价候选图像作为最终照片是否自然、稳定且无副作用。"
            "不要评价它是否接近某张目标图。检查曝光、高光滚降、阴影、局部对比度、颜色白平衡、肤色、伪影和结构。"
            "不要猜测模型，只输出严格 JSON。"
        )
        user = f"Image 1 是真实相机中心图，Image 2 是 alpha={alpha:.3f} 的候选亮度调整结果。"
    else:
        raise ValueError(f"Unknown VLM prompt kind: {kind}")
    return {"system": system, "user": user}


def validate_response(record: Mapping[str, object], *, required_scores: Iterable[str]) -> dict[str, object]:
    normalized = dict(record)
    for key in required_scores:
        value = normalized.get(key)
        if value is None and key == "skin_tone_naturalness":
            continue
        if not isinstance(value, (int, float)) or not 1 <= float(value) <= 5:
            raise ValueError(f"Invalid VLM score for {key}: {value}")
        normalized[key] = float(value)
    confidence = normalized.get("confidence")
    if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1):
        raise ValueError(f"Invalid confidence: {confidence}")
    return normalized


def aggregate_repeats(
    records: Iterable[Mapping[str, object]],
    *,
    required_scores: Iterable[str],
    unstable_std_threshold: float = 1.0,
) -> dict[str, object]:
    validated = [validate_response(record, required_scores=required_scores) for record in records]
    if not validated:
        raise ValueError("No VLM repetitions")
    scores: dict[str, dict[str, float]] = {}
    unstable = False
    for key in required_scores:
        values = [float(record[key]) for record in validated if record.get(key) is not None]
        if not values:
            scores[key] = {"median": float("nan"), "mean": float("nan"), "std": float("nan"), "range": float("nan")}
            continue
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        scores[key] = {
            "median": float(statistics.median(values)),
            "mean": float(statistics.fmean(values)),
            "std": float(std),
            "range": float(max(values) - min(values)),
        }
        unstable = unstable or std > unstable_std_threshold
    return {"scores": scores, "unstable_judgment": unstable, "repeat_count": len(validated)}
