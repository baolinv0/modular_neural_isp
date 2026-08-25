from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "version": "1.0",
    "metric_version": "1.0",
    "prompt_version": "1.0",
    "weights": {
        "overall": {"brightness": 0.30, "color": 0.30, "tone": 0.40},
        "objective_vlm": {
            "brightness": {"objective": 0.60, "vlm": 0.40},
            "color": {"objective": 0.50, "vlm": 0.50},
            "tone": {"objective": 0.35, "vlm": 0.65},
        },
    },
    "metrics": {
        "brightness": {
            "weights": {"face_ev": 0.45, "face_background_ev": 0.30, "clip_ratio": 0.10, "dark_ratio": 0.10, "background_ev": 0.05},
            "face_ev": {"good": 0.15, "bad": 1.00},
            "face_background_ev": {"good": 0.15, "bad": 1.00},
            "clip_ratio": {"good": 0.005, "bad": 0.08},
            "dark_ratio": {"good": 0.01, "bad": 0.15},
            "background_ev": {"good": 0.20, "bad": 1.20},
        },
        "color": {
            "weights": {"delta_e00": 0.55, "hue_deg": 0.20, "chroma": 0.15, "skin_l": 0.05, "skin_env": 0.05},
            "delta_e00": {"good": 2.0, "bad": 12.0},
            "hue_deg": {"good": 3.0, "bad": 25.0},
            "chroma": {"good": 2.0, "bad": 15.0},
            "skin_l": {"good": 3.0, "bad": 15.0},
            "skin_env": {"good": 3.0, "bad": 18.0},
        },
        "tone": {
            "weights": {"quantile_ev": 0.45, "contrast_ev": 0.20, "highlight_ev": 0.15, "shadow_ev": 0.15, "ring_ev": 0.05},
            "quantile_ev": {"good": 0.10, "bad": 0.80},
            "contrast_ev": {"good": 0.10, "bad": 0.70},
            "highlight_ev": {"good": 0.10, "bad": 0.80},
            "shadow_ev": {"good": 0.10, "bad": 0.80},
            "ring_ev": {"good": 0.10, "bad": 0.80},
            "quantile_weights": [0.07, 0.10, 0.16, 0.24, 0.18, 0.14, 0.11],
        },
    },
    "roi": {
        "min_face_area_ratio": 0.02,
        "min_skin_pixels": 120,
        "ring_scale": 1.42,
        "skin_cr_min": 128,
        "skin_cr_max": 180,
        "skin_cb_min": 70,
        "skin_cb_max": 135,
    },
    "validation": {
        "warn_face_center_distance": 0.16,
        "invalid_face_center_distance": 0.38,
        "warn_face_scale_ev": 0.45,
        "invalid_face_scale_ev": 1.20,
        "min_valid_scenes": 1,
        "min_valid_ratio": 0.60,
    },
    "vlm": {
        "mode": "disabled",
        "required": False,
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "",
        "temperature": 0.0,
        "timeout_seconds": 120,
        "max_image_side": 1024,
        "confidence_threshold": 0.60,
        "min_valid_ratio": 0.70,
        "order_reversal": True,
        "near_gate_margin": 2.0,
    },
    "gates": {
        "overall_min": 85.0,
        "dimension_min": {"brightness": 82.0, "color": 82.0, "tone": 80.0},
        "p10_min": {"brightness": 75.0, "color": 75.0, "tone": 72.0},
        "failure_threshold": 60.0,
        "failure_rate_max": 0.05,
        "regression": {"overall_max_drop": 1.0, "dimension_max_drop": 2.0, "scene_regression_rate_max": 0.10, "failure_rate_worsen_tolerance": 0.0},
        "improvement": {"min_overall_gain": 1.5},
    },
    "content_guard": {"enabled": True, "ssim_min": 0.45, "gradient_corr_min": 0.20},
    "reporting": {"top_k": 10, "include_holdout_scene_details": False},
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


@dataclass(frozen=True)
class EvaluatorConfig:
    data: dict[str, Any]

    @classmethod
    def defaults(cls) -> "EvaluatorConfig":
        return cls(deepcopy(DEFAULT_CONFIG))

    @classmethod
    def load(cls, path: Path | str | None) -> "EvaluatorConfig":
        if path is None:
            return cls.defaults()
        payload = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(payload, dict):
            raise ValueError("evaluator config must be a mapping")
        return cls(_deep_merge(DEFAULT_CONFIG, payload))

    def with_overrides(self, values: dict[str, Any]) -> "EvaluatorConfig":
        return EvaluatorConfig(_deep_merge(self.data, values))

    def get(self, dotted: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
