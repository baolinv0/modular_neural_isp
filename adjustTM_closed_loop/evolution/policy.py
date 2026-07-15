from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .features import FEATURE_NAMES, extract_image_features
from .teacher_manifest import load_teacher_manifest


@dataclass(frozen=True)
class PolicyPrediction:
    alpha: float
    raw_alpha: float
    in_domain: bool
    domain_distance: float
    reason: str


@dataclass
class RidgeAlphaPolicy:
    feature_names: Sequence[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha_min: float = -1.0
    alpha_max: float = 1.0
    ood_threshold: float = 6.0

    def __post_init__(self) -> None:
        self.feature_names = tuple(str(item) for item in self.feature_names)
        self.feature_mean = np.asarray(self.feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(self.feature_scale, dtype=np.float64)
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = (len(FEATURE_NAMES),)
        if self.feature_names != tuple(FEATURE_NAMES):
            raise ValueError("Policy feature_names do not match the frozen extractor contract")
        if self.feature_mean.shape != expected or self.feature_scale.shape != expected or self.coefficients.shape != expected:
            raise ValueError(f"Policy arrays must all have shape {expected}")
        if not all(np.isfinite(array).all() for array in (self.feature_mean, self.feature_scale, self.coefficients)):
            raise ValueError("Policy arrays must be finite")
        if not np.isfinite(self.intercept) or self.alpha_min > self.alpha_max or self.ood_threshold <= 0:
            raise ValueError("Invalid policy intercept, alpha range, or OOD threshold")

    def predict_features(self, features: np.ndarray) -> PolicyPrediction:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != self.feature_mean.shape:
            raise ValueError(f"Expected feature shape {self.feature_mean.shape}, got {features.shape}")
        z = (features - self.feature_mean) / np.maximum(self.feature_scale, 1e-8)
        distance = float(np.sqrt(np.mean(z * z)))
        raw = float(self.intercept + z @ self.coefficients)
        if not np.isfinite(raw) or distance > self.ood_threshold:
            return PolicyPrediction(0.0, raw, False, distance, "ood_fallback")
        return PolicyPrediction(float(np.clip(raw, self.alpha_min, self.alpha_max)), raw, True, distance, "predicted")

    def predict_path(self, path: str | Path) -> PolicyPrediction:
        return self.predict_features(extract_image_features(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "type": "weighted_ridge_alpha_policy", "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(), "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(), "intercept": self.intercept,
            "alpha_min": self.alpha_min, "alpha_max": self.alpha_max, "ood_threshold": self.ood_threshold,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RidgeAlphaPolicy":
        item = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(item["feature_names"], np.asarray(item["feature_mean"]), np.asarray(item["feature_scale"]),
                   np.asarray(item["coefficients"]), float(item["intercept"]), float(item["alpha_min"]),
                   float(item["alpha_max"]), float(item["ood_threshold"]))


def _split_report(model: RidgeAlphaPolicy, records: list[Any]) -> dict[str, Any]:
    if not records:
        return {"sample_count": 0, "weighted_mae": None, "weighted_rmse": None, "ood_rate": None}
    x = np.stack([extract_image_features(record.input_path) for record in records])
    y = np.asarray([record.selected_alpha for record in records], dtype=np.float64)
    w = np.asarray([max(record.sample_weight, 1e-6) for record in records], dtype=np.float64)
    predictions = [model.predict_features(row) for row in x]
    pred = np.asarray([item.alpha for item in predictions], dtype=np.float64)
    return {
        "sample_count": len(records),
        "weighted_mae": float(np.average(np.abs(pred - y), weights=w)),
        "weighted_rmse": float(np.sqrt(np.average((pred - y) ** 2, weights=w))),
        "ood_rate": float(np.mean([not item.in_domain for item in predictions])),
    }


def fit_policy_from_manifest(
    manifest_path: str | Path, *, ridge: float = 1e-2, alpha_min: float = -1.0,
    alpha_max: float = 1.0, ood_threshold: float = 6.0,
) -> tuple[RidgeAlphaPolicy, dict[str, Any]]:
    records = load_teacher_manifest(manifest_path)
    train_records = [record for record in records if record.split == "train"]
    if len(train_records) < 2:
        raise ValueError("At least two training teacher records are required")
    x = np.stack([extract_image_features(record.input_path) for record in train_records])
    y = np.asarray([record.selected_alpha for record in train_records], dtype=np.float64)
    w = np.asarray([max(record.sample_weight, 1e-6) for record in train_records], dtype=np.float64)
    mean = np.average(x, axis=0, weights=w)
    variance = np.average((x - mean) ** 2, axis=0, weights=w)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    z = (x - mean) / scale
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    sqrt_w = np.sqrt(w)[:, None]
    lhs = (design * sqrt_w).T @ (design * sqrt_w)
    penalty = np.eye(lhs.shape[0]) * float(ridge)
    penalty[0, 0] = 0.0
    rhs = (design * sqrt_w).T @ (y * sqrt_w[:, 0])
    beta = np.linalg.solve(lhs + penalty, rhs)
    model = RidgeAlphaPolicy(FEATURE_NAMES, mean, scale, beta[1:], float(beta[0]), alpha_min, alpha_max, ood_threshold)
    split_records = {
        "train": train_records,
        "validation": [record for record in records if record.split == "validation"],
        "test": [record for record in records if record.split == "test"],
    }
    report = {
        "sample_count": len(records),
        "fit_sample_count": len(train_records),
        "baseline_anchor_count": sum(abs(record.selected_alpha) <= 1e-8 for record in records),
        "feature_count": len(FEATURE_NAMES),
        "splits": {name: _split_report(model, subset) for name, subset in split_records.items()},
    }
    # Backward-compatible top-level train metrics for simple scripts.
    report["weighted_mae"] = report["splits"]["train"]["weighted_mae"]
    report["weighted_rmse"] = report["splits"]["train"]["weighted_rmse"]
    return model, report
