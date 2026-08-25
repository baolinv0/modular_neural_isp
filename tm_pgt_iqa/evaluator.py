from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from .config import IQAConfig
from .metrics import load_rgb, extract_features, score_features, evaluate_guards
from .segmentation import load_label_map, resize_masks
from .vlm import QwenVLMClient, VLMReview


@dataclass
class CandidateResult:
    candidate: str
    quality_score: float
    metrics: dict
    features: dict
    guards: dict
    vlm: dict | None
    pool_outlier: bool
    pgt_class: str
    training_weight: float

    def to_dict(self) -> dict:
        return asdict(self)


class TMPGTEvaluator:
    def __init__(self, config: IQAConfig):
        self.config = config
        self.vlm_client = QwenVLMClient(config.vlm) if config.vlm.enabled else None

    def evaluate_one(self, image_path: str | Path, label_path: str | Path, run_vlm: bool = True) -> CandidateResult:
        rgb = load_rgb(str(image_path))
        masks = load_label_map(label_path, self.config.labels)
        if masks.face.shape != rgb.shape[:2]:
            masks = resize_masks(masks, rgb.shape[:2])
        features = extract_features(rgb, masks)
        quality = score_features(features, self.config.weights)
        guards = evaluate_guards(features, self.config.guards)
        review: VLMReview | None = None
        if run_vlm and self.vlm_client is not None:
            review = self.vlm_client.review(
                rgb,
                masks,
                {
                    "metrics": quality.to_dict(),
                    "features": features.to_dict(),
                    "guards": guards.to_dict(),
                },
            )
        pgt_class, weight = self._classify(quality.overall, guards.passed, review, pool_outlier=False)
        return CandidateResult(
            candidate=str(image_path),
            quality_score=quality.overall,
            metrics=quality.to_dict(),
            features=features.to_dict(),
            guards=guards.to_dict(),
            vlm=review.to_dict() if review else None,
            pool_outlier=False,
            pgt_class=pgt_class,
            training_weight=weight,
        )

    def _classify(self, score: float, guards_passed: bool, review: VLMReview | None, pool_outlier: bool) -> tuple[str, float]:
        if not guards_passed or pool_outlier:
            return "REJECT", 0.0
        if review and review.decision == "REJECT" and review.confidence >= self.config.thresholds.qwen_reject_confidence:
            return "REJECT", 0.0
        if score >= self.config.thresholds.certified_score and (review is None or review.decision == "ACCEPT"):
            return "CERTIFIED_PGT", 1.0
        if score >= self.config.thresholds.usable_score:
            return "USABLE_PGT", 0.5
        return "REJECT", 0.0

    def apply_pool_consistency(self, results: list[CandidateResult]) -> list[CandidateResult]:
        if len(results) < 3:
            return results
        values = np.array([r.features["face_bg_ev"] for r in results], dtype=float)
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        if mad < 1e-6:
            return results
        robust_z = 0.6745 * np.abs(values - med) / mad
        for r, z in zip(results, robust_z):
            if z > self.config.guards.pool_outlier_z:
                r.pool_outlier = True
                review = VLMReview(**r.vlm) if r.vlm else None
                r.pgt_class, r.training_weight = self._classify(
                    r.quality_score,
                    r.guards["passed"],
                    review,
                    True,
                )
        return results
