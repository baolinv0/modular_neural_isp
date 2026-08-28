from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from .config import IQAConfig
from .metrics import load_rgb, extract_features, score_features, evaluate_guards
from .segmentation import load_label_map, resize_masks
from .vlm import QwenVLMClient, VLMReview
from .semantic_judge import QwenSemanticJudge, SemanticReview


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
    semantic: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TMPGTEvaluator:
    def __init__(self, config: IQAConfig):
        self.config = config
        self.vlm_client = QwenVLMClient(config.vlm) if config.vlm.enabled else None
        self.semantic_judge = (
            QwenSemanticJudge(config.vlm)
            if config.vlm.enabled and config.semantic.enabled
            else None
        )

    def evaluate_one(
        self,
        image_path: str | Path,
        label_path: str | Path,
        run_vlm: bool = True,
        source_path: str | Path | None = None,
    ) -> CandidateResult:
        rgb = load_rgb(str(image_path))
        masks = load_label_map(label_path, self.config.labels)
        if masks.face.shape != rgb.shape[:2]:
            masks = resize_masks(masks, rgb.shape[:2])

        features = extract_features(rgb, masks)
        quality = score_features(features, self.config.weights)
        guards = evaluate_guards(features, self.config.guards)
        evidence = {
            "metrics": quality.to_dict(),
            "features": features.to_dict(),
            "guards": guards.to_dict(),
        }

        review: VLMReview | None = None
        semantic: SemanticReview | None = None
        if run_vlm and source_path is not None and self.semantic_judge is not None:
            source_rgb = load_rgb(str(source_path))
            semantic = self.semantic_judge.review(source_rgb, rgb, masks, evidence)
        elif run_vlm and self.vlm_client is not None:
            review = self.vlm_client.review(rgb, masks, evidence)

        pgt_class, weight = self._classify(
            quality.overall,
            guards.passed,
            review,
            pool_outlier=False,
            semantic=semantic,
        )
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
            semantic=semantic.to_dict() if semantic else None,
        )

    def _classify(
        self,
        score: float,
        guards_passed: bool,
        review: VLMReview | None,
        pool_outlier: bool,
        semantic: SemanticReview | None = None,
    ) -> tuple[str, float]:
        if not guards_passed or pool_outlier:
            return "REJECT", 0.0
        if semantic and semantic.confidence >= self.config.semantic.reject_confidence:
            if semantic.tm_only == "FAIL" or semantic.semantic_quality == "POOR":
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
                r.pgt_class = "REJECT"
                r.training_weight = 0.0
        return results

    def rank_candidates(
        self,
        results: list[CandidateResult],
        source_path: str | Path,
        mask_paths: dict[str, str],
        top_k: int | None = None,
    ) -> dict:
        eligible = [r for r in results if r.pgt_class != "REJECT"]
        eligible.sort(key=lambda r: r.quality_score, reverse=True)
        top = eligible[: top_k or self.config.semantic.top_k]
        if not top:
            return {"winner": None, "equivalent_top_set": [], "points": {}, "comparisons": []}
        if len(top) == 1 or self.semantic_judge is None:
            name = top[0].candidate
            return {"winner": name, "equivalent_top_set": [name], "points": {name: 0.0}, "comparisons": []}

        source_rgb = load_rgb(str(source_path))
        points = {r.candidate: 0.0 for r in top}
        comparisons: list[dict] = []
        candidate_data = {}
        for r in top:
            rgb = load_rgb(r.candidate)
            masks = load_label_map(mask_paths[r.candidate], self.config.labels)
            if masks.face.shape != rgb.shape[:2]:
                masks = resize_masks(masks, rgb.shape[:2])
            candidate_data[r.candidate] = (rgb, masks)

        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                a_rgb, a_masks = candidate_data[a.candidate]
                b_rgb, b_masks = candidate_data[b.candidate]
                a_evidence = {
                    "candidate_name": a.candidate,
                    "quality_score": a.quality_score,
                    "metrics": a.metrics,
                    "features": a.features,
                    "semantic": a.semantic,
                }
                b_evidence = {
                    "candidate_name": b.candidate,
                    "quality_score": b.quality_score,
                    "metrics": b.metrics,
                    "features": b.features,
                    "semantic": b.semantic,
                }
                review = self.semantic_judge.compare(
                    source_rgb,
                    a_rgb,
                    a_masks,
                    a_evidence,
                    b_rgb,
                    b_masks,
                    b_evidence,
                )
                effective = (
                    review.preference
                    if review.confidence >= self.config.semantic.pairwise_min_confidence
                    else "EQUIVALENT"
                )
                if effective == "A_BETTER":
                    points[a.candidate] += 2.0
                elif effective == "B_BETTER":
                    points[b.candidate] += 2.0
                else:
                    points[a.candidate] += 1.0
                    points[b.candidate] += 1.0
                comparisons.append(
                    {
                        "a": a.candidate,
                        "b": b.candidate,
                        **review.to_dict(),
                        "effective_preference": effective,
                    }
                )

        ordered = sorted(
            top,
            key=lambda r: (points[r.candidate], r.quality_score),
            reverse=True,
        )
        best_points = points[ordered[0].candidate]
        equivalent = [
            r.candidate
            for r in ordered
            if best_points - points[r.candidate] <= self.config.semantic.equivalent_margin
        ]
        return {
            "winner": ordered[0].candidate,
            "equivalent_top_set": equivalent,
            "points": points,
            "comparisons": comparisons,
        }
