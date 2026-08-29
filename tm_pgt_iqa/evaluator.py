from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from .config import IQAConfig
from .metrics import (
    evaluate_guards,
    evaluate_source_preservation,
    extract_features,
    load_rgb,
    score_features,
)
from .segmentation import load_label_map, resize_masks
from .vlm import QwenVLMClient, VLMReview
from .semantic_judge import QwenSemanticJudge, SceneAnalysis, SemanticReview


def select_semantic_topk(results: list["CandidateResult"], top_k: int) -> list["CandidateResult"]:
    """Keep objective top-two, then deterministic best family representatives.

    Manifests supply ``family``; this function intentionally never infers one
    from a filename.  Invalid and hard-rejected candidates are excluded before
    Qwen consumes its limited pairwise budget.
    """
    eligible = sorted(
        (result for result in results if result.pgt_class != "REJECT" and result.guards.get("passed", True)),
        key=lambda result: (-result.quality_score, result.candidate),
    )
    if top_k <= 0:
        return []
    selected = eligible[: min(2, top_k)]
    selected_names = {result.candidate for result in selected}
    selected_families = {result.family for result in selected}
    for result in eligible:
        if len(selected) >= top_k:
            break
        if result.candidate not in selected_names and result.family not in selected_families:
            selected.append(result)
            selected_names.add(result.candidate)
            selected_families.add(result.family)
    for result in eligible:
        if len(selected) >= top_k:
            break
        if result.candidate not in selected_names:
            selected.append(result)
            selected_names.add(result.candidate)
    return selected


def _semantic_failure(semantic: SemanticReview | dict | None) -> bool:
    if semantic is None:
        return False
    if isinstance(semantic, SemanticReview):
        tm_only, quality, naturalness = semantic.tm_only, semantic.semantic_quality, semantic.naturalness
    else:
        tm_only = str(semantic.get("tm_only", "SUSPICIOUS")).upper()
        quality = str(semantic.get("semantic_quality", "ACCEPTABLE")).upper()
        naturalness = semantic.get("naturalness", {}) or {}
    return tm_only == "FAIL" or quality == "POOR" or any(str(level).upper() == "MAJOR" for level in naturalness.values())


def _selection_population(results: list["CandidateResult"]) -> list["CandidateResult"]:
    """Freeze selection to the semantic Top-K once semantic review has begun."""
    reviewed = [result for result in results if result.semantic_reviewed]
    return reviewed if reviewed else results


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
    source_preservation: dict | None = None
    # This is populated from CandidateManifest by the V2 batch path.
    family: str = "unknown"
    pool_confidence: str = "HIGH"
    selection_confidence: str | None = None
    semantic_reviewed: bool = False

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
        family: str = "unknown",
    ) -> CandidateResult:
        rgb = load_rgb(str(image_path))
        masks = load_label_map(label_path, self.config.labels)
        if masks.face.shape != rgb.shape[:2]:
            masks = resize_masks(masks, rgb.shape[:2])

        features = extract_features(rgb, masks, self.config.guards)
        quality = score_features(features, self.config.weights, self.config.objective)
        preservation = None
        source_rgb = None
        if source_path is not None:
            source_rgb = load_rgb(str(source_path))
            preservation = evaluate_source_preservation(source_rgb, rgb, masks, self.config.guards)
        guards = evaluate_guards(features, self.config.guards, preservation)
        evidence = {
            "metrics": quality.to_dict(),
            "features": features.to_dict(),
            "guards": guards.to_dict(),
            "source_preservation": preservation.to_dict() if preservation else None,
        }

        review: VLMReview | None = None
        semantic: SemanticReview | None = None
        if run_vlm and source_path is not None and self.semantic_judge is not None:
            assert source_rgb is not None
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
            source_preservation=preservation.to_dict() if preservation else None,
            family=family,
        )

    def _classify(
        self,
        score: float,
        guards_passed: bool,
        review: VLMReview | None,
        pool_outlier: bool,
        semantic: SemanticReview | None = None,
    ) -> tuple[str, float]:
        if not guards_passed:
            return "REJECT", 0.0
        # Pool consistency is deliberately not an image-quality guard.  Qwen
        # TM-only failures and MAJOR semantic failures are.
        if _semantic_failure(semantic):
            return "REJECT", 0.0
        if review and review.decision == "REJECT" and review.confidence >= self.config.thresholds.qwen_reject_confidence:
            return "REJECT", 0.0
        if score >= self.config.thresholds.certified_score and (review is None or review.decision == "ACCEPT"):
            return "CERTIFIED_PGT", 1.0
        if score >= self.config.thresholds.usable_score:
            return "USABLE_PGT", 0.5
        return "REJECT", 0.0

    def apply_pool_consistency(self, results: list[CandidateResult]) -> list[CandidateResult]:
        """Set family-balanced pool confidence without changing PGT quality."""
        if not results:
            return results
        values_by_family: dict[str, list[float]] = {}
        for result in results:
            values_by_family.setdefault(result.family, []).append(float(result.features.get("face_bg_ev", 0.0)))
        family_medians = np.asarray([np.median(values) for values in values_by_family.values()], dtype=float)
        # A pool with only one represented family has no cross-family balance to
        # measure.  Fall back to that family's internal distribution solely for
        # a confidence diagnostic; it still cannot reject the candidate.
        reference = family_medians if len(family_medians) >= 2 else np.asarray(
            [float(result.features.get("face_bg_ev", 0.0)) for result in results], dtype=float
        )
        center = float(np.median(reference))
        mad = float(np.median(np.abs(reference - center)))
        # A zero MAD is common in simple pools; it is evidence of agreement,
        # not a failure mode.
        for result in results:
            value = float(result.features.get("face_bg_ev", 0.0))
            z = 0.0 if mad < 1e-6 else 0.6745 * abs(value - center) / mad
            result.pool_outlier = bool(z > self.config.guards.pool_outlier_z)
            result.pool_confidence = "LOW" if result.pool_outlier else (
                "MEDIUM" if z > self.config.guards.pool_outlier_z * 0.5 else "HIGH"
            )
        return results

    def review_semantic_topk(
        self,
        results: list[CandidateResult],
        source_path: str | Path,
        source_mask_path: str | Path,
        mask_paths: dict[str, str],
        top_k: int | None = None,
    ) -> dict:
        """Run scene understanding once and reuse it for all eligible Top-K."""
        top = select_semantic_topk(results, top_k or self.config.semantic.top_k)
        if not top or self.semantic_judge is None:
            return {"scene": None, "reviewed": []}
        source_rgb = load_rgb(str(source_path))
        source_masks = load_label_map(source_mask_path, self.config.labels)
        if source_masks.face.shape != source_rgb.shape[:2]:
            source_masks = resize_masks(source_masks, source_rgb.shape[:2])
        scene: SceneAnalysis = self.semantic_judge.analyze_scene(source_rgb, source_masks)
        reviewed: list[str] = []
        for result in top:
            rgb = load_rgb(result.candidate)
            masks = load_label_map(mask_paths[result.candidate], self.config.labels)
            if masks.face.shape != rgb.shape[:2]:
                masks = resize_masks(masks, rgb.shape[:2])
            evidence = {
                "candidate_name": result.candidate,
                "quality_score": result.quality_score,
                "metrics": result.metrics,
                "features": result.features,
                "guards": result.guards,
                "source_preservation": result.source_preservation,
            }
            semantic = self.semantic_judge.review_with_scene(source_rgb, rgb, masks, evidence, scene)
            result.semantic = semantic.to_dict()
            result.semantic_reviewed = True
            result.pgt_class, result.training_weight = self._classify(
                result.quality_score, result.guards.get("passed", True), None, result.pool_outlier, semantic
            )
            reviewed.append(result.candidate)
        return {"scene": scene.to_dict(), "reviewed": reviewed}

    def rank_candidates(
        self,
        results: list[CandidateResult],
        source_path: str | Path,
        mask_paths: dict[str, str],
        top_k: int | None = None,
    ) -> dict:
        top = select_semantic_topk(_selection_population(results), top_k or self.config.semantic.top_k)
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
                    points[a.candidate] += 1.0
                    points[b.candidate] -= 1.0
                elif effective == "B_BETTER":
                    points[b.candidate] += 1.0
                    points[a.candidate] -= 1.0
                comparisons.append(
                    {
                        "a": a.candidate,
                        "b": b.candidate,
                        **review.to_dict(),
                        "effective_preference": effective,
                    }
                )

        ordered = sorted(top, key=lambda r: (-points[r.candidate], -r.quality_score, r.candidate))
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

    def _selection_confidence(self, winner: CandidateResult, eligible: list[CandidateResult], tournament: dict | None) -> str:
        ordered_scores = sorted((r.quality_score for r in eligible if r.candidate != winner.candidate), reverse=True)
        score_gap = winner.quality_score - ordered_scores[0] if ordered_scores else self.config.semantic.high_objective_separation
        if score_gap < self.config.semantic.low_objective_separation:
            objective = "LOW"
        elif score_gap >= self.config.semantic.high_objective_separation:
            objective = "HIGH"
        else:
            objective = "MEDIUM"
        ranking = "LOW"
        if tournament:
            points = tournament.get("points", {})
            winner_points = float(points.get(winner.candidate, 0.0))
            runner_points = max((float(value) for name, value in points.items() if name != winner.candidate), default=winner_points)
            pairwise_confidences = [float(item.get("confidence", 0.0)) for item in tournament.get("comparisons", [])]
            pairwise_confidence = min(pairwise_confidences, default=0.0)
            if len(tournament.get("equivalent_top_set", [])) > 1 or winner_points <= runner_points:
                ranking = "LOW"
            elif pairwise_confidence < self.config.semantic.pairwise_min_confidence:
                ranking = "LOW"
            elif winner_points - runner_points >= 2.0 and pairwise_confidence >= self.config.semantic.selection_qwen_high_confidence:
                ranking = "HIGH"
            else:
                ranking = "MEDIUM"
        semantic = winner.semantic or {}
        qwen_confidence = float(semantic.get("confidence", 0.0)) if isinstance(semantic, dict) else 0.0
        if qwen_confidence >= self.config.semantic.selection_qwen_high_confidence:
            qwen = "HIGH"
        elif qwen_confidence >= self.config.semantic.pairwise_min_confidence:
            qwen = "MEDIUM"
        else:
            qwen = "LOW"
        levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        # Pool diagnostics intentionally do not participate in this decision.
        return min((objective, qwen, ranking), key=lambda value: levels.get(value, 0))

    def finalize_selection(self, results: list[CandidateResult], tournament: dict | None = None) -> dict:
        """Select a PGT while keeping class and selection certainty independent."""
        eligible = [
            result for result in _selection_population(results)
            if result.pgt_class != "REJECT" and result.guards.get("passed", True)
        ]
        if not eligible:
            return {"selected": None, "pgt_class": "REJECT", "training_weight": 0.0, "selection_confidence": "LOW"}
        by_name = {result.candidate: result for result in eligible}
        requested = (tournament or {}).get("winner")
        winner = by_name.get(requested) if requested else None
        if winner is None:
            winner = min(eligible, key=lambda result: (-result.quality_score, result.candidate))
        pgt_class, _ = self._classify(winner.quality_score, winner.guards.get("passed", True), None, winner.pool_outlier, winner.semantic)
        if pgt_class == "REJECT":
            return {"selected": winner.candidate, "pgt_class": "REJECT", "training_weight": 0.0, "selection_confidence": "LOW"}
        confidence = self._selection_confidence(winner, eligible, tournament)
        weight = 1.0 if pgt_class == "CERTIFIED_PGT" else {"HIGH": 0.7, "MEDIUM": 0.5, "LOW": 0.3}[confidence]
        winner.pgt_class = pgt_class
        winner.training_weight = weight
        winner.selection_confidence = confidence
        return {
            "selected": winner.candidate,
            "pgt_class": pgt_class,
            "training_weight": weight,
            "selection_confidence": confidence,
        }
