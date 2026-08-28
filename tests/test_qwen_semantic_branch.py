from __future__ import annotations

import numpy as np

from tm_pgt_iqa.config import IQAConfig
from tm_pgt_iqa.segmentation import SemanticMasks
from tm_pgt_iqa.semantic_judge import (
    QwenSemanticJudge,
    parse_semantic_review,
    parse_pairwise_review,
)


def _rgb(value: float) -> np.ndarray:
    return np.full((16, 16, 3), value, dtype=np.float32)


def _masks() -> SemanticMasks:
    face = np.zeros((16, 16), dtype=bool)
    face[4:12, 4:12] = True
    skin = np.zeros_like(face)
    skin[5:11, 5:11] = True
    background = ~face
    return SemanticMasks(face=face, skin=skin, background=background)


def test_parse_semantic_review_returns_compact_structured_judgment():
    text = '''```json
    {
      "scene_type":"BACKLIGHT",
      "scene_intent":{
        "face_lift_needed":true,
        "background_preservation":"HIGH",
        "shadow_atmosphere":"MEDIUM",
        "highlight_priority":"HIGH"
      },
      "naturalness":{
        "FACE_TOO_FLAT":"NONE",
        "FACE_OVER_LIFTED":"MINOR",
        "OVER_HDR_LOOK":"NONE",
        "LIGHTING_CAUSALITY_BROKEN":"NONE"
      },
      "tm_only":"PASS",
      "semantic_quality":"GOOD",
      "confidence":0.91,
      "summary":"Backlit face is lifted without breaking the lighting relationship."
    }
    ```'''
    review = parse_semantic_review(text)
    assert review.scene_type == "BACKLIGHT"
    assert review.scene_intent.face_lift_needed is True
    assert review.naturalness["FACE_OVER_LIFTED"] == "MINOR"
    assert review.tm_only == "PASS"
    assert review.semantic_quality == "GOOD"
    assert review.confidence == 0.91


def test_semantic_payload_contains_source_candidate_and_overlay():
    cfg = IQAConfig().vlm
    judge = QwenSemanticJudge(cfg)
    payload = judge.build_semantic_payload(
        source_rgb=_rgb(0.25),
        candidate_rgb=_rgb(0.55),
        masks=_masks(),
        evidence={"metrics": {"face_tone": 88}},
    )
    content = payload["messages"][0]["content"]
    images = [x for x in content if x["type"] == "image_url"]
    assert len(images) == 3
    assert all(x["image_url"]["url"].startswith("data:image/jpeg;base64,") for x in images)
    prompt = next(x["text"] for x in content if x["type"] == "text")
    assert "Image 1 = SOURCE" in prompt
    assert "Image 2 = CANDIDATE" in prompt
    assert "TM-only" in prompt


def test_parse_pairwise_review_supports_equivalent():
    review = parse_pairwise_review(
        '{"preference":"EQUIVALENT","primary_reason":"Both preserve natural face tone.","confidence":0.78}'
    )
    assert review.preference == "EQUIVALENT"
    assert review.confidence == 0.78


def test_pairwise_payload_contains_source_a_b_and_two_overlays():
    cfg = IQAConfig().vlm
    judge = QwenSemanticJudge(cfg)
    payload = judge.build_pairwise_payload(
        source_rgb=_rgb(0.25),
        a_rgb=_rgb(0.48),
        a_masks=_masks(),
        a_evidence={"quality_score": 86},
        b_rgb=_rgb(0.52),
        b_masks=_masks(),
        b_evidence={"quality_score": 87},
    )
    content = payload["messages"][0]["content"]
    images = [x for x in content if x["type"] == "image_url"]
    assert len(images) == 5
    prompt = next(x["text"] for x in content if x["type"] == "text")
    assert "A_BETTER" in prompt and "B_BETTER" in prompt and "EQUIVALENT" in prompt


def test_config_has_semantic_branch_defaults():
    cfg = IQAConfig()
    assert cfg.semantic.enabled is True
    assert cfg.semantic.top_k == 5
    assert cfg.semantic.reject_confidence == 0.80


def test_high_confidence_tm_only_fail_rejects_candidate():
    from tm_pgt_iqa.evaluator import TMPGTEvaluator
    from tm_pgt_iqa.semantic_judge import SemanticReview, SceneIntent
    cfg = IQAConfig()
    evaluator = TMPGTEvaluator(cfg)
    review = SemanticReview(
        scene_type="NORMAL",
        scene_intent=SceneIntent(False, "MEDIUM", "MEDIUM", "MEDIUM"),
        naturalness={},
        tm_only="FAIL",
        semantic_quality="POOR",
        confidence=0.92,
        summary="face content changed beyond tone mapping",
    )
    cls, weight = evaluator._classify(95.0, True, None, False, semantic=review)
    assert cls == "REJECT"
    assert weight == 0.0


def test_pairwise_ranking_uses_qwen_preference_without_changing_quality_score(tmp_path):
    from tm_pgt_iqa.evaluator import TMPGTEvaluator, CandidateResult
    from tm_pgt_iqa.semantic_judge import PairwiseReview
    from PIL import Image

    source = tmp_path / "source.png"
    Image.fromarray((_rgb(0.3) * 255).astype(np.uint8)).save(source)
    masks = _masks()
    label = np.zeros((16, 16), dtype=np.uint8)
    label[masks.face] = 1
    label[masks.skin] = 2

    results = []
    mask_paths = {}
    for name, value, score in (("a", 0.48, 88.0), ("b", 0.52, 87.0), ("c", 0.50, 86.0)):
        image_path = tmp_path / f"{name}.png"
        mask_path = tmp_path / f"{name}_mask.png"
        Image.fromarray((_rgb(value) * 255).astype(np.uint8)).save(image_path)
        Image.fromarray(label).save(mask_path)
        mask_paths[str(image_path)] = str(mask_path)
        results.append(CandidateResult(str(image_path), score, {"overall": score}, {}, {"passed": True}, None, False, "CERTIFIED_PGT", 1.0, semantic=None))

    cfg = IQAConfig()
    evaluator = TMPGTEvaluator(cfg)

    class FakeJudge:
        def compare(self, source_rgb, a_rgb, a_masks, a_evidence, b_rgb, b_masks, b_evidence):
            a = a_evidence["candidate_name"]
            b = b_evidence["candidate_name"]
            if a.endswith("b.png"):
                return PairwiseReview("A_BETTER", "better tone", 0.9)
            if b.endswith("b.png"):
                return PairwiseReview("B_BETTER", "better tone", 0.9)
            return PairwiseReview("EQUIVALENT", "similar", 0.9)

    evaluator.semantic_judge = FakeJudge()
    ranking = evaluator.rank_candidates(results, source, mask_paths, top_k=3)
    assert ranking["winner"].endswith("b.png")
    assert [r.quality_score for r in results] == [88.0, 87.0, 86.0]
