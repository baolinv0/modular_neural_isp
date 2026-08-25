from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from tm_pgt_iqa.config import IQAConfig, VLMConfig
from tm_pgt_iqa.segmentation import load_label_map
from tm_pgt_iqa.metrics import extract_features, score_features, evaluate_guards
from tm_pgt_iqa.vlm import parse_review, QwenVLMClient
from tm_pgt_iqa.evaluator import TMPGTEvaluator, CandidateResult


def make_scene(tmp_path: Path, value=0.45):
    h, w = 32, 32
    rgb = np.full((h, w, 3), value, dtype=np.float32)
    for y in range(8, 24):
        rgb[y, 8:24] = np.clip(value + (y - 16) * 0.012, 0, 1)
    image_path = tmp_path / "a.png"
    Image.fromarray((rgb * 255).astype(np.uint8)).save(image_path)
    label = np.zeros((h, w), dtype=np.uint8)
    label[7:25, 7:25] = 1
    label[9:23, 9:23] = 2
    mask_path = tmp_path / "a_mask.png"
    Image.fromarray(label).save(mask_path)
    return rgb, image_path, mask_path


def test_semantic_label_map_builds_face_skin_background(tmp_path):
    _, _, mask_path = make_scene(tmp_path)
    masks = load_label_map(mask_path, IQAConfig().labels)
    assert masks.face.sum() > masks.skin.sum() > 0
    assert masks.background.sum() > 0


def test_metrics_produce_five_quality_dimensions(tmp_path):
    rgb, _, mask_path = make_scene(tmp_path)
    cfg = IQAConfig()
    masks = load_label_map(mask_path, cfg.labels)
    features = extract_features(rgb, masks)
    quality = score_features(features, cfg.weights)
    assert 0 <= quality.exposure <= 100
    assert 0 <= quality.dynamic_range <= 100
    assert 0 <= quality.face_tone <= 100
    assert 0 <= quality.face_background <= 100
    assert 0 <= quality.naturalness <= 100
    assert 0 <= quality.overall <= 100


def test_clipped_face_triggers_guard(tmp_path):
    rgb, _, mask_path = make_scene(tmp_path)
    cfg = IQAConfig()
    masks = load_label_map(mask_path, cfg.labels)
    rgb[masks.face] = 1.0
    features = extract_features(rgb, masks)
    guards = evaluate_guards(features, cfg.guards)
    assert not guards.passed
    assert "FACE_HIGHLIGHT_CLIP" in guards.failures


def test_qwen_parser_keeps_only_supported_failure_labels():
    review = parse_review('''```json\n{"decision":"REJECT","failures":["FACE_TOO_FLAT","NOISE"],"confidence":0.9,"summary":"flat"}\n```''')
    assert review.decision == "REJECT"
    assert review.failures == ["FACE_TOO_FLAT"]
    assert review.confidence == 0.9


def test_qwen_payload_contains_two_local_images(tmp_path):
    rgb, _, mask_path = make_scene(tmp_path)
    cfg = IQAConfig(vlm=VLMConfig(enabled=True, model="qwen3.8"))
    masks = load_label_map(mask_path, cfg.labels)
    payload = QwenVLMClient(cfg.vlm).build_payload(rgb, masks, {"score": 80})
    content = payload["messages"][0]["content"]
    urls = [x["image_url"]["url"] for x in content if x["type"] == "image_url"]
    assert len(urls) == 2
    assert all(u.startswith("data:image/jpeg;base64,") for u in urls)
    assert payload["model"] == "qwen3.8"


def test_evaluator_runs_without_vlm(tmp_path):
    _, image_path, mask_path = make_scene(tmp_path)
    cfg = IQAConfig()
    cfg.vlm.enabled = False
    result = TMPGTEvaluator(cfg).evaluate_one(image_path, mask_path, run_vlm=False)
    assert result.vlm is None
    assert result.pgt_class in {"CERTIFIED_PGT", "USABLE_PGT", "REJECT"}
    assert 0 <= result.quality_score <= 100


def test_high_confidence_qwen_reject_overrides_good_score():
    cfg = IQAConfig()
    evaluator = TMPGTEvaluator(cfg)
    from tm_pgt_iqa.vlm import VLMReview
    cls, weight = evaluator._classify(95, True, VLMReview("REJECT", ["OVER_HDR_LOOK"], 0.9, "bad"), False)
    assert cls == "REJECT"
    assert weight == 0.0


def test_pool_outlier_is_rejected():
    cfg = IQAConfig()
    cfg.vlm.enabled = False
    evaluator = TMPGTEvaluator(cfg)
    def r(name, ev):
        return CandidateResult(name, 90, {"overall": 90}, {"face_bg_ev": ev}, {"passed": True, "failures": []}, None, False, "CERTIFIED_PGT", 1.0)
    results = [r("a", 0.2), r("b", 0.21), r("c", 0.22), r("d", 2.5)]
    evaluator.apply_pool_consistency(results)
    assert results[-1].pool_outlier
    assert results[-1].pgt_class == "REJECT"
