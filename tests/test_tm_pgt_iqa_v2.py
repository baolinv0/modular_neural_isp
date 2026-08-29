from __future__ import annotations

import json
import numpy as np
from PIL import Image

from tm_pgt_iqa.candidate_generation import generate_pool, write_pool
from tm_pgt_iqa.config import IQAConfig, load_config
from tm_pgt_iqa.evaluator import CandidateResult, TMPGTEvaluator, select_semantic_topk
from tm_pgt_iqa.metrics import (
    evaluate_guards,
    evaluate_source_preservation,
    extract_features,
    luminance,
    score_features,
)
from tm_pgt_iqa.segmentation import SemanticMasks


def _scene() -> tuple[np.ndarray, SemanticMasks]:
    """A backlit portrait-like synthetic scene with a dark face."""
    h, w = 48, 48
    rgb = np.full((h, w, 3), 0.78, dtype=np.float32)
    rgb[14:34, 14:34] = (0.22, 0.17, 0.14)
    # Retain an intra-face luminance structure for the tone-shape transform.
    rgb[18:30, 18:30] *= np.linspace(0.65, 1.30, 12, dtype=np.float32)[:, None, None]

    face = np.zeros((h, w), dtype=bool)
    face[14:34, 14:34] = True
    skin = face.copy()
    human = np.zeros_like(face)
    human[10:38, 10:38] = True
    background = ~human
    return rgb, SemanticMasks(face=face, skin=skin, background=background, human=human)


def _by_id(pool):
    return {candidate.manifest.candidate_id: candidate for candidate in pool}


def test_retinex_levels_have_monotonic_face_and_global_luminance():
    source, masks = _scene()
    pool = _by_id(generate_pool(source, masks, IQAConfig(), source_name="scene.png"))
    ids = ("a_m30", "a_m20", "a_m10", "a_000", "a_p10", "a_p20", "a_p30", "a_p40")
    face_medians = [float(np.median(luminance(pool[name].rgb)[masks.face])) for name in ids]
    global_medians = [float(np.median(luminance(pool[name].rgb))) for name in ids]
    assert all(b > a for a, b in zip(face_medians, face_medians[1:]))
    assert all(b > a for a, b in zip(global_medians, global_medians[1:]))
    assert pool["a_p20"].manifest.family == "retinex"
    assert pool["a_p20"].manifest.parameters["ev"] == 0.30


def test_local_face_lift_changes_face_more_than_background():
    source, masks = _scene()
    pool = _by_id(generate_pool(source, masks, IQAConfig(), source_name="scene.png"))
    candidate = pool["face_lift_mid"].rgb
    source_y, candidate_y = luminance(source), luminance(candidate)
    face_delta = float(np.median(candidate_y[masks.face] - source_y[masks.face]))
    bg_delta = float(np.median(candidate_y[masks.background] - source_y[masks.background]))
    assert face_delta > 0.0
    assert face_delta > 8.0 * bg_delta + 1e-5
    assert pool["face_lift_mid"].manifest.parameters == {"lift_ev": 0.60}


def test_tone_shape_preserves_face_median_exposure():
    source, masks = _scene()
    cfg = IQAConfig()
    pool = _by_id(generate_pool(source, masks, cfg, source_name="scene.png"))
    source_ev = np.log2(np.median(luminance(source)[masks.face]) + 1e-6)
    for candidate_id in ("tone_shadow_preserve", "tone_soft_highlight"):
        candidate_ev = np.log2(np.median(luminance(pool[candidate_id].rgb)[masks.face]) + 1e-6)
        assert abs(candidate_ev - source_ev) <= cfg.candidate_generation.tone_shape.median_ev_tolerance


def test_tone_shape_changes_face_distribution_without_moving_median_ev():
    source, masks = _scene()
    cfg = IQAConfig()
    candidate = _by_id(generate_pool(source, masks, cfg, source_name="scene.png"))["tone_shadow_preserve"].rgb
    source_face_ev = np.log2(luminance(source)[masks.face] + 1e-6)
    candidate_face_ev = np.log2(luminance(candidate)[masks.face] + 1e-6)
    source_span = float(np.percentile(source_face_ev, 90) - np.percentile(source_face_ev, 10))
    candidate_span = float(np.percentile(candidate_face_ev, 90) - np.percentile(candidate_face_ev, 10))
    assert abs(candidate_span - source_span) > 0.01
    assert abs(float(np.median(candidate_face_ev) - np.median(source_face_ev))) <= cfg.candidate_generation.tone_shape.median_ev_tolerance


def test_label_maps_derive_human_core_rings_and_soft_masks_are_preferred():
    h, w = 32, 32
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[6:26, 6:26] = 3
    labels[10:22, 10:22] = 1
    labels[12:20, 12:20] = 2
    soft_face = np.zeros((h, w), dtype=np.float32)
    soft_face[11:21, 11:21] = 1.0
    masks = SemanticMasks.from_label_map(labels, IQAConfig().labels, soft_face=soft_face)
    assert masks.human[7, 7]
    assert masks.face_core[15, 15]
    assert masks.face_inner_ring.any()
    assert masks.face_outer_ring.any()
    # The supplied soft map—not the label extent—defines the face support.
    assert not masks.face[10, 10]
    assert masks.face[15, 15]


def test_soft_masks_produce_exclusive_face_human_background_regions():
    labels = np.zeros((24, 24), dtype=np.uint8)
    labels[4:20, 4:20] = 3
    soft_face = np.zeros_like(labels, dtype=np.float32)
    soft_face[9:15, 9:15] = 1.0
    soft_human = np.zeros_like(labels, dtype=np.float32)
    soft_human[6:18, 6:18] = 1.0  # Deliberately overlaps the face map.
    masks = SemanticMasks.from_label_map(
        labels, IQAConfig().labels, soft_face=soft_face, soft_human=soft_human
    )
    assert not np.any(masks.face & masks.human)
    assert not np.any(masks.face & masks.background)
    assert not np.any(masks.human & masks.background)
    assert np.all(masks.face | masks.human | masks.background)
    assert np.all(masks.skin <= masks.face)


def test_retinex_preserves_source_luminance_structure_and_derives_provenance():
    source, masks = _scene()
    pool = _by_id(generate_pool(source, masks, IQAConfig()))
    source_y = luminance(source).reshape(-1)
    candidate_y = luminance(pool["a_p20"].rgb).reshape(-1)
    assert np.corrcoef(source_y, candidate_y)[0, 1] > 0.99999
    assert pool["a_p20"].manifest.source == "memory://unspecified-source"


def test_write_pool_persists_authoritative_manifest(tmp_path):
    source, masks = _scene()
    pool = generate_pool(source, masks, IQAConfig(), source_name="scene.png")
    write_pool(pool, tmp_path)
    manifest = json.loads((tmp_path / "face_lift_mid.json").read_text(encoding="utf-8"))
    assert manifest == {
        "candidate_id": "face_lift_mid",
        "family": "local_face_tm",
        "parameters": {"lift_ev": 0.60},
        "source": "scene.png",
    }


def test_supplied_derived_mask_shape_must_match_base_mask():
    face = np.ones((8, 8), dtype=bool)
    try:
        SemanticMasks(face=face, skin=face, background=~face, face_core=np.ones((7, 8), dtype=bool))
    except ValueError as error:
        assert "face_core" in str(error)
    else:
        raise AssertionError("mismatched derived mask shape should fail")


def test_global_brightening_increases_face_exposure_score():
    source, masks = _scene()
    darker = np.clip(source * 0.55, 0.0, 1.0)
    brighter = np.clip(source * 1.45, 0.0, 1.0)
    cfg = IQAConfig()
    dark_quality = score_features(extract_features(darker, masks), cfg.weights)
    bright_quality = score_features(extract_features(brighter, masks), cfg.weights)
    assert bright_quality.exposure > dark_quality.exposure


def test_flat_face_reduces_face_tone_score():
    source, masks = _scene()
    flat = source.copy()
    flat[masks.face] = np.median(source[masks.face], axis=0)
    cfg = IQAConfig()
    structured_quality = score_features(extract_features(source, masks), cfg.weights)
    flat_quality = score_features(extract_features(flat, masks), cfg.weights)
    assert flat_quality.face_tone < structured_quality.face_tone


def test_clipping_guard_remains_a_hard_failure():
    source, masks = _scene()
    clipped = source.copy()
    clipped[masks.face] = 1.0
    guards = evaluate_guards(extract_features(clipped, masks), IQAConfig().guards)
    assert not guards.passed
    assert "FACE_HIGHLIGHT_CLIP" in guards.failures


def test_feature_clip_and_dark_ratios_use_guard_config_thresholds():
    source, masks = _scene()
    source[masks.face] = 0.80
    cfg = IQAConfig()
    # Thresholds apply to linear-RGB luminance, not display-referred sRGB.
    cfg.guards.face_highlight_threshold = 0.55
    cfg.guards.face_dark_threshold = 0.65
    features = extract_features(source, masks, cfg.guards)
    assert features.face_clip_ratio == 1.0
    assert features.face_dark_ratio == 1.0


def test_local_face_lift_changes_face_background_relation():
    source, masks = _scene()
    lifted = source.copy()
    lifted[masks.face] = np.clip(lifted[masks.face] * 1.8, 0.0, 1.0)
    source_features = extract_features(source, masks)
    lifted_features = extract_features(lifted, masks)
    assert lifted_features.face_bg_ev > source_features.face_bg_ev + 0.5


def test_clear_non_tm_structural_modification_fails_source_preservation():
    source, masks = _scene()
    rewritten = source.copy()
    # Replace face structure with a bright block: an exposure-only transform
    # cannot create this new strong edge geometry.
    rewritten[17:31, 17:31] = (0.9, 0.9, 0.9)
    preservation = evaluate_source_preservation(source, rewritten, masks, IQAConfig().guards)
    assert preservation.status == "FAIL"
    assert preservation.structural_failure


def test_local_face_lift_is_not_failed_only_for_low_frequency_difference():
    source, masks = _scene()
    cfg = IQAConfig()
    # This deliberately makes LF change alone breach the fail threshold.  The
    # candidate still preserves the independent edge/face-structure signals.
    cfg.guards.preservation_low_frequency_fail = 0.001
    lifted = _by_id(generate_pool(source, masks, cfg))['face_lift_high'].rgb
    preservation = evaluate_source_preservation(source, lifted, masks, cfg.guards)
    assert preservation.low_frequency_error > cfg.guards.preservation_low_frequency_fail
    assert preservation.edge_position_agreement > cfg.guards.preservation_edge_agreement_fail
    assert not preservation.structural_failure
    assert preservation.status != "FAIL"


def test_legacy_halo_reject_yaml_controls_v2_halo_warning(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("guards:\n  halo_reject: 0.50\n", encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.guards.halo_warning is None
    source, masks = _scene()
    halo = source.copy()
    halo[masks.face_inner_ring] = 1.0
    features = extract_features(halo, masks, cfg.guards)
    assert features.halo_strength < cfg.guards.halo_reject
    assert "LOCAL_TM_HALO" not in evaluate_guards(features, cfg.guards).warnings


def test_evaluate_one_attaches_preservation_and_rejects_structural_edit(tmp_path):
    source, masks = _scene()
    rewritten = source.copy()
    rewritten[17:31, 17:31] = (0.9, 0.9, 0.9)
    source_path = tmp_path / "source.png"
    candidate_path = tmp_path / "candidate.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray((source * 255).astype(np.uint8)).save(source_path)
    Image.fromarray((rewritten * 255).astype(np.uint8)).save(candidate_path)
    labels = np.zeros(source.shape[:2], dtype=np.uint8)
    labels[masks.face] = 1
    labels[masks.skin] = 2
    labels[masks.human] = 3
    Image.fromarray(labels).save(mask_path)
    cfg = IQAConfig()
    cfg.vlm.enabled = False
    result = TMPGTEvaluator(cfg).evaluate_one(
        candidate_path, mask_path, run_vlm=False, source_path=source_path
    )
    assert result.source_preservation is not None
    assert result.source_preservation["status"] == "FAIL"
    assert "SOURCE_PRESERVATION_FAIL" in result.guards["failures"]
    assert result.pgt_class == "REJECT"


def _selection_result(name: str, score: float, family: str, semantic=None) -> CandidateResult:
    return CandidateResult(
        candidate=name,
        quality_score=score,
        metrics={"overall": score},
        features={"face_bg_ev": 0.0},
        guards={"passed": True, "failures": []},
        vlm=None,
        pool_outlier=False,
        pgt_class="CERTIFIED_PGT",
        training_weight=1.0,
        semantic=semantic,
        family=family,
    )


def test_family_balanced_topk_keeps_top_two_then_other_families():
    results = [
        _selection_result("r1", 99.0, "retinex"),
        _selection_result("r2", 98.0, "retinex"),
        _selection_result("r3", 97.0, "retinex"),
        _selection_result("r4", 96.0, "retinex"),
        _selection_result("local", 95.0, "local_face_tm"),
        _selection_result("shape", 94.0, "tone_shape"),
        _selection_result("gain", 93.0, "gain"),
    ]
    selected = select_semantic_topk(results, top_k=4)
    assert [result.candidate for result in selected] == ["r1", "r2", "local", "shape"]


def test_family_balanced_topk_uses_candidate_id_as_stable_score_tie_breaker():
    results = [
        _selection_result("z_retinex", 90.0, "retinex"),
        _selection_result("b_retinex", 90.0, "retinex"),
        _selection_result("a_retinex", 90.0, "retinex"),
        _selection_result("local", 80.0, "local_face_tm"),
    ]
    selected = select_semantic_topk(results, top_k=2)
    assert [result.candidate for result in selected] == ["a_retinex", "b_retinex"]


def test_qwen_tm_only_fail_rejects_even_at_low_model_confidence():
    from tm_pgt_iqa.semantic_judge import SceneIntent, SemanticReview

    review = SemanticReview(
        scene_type="NORMAL",
        scene_intent=SceneIntent(False, "MEDIUM", "MEDIUM", "MEDIUM"),
        naturalness={},
        tm_only="FAIL",
        semantic_quality="GOOD",
        confidence=0.10,
        summary="content modification",
    )
    cls, weight = TMPGTEvaluator(IQAConfig())._classify(95.0, True, None, False, semantic=review)
    assert (cls, weight) == ("REJECT", 0.0)


def test_pairwise_winner_is_selected_with_win_tie_loss_point_tournament():
    evaluator = TMPGTEvaluator(IQAConfig())
    results = [
        _selection_result("a", 91.0, "retinex"),
        _selection_result("b", 90.0, "local_face_tm"),
        _selection_result("c", 89.0, "tone_shape"),
    ]
    final = evaluator.finalize_selection(
        results,
        {
            "winner": "b",
            "equivalent_top_set": ["b"],
            "points": {"a": -1.0, "b": 2.0, "c": -1.0},
            "comparisons": [],
        },
    )
    assert final["selected"] == "b"
    assert final["pgt_class"] == "CERTIFIED_PGT"
    assert final["training_weight"] == 1.0


def test_low_objective_separation_does_not_downgrade_certified_quality():
    evaluator = TMPGTEvaluator(IQAConfig())
    results = [
        _selection_result("winner", 90.0, "retinex"),
        _selection_result("nearby", 89.5, "local_face_tm"),
    ]
    final = evaluator.finalize_selection(
        results,
        {
            "winner": "winner",
            "equivalent_top_set": ["winner", "nearby"],
            "points": {"winner": 0.0, "nearby": 0.0},
            "comparisons": [],
        },
    )
    assert final["pgt_class"] == "CERTIFIED_PGT"
    assert final["training_weight"] == 1.0
    assert final["selection_confidence"] == "LOW"


def test_pool_confidence_cannot_lower_selection_confidence_or_training_weight():
    evaluator = TMPGTEvaluator(IQAConfig())
    semantic = {
        "tm_only": "PASS",
        "semantic_quality": "GOOD",
        "naturalness": {},
        "confidence": 0.95,
    }
    winner = _selection_result("winner", 95.0, "retinex", semantic=semantic)
    winner.pool_confidence = "LOW"
    other = _selection_result("other", 80.0, "local_face_tm", semantic=semantic)
    final = evaluator.finalize_selection(
        [winner, other],
        {
            "winner": "winner",
            "equivalent_top_set": ["winner"],
            "points": {"winner": 3.0, "other": -3.0},
            "comparisons": [{"confidence": 0.95}, {"confidence": 0.95}],
        },
    )
    assert final["pgt_class"] == "CERTIFIED_PGT"
    assert final["training_weight"] == 1.0
    assert final["selection_confidence"] == "HIGH"


def test_low_qwen_review_confidence_lowers_selection_confidence():
    evaluator = TMPGTEvaluator(IQAConfig())
    winner = _selection_result("winner", 95.0, "retinex", semantic={
        "tm_only": "PASS", "semantic_quality": "GOOD", "naturalness": {}, "confidence": 0.10,
    })
    other = _selection_result("other", 80.0, "local_face_tm")
    final = evaluator.finalize_selection(
        [winner, other],
        {"winner": "winner", "equivalent_top_set": ["winner"], "points": {"winner": 3.0, "other": -3.0}, "comparisons": [{"confidence": 0.95}]},
    )
    assert final["pgt_class"] == "CERTIFIED_PGT"
    assert final["selection_confidence"] == "LOW"


def test_finalize_objective_fallback_uses_stable_candidate_id_tie_breaker():
    evaluator = TMPGTEvaluator(IQAConfig())
    results = [
        _selection_result("z_candidate", 90.0, "retinex"),
        _selection_result("a_candidate", 90.0, "local_face_tm"),
    ]
    final = evaluator.finalize_selection(results)
    assert final["selected"] == "a_candidate"


def test_semantic_rejection_cannot_promote_unreviewed_candidate_during_finalization():
    evaluator = TMPGTEvaluator(IQAConfig())
    rejected = _selection_result("reviewed_rejected", 99.0, "retinex")
    rejected.pgt_class = "REJECT"
    rejected.semantic_reviewed = True
    reviewed = _selection_result("reviewed_usable", 80.0, "local_face_tm")
    reviewed.semantic_reviewed = True
    unreviewed = _selection_result("unreviewed_c", 98.0, "tone_shape")
    final = evaluator.finalize_selection([rejected, reviewed, unreviewed])
    assert final["selected"] == "reviewed_usable"


def test_ranking_stays_with_reviewed_set_after_a_semantic_rejection(tmp_path):
    source = np.full((8, 8, 3), 0.3, dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.uint8)
    source_path = tmp_path / "source.png"
    Image.fromarray((source * 255).astype(np.uint8)).save(source_path)
    results, mask_paths = [], {}
    for name, score, family in (("rejected", 99.0, "retinex"), ("reviewed", 80.0, "local_face_tm"), ("unreviewed", 98.0, "tone_shape")):
        image_path, mask_path = tmp_path / f"{name}.png", tmp_path / f"{name}_mask.png"
        Image.fromarray((source * 255).astype(np.uint8)).save(image_path)
        Image.fromarray(labels).save(mask_path)
        result = _selection_result(str(image_path), score, family)
        result.semantic_reviewed = name != "unreviewed"
        if name == "rejected":
            result.pgt_class = "REJECT"
        results.append(result)
        mask_paths[str(image_path)] = str(mask_path)

    evaluator = TMPGTEvaluator(IQAConfig())
    ranking = evaluator.rank_candidates(results, source_path, mask_paths, top_k=4)
    assert ranking["winner"].endswith("reviewed.png")


def test_scene_understanding_is_called_once_and_reused_for_topk_candidates(tmp_path):
    from tm_pgt_iqa.semantic_judge import SceneAnalysis, SceneIntent, SemanticReview

    source = np.full((16, 16, 3), 0.3, dtype=np.float32)
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[4:12, 4:12] = 1
    source_path = tmp_path / "source.png"
    source_mask_path = tmp_path / "source_mask.png"
    Image.fromarray((source * 255).astype(np.uint8)).save(source_path)
    source_labels = np.zeros((16, 16), dtype=np.uint8)
    source_labels[1:3, 1:3] = 1
    Image.fromarray(source_labels).save(source_mask_path)
    mask_paths = {}
    results = []
    for name, score, family in (("a", 91.0, "retinex"), ("b", 90.0, "local_face_tm")):
        image_path = tmp_path / f"{name}.png"
        mask_path = tmp_path / f"{name}_mask.png"
        Image.fromarray((source * (1.0 if name == "a" else 1.1) * 255).astype(np.uint8)).save(image_path)
        Image.fromarray(labels).save(mask_path)
        mask_paths[str(image_path)] = str(mask_path)
        results.append(_selection_result(str(image_path), score, family))

    class FakeJudge:
        def __init__(self):
            self.scene_calls = 0
            self.candidate_calls = 0

        def analyze_scene(self, source_rgb, masks):
            self.scene_calls += 1
            assert masks.face[1, 1]
            assert not masks.face[5, 5]
            return SceneAnalysis("BACKLIGHT", SceneIntent(True, "HIGH", "MEDIUM", "HIGH"), 0.9, "dark face")

        def review_with_scene(self, source_rgb, candidate_rgb, masks, evidence, scene):
            self.candidate_calls += 1
            assert scene.scene_type == "BACKLIGHT"
            return SemanticReview(scene.scene_type, scene.scene_intent, {}, "PASS", "GOOD", 0.9, "natural")

    evaluator = TMPGTEvaluator(IQAConfig())
    fake = FakeJudge()
    evaluator.semantic_judge = fake
    semantic = evaluator.review_semantic_topk(results, source_path, source_mask_path, mask_paths, top_k=2)
    assert fake.scene_calls == 1
    assert fake.candidate_calls == 2
    assert semantic["scene"]["scene_type"] == "BACKLIGHT"
    assert all(result.semantic["scene_type"] == "BACKLIGHT" for result in results)


def test_cli_defers_source_aware_semantic_calls_until_after_objective_topk(tmp_path):
    """The production entrypoint must not invoke Qwen from evaluate_one per candidate."""
    from unittest.mock import patch
    import sys
    from tm_pgt_iqa.cli import main

    images_dir, masks_dir = tmp_path / "images", tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    rgb = np.full((8, 8, 3), 0.3, dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.uint8)
    for name in ("a", "b"):
        Image.fromarray((rgb * 255).astype(np.uint8)).save(images_dir / f"{name}.png")
        Image.fromarray(labels).save(masks_dir / f"{name}.png")
        (images_dir / f"{name}.json").write_text(json.dumps({"family": "retinex" if name == "a" else "local_face_tm"}), encoding="utf-8")
    source_path, source_mask_path = tmp_path / "source.png", tmp_path / "source_mask.png"
    Image.fromarray((rgb * 255).astype(np.uint8)).save(source_path)
    Image.fromarray(labels).save(source_mask_path)
    output = tmp_path / "report.json"

    class FakeEvaluator:
        instance = None

        def __init__(self, config):
            self.semantic_judge = object()
            self.evaluate_calls = []
            self.semantic_calls = []
            FakeEvaluator.instance = self

        def evaluate_one(self, image_path, label_path, **kwargs):
            self.evaluate_calls.append(kwargs)
            return _selection_result(str(image_path), 90.0, kwargs["family"])

        def apply_pool_consistency(self, results):
            return results

        def review_semantic_topk(self, results, source, source_mask, mask_paths):
            self.semantic_calls.append((source, source_mask, mask_paths))
            return {"scene": {"scene_type": "NORMAL"}, "reviewed": [result.candidate for result in results]}

        def rank_candidates(self, results, source, mask_paths):
            return {"winner": results[0].candidate, "equivalent_top_set": [results[0].candidate], "points": {}, "comparisons": []}

        def finalize_selection(self, results, tournament):
            return {"selected": results[0].candidate, "pgt_class": "CERTIFIED_PGT", "training_weight": 1.0, "selection_confidence": "HIGH"}

    argv = [
        "tm_pgt_iqa", "--images", str(images_dir), "--masks", str(masks_dir),
        "--source", str(source_path), "--source-mask", str(source_mask_path), "--output", str(output),
    ]
    with patch("tm_pgt_iqa.cli.TMPGTEvaluator", FakeEvaluator), patch.object(sys, "argv", argv):
        assert main() == 0
    assert all(call["run_vlm"] is False for call in FakeEvaluator.instance.evaluate_calls)
    assert len(FakeEvaluator.instance.semantic_calls) == 1
    assert FakeEvaluator.instance.semantic_calls[0][1] == source_mask_path
