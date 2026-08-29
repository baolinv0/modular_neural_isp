from __future__ import annotations

import json
import numpy as np

from tm_pgt_iqa.candidate_generation import generate_pool, write_pool
from tm_pgt_iqa.config import IQAConfig
from tm_pgt_iqa.metrics import luminance
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
    assert pool["a_p20"].manifest.source.startswith("memory://sha256/")


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
