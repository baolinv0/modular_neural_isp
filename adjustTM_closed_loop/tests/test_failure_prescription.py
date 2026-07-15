from __future__ import annotations

from adjustTM_closed_loop.data_prescription import build_data_prescription
from adjustTM_closed_loop.failure_taxonomy import classify_failure_reasons
from adjustTM_closed_loop.schemas import FailureCode, QwenSceneResult


def test_failure_taxonomy_maps_qwen_reasons_to_tm_codes() -> None:
    codes = classify_failure_reasons(
        ["max_luminance_clip_ratio", "max_shadow_ratio", "max_color_drift", "max_monotonic_violation_rate"]
    )
    assert codes == {
        FailureCode.HIGHLIGHT_CLIPPING,
        FailureCode.SHADOW_CRUSH,
        FailureCode.CHROMA_DRIFT,
        FailureCode.CONTROL_CURVE,
    }


def test_unknown_reason_is_preserved_as_structural_artifact() -> None:
    assert classify_failure_reasons(["unexpected_geometry_change"]) == {FailureCode.STRUCTURAL_ARTIFACT}


def test_data_prescription_contains_positive_boundary_negative_and_regression_sets() -> None:
    scenes = {
        "a.png": QwenSceneResult("a.png", "REGENERATE", ("max_luminance_clip_ratio",), {}),
        "b.png": QwenSceneResult("b.png", "REVIEW", ("max_luminance_clip_ratio",), {}),
        "c.png": QwenSceneResult("c.png", "REGENERATE", ("max_monotonic_violation_rate",), {}),
        "d.png": QwenSceneResult("d.png", "KEEP", (), {}),
    }
    prescription = build_data_prescription(scenes)
    clipping = next(item for item in prescription.tasks if item.failure_code is FailureCode.HIGHLIGHT_CLIPPING)
    assert clipping.target_module == "Gain/GTM control"
    assert set(clipping.positive_scenes) == {"a.png", "b.png"}
    assert "d.png" in clipping.regression_anchor_scenes
    assert clipping.required_supervision
    assert clipping.acceptance_gates


def test_no_failure_scenes_produces_empty_tasks() -> None:
    prescription = build_data_prescription({"ok.png": QwenSceneResult("ok.png", "KEEP", (), {})})
    assert prescription.tasks == ()
