import numpy as np

from photofinishing.evaluate.metrics import (
    compute_all_metrics,
    compute_color_metrics,
    compute_luminance_metrics,
    linear_to_srgb,
)


def _constant_linear(value, shape=(32, 32, 3)):
    return linear_to_srgb(np.full(shape, value, np.float32))


def test_identical_non_aligned_images_have_zero_metrics():
    image = _constant_linear(0.2)
    metrics = compute_all_metrics(image, image)
    for name, value in metrics.items():
        if name.startswith("semantic_"):
            continue
        assert abs(value) < 1e-6, (name, value)


def test_signed_ev_error_detects_one_stop_brightness_shift():
    reference = _constant_linear(0.2)
    output = _constant_linear(0.4)
    metrics = compute_luminance_metrics(output, reference)
    assert 0.98 < metrics["signed_ev_error"] < 1.02
    assert 0.98 < metrics["absolute_ev_error"] < 1.02


def test_tone_shape_metric_is_exposure_invariant_but_detects_contrast_change():
    x = np.linspace(0.03, 0.8, 32 * 32, dtype=np.float32).reshape(32, 32, 1)
    reference = linear_to_srgb(np.repeat(x, 3, axis=2))
    brighter = linear_to_srgb(np.clip(np.repeat(x * 1.5, 3, axis=2), 0, 1))
    contrast = linear_to_srgb(np.clip(np.repeat(np.power(x, 1.6), 3, axis=2), 0, 1))
    bright_metrics = compute_luminance_metrics(brighter, reference)
    contrast_metrics = compute_luminance_metrics(contrast, reference)
    assert bright_metrics["absolute_ev_error"] > 0.4
    assert bright_metrics["tone_shape_mae"] < contrast_metrics["tone_shape_mae"]


def test_color_metrics_detect_chroma_shift_without_luminance_shift():
    reference_linear = np.full((32, 32, 3), 0.25, np.float32)
    output_linear = reference_linear.copy()
    output_linear[..., 0] += 0.08
    output_linear[..., 1] -= 0.04
    output = linear_to_srgb(np.clip(output_linear, 0, 1))
    reference = linear_to_srgb(reference_linear)
    luma = compute_luminance_metrics(output, reference)
    color = compute_color_metrics(output, reference)
    assert luma["absolute_ev_error"] < 0.1
    assert color["cbcr_swd"] > 0.01
    assert color["chroma_mean_error"] > 0.01
    assert color["luminance_conditioned_cbcr_swd"] > 0.01


def test_semantic_metrics_use_separate_non_aligned_masks_and_report_composition_gap():
    reference = _constant_linear(0.2)
    output = reference.copy()
    output[:, :16, 0] = np.clip(output[:, :16, 0] + 0.2, 0, 1)
    input_skin = np.zeros((32, 32), bool)
    reference_skin = np.zeros((32, 32), bool)
    input_skin[:, :16] = True
    reference_skin[:, 16:24] = True
    metrics = compute_color_metrics(
        output,
        reference,
        input_semantic_masks={"skin": input_skin},
        reference_semantic_masks={"skin": reference_skin},
    )
    assert metrics["semantic_skin_lab_swd"] > 0
    assert 0.24 < metrics["semantic_skin_area_gap"] < 0.26
    assert metrics["semantic_composition_max_gap"] == metrics["semantic_skin_area_gap"]
    assert np.isnan(metrics["semantic_sky_lab_swd"])
