"""Behavioral contracts for pre-render HDR tone mapping."""
import importlib

import pytest
import torch
from torch import nn


@pytest.fixture
def mapper_type():
    def construct(*args, **kwargs):
        try:
            mapper = importlib.import_module("capture_tm.tone").PatentToneMapper
        except ModuleNotFoundError:
            pytest.fail("capture_tm.tone has not been implemented")
        return mapper(*args, **kwargs)
    return construct


def gray(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1).repeat(1, 3, 1, 1)


@pytest.mark.parametrize("mode", ["apple", "samsung"])
def test_hdr_highlights_are_not_clipped_before_mapping(mapper_type, mode):
    out = mapper_type(mode, backend=nn.Identity())(gray([0., .1, .5, 1., 2., 4.]))
    y = out["pre_backend"][0, 0, 0]
    assert y[0] == 0
    assert torch.all(y[1:] > y[:-1])
    assert torch.all((y >= 0) & (y <= 1))
    assert torch.isfinite(out["tone_curve"]).all()
    assert torch.all(out["tone_curve"].diff(dim=-1) >= -1e-6)
    assert out["tone_curve_input"].shape == out["tone_curve"].shape
    assert out["tone_curve_input"][0, -1] >= 4


def test_apple_applies_explicit_gain_once_in_linear_midtones(mapper_type):
    out = mapper_type("apple", backend=nn.Identity())(gray([.05, .10, .20, .30]), virtual_gain=2.)
    torch.testing.assert_close(out["output"][0, 0, 0], torch.tensor([.10, .20, .40, .60]))
    assert out["virtual_gain"].item() == 2.


@pytest.mark.parametrize("mode", ["apple", "samsung"])
def test_capture_compensation_does_not_cancel_render_intent(mapper_type, mode):
    mapper = mapper_type(mode, backend=nn.Identity())
    x = gray([.05, .1, .15, .2])
    neutral = mapper(x, capture_bias_ev=1.)
    styled = mapper(x, capture_bias_ev=1., render_intent_ev=1.)
    assert neutral["virtual_gain"].item() == .5
    assert styled["virtual_gain"].item() == 1.
    assert styled["capture_compensation_ev"].item() == -1.
    assert styled["output"].mean() > neutral["output"].mean()


def test_apple_lowtone_reference_is_stable_across_real_capture_scales(mapper_type):
    mapper = mapper_type("apple", backend=nn.Identity())
    reference = gray([.03, .06, .1, .2])
    normal = mapper(reference)["pre_backend"]
    biased = mapper(reference * 2., capture_bias_ev=1.)["pre_backend"]
    torch.testing.assert_close(normal, biased)


def test_samsung_gain_changes_target_histogram_curve_and_output(mapper_type):
    mapper = mapper_type("samsung", backend=nn.Identity(), curve_bins=128)
    x = gray(torch.linspace(.02, .25, 128).tolist())
    one = mapper(x, virtual_gain=1.)
    two = mapper(x, virtual_gain=2.)
    torch.testing.assert_close(one["source_histogram"], two["source_histogram"])
    assert not torch.allclose(one["target_histogram"], two["target_histogram"])
    assert not torch.allclose(one["tone_curve"], two["tone_curve"])
    assert two["output"].mean() > one["output"].mean()
    torch.testing.assert_close(two["target_histogram"].sum(-1), torch.ones(1))
    # A second gain multiplication would make the maximum approach 1 rather than .5.
    assert two["pre_backend"].max() < .6


def test_samsung_dark_flat_fallback_preserves_capture_compensation_and_intent(mapper_type):
    mapper = mapper_type("samsung", backend=nn.Identity())
    neutral = mapper(gray([.001]))
    compensated = mapper(gray([.002]), capture_bias_ev=1.)
    styled = mapper(gray([.002]), capture_bias_ev=1., render_intent_ev=1.)
    torch.testing.assert_close(neutral["output"], gray([.001]), rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(compensated["output"], gray([.001]), rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(styled["output"], gray([.002]), rtol=1e-6, atol=1e-9)
    assert neutral["degenerate_distribution_mask"].tolist() == [True]
    assert compensated["degenerate_distribution_mask"].tolist() == [True]
    torch.testing.assert_close(compensated["scaled_histogram"].sum(-1), torch.ones(1))


def test_samsung_distribution_fallback_uses_resolution_per_image(mapper_type):
    mapper = mapper_type("samsung", backend=nn.Identity())
    # First log-input bin spans about .00272; these two values are unresolved.
    x = torch.cat([gray([.001, .002]), gray([.01, .2])])
    out = mapper(x)
    assert out["degenerate_distribution_mask"].dtype == torch.bool
    assert out["degenerate_distribution_mask"].tolist() == [True, False]
    torch.testing.assert_close(out["pre_backend"][0], x[0], rtol=1e-6, atol=1e-9)
    assert out["target_histogram"].shape[0] == 2
    assert not torch.allclose(out["tone_curve"][1], out["tone_curve_input"][1])


def test_clipped_compensation_reports_requested_and_actual_gain(mapper_type):
    out = mapper_type("apple", backend=nn.Identity(), max_render_ev=2.)(
        gray([.1]), capture_bias_ev=8.)
    assert out["requested_capture_compensation_ev"].item() == -8.
    assert out["capture_compensation_ev"].item() == -2.
    assert out["requested_render_ev"].item() == -8.
    assert out["applied_render_ev"].item() == -2.
    assert out["virtual_gain"].item() == .25


@pytest.mark.parametrize("mode", ["apple", "samsung"])
def test_per_image_requests_and_black_input_are_finite(mapper_type, mode):
    mapper = mapper_type(mode, backend=nn.Identity())
    x = torch.cat([gray([.1, .2]), gray([.1, .2])])
    out = mapper(x, render_intent_ev=torch.tensor([0., 1.]))
    assert out["output"][1].mean() > out["output"][0].mean()
    assert torch.isfinite(mapper(torch.zeros_like(x))["output"]).all()
    assert mapper(torch.zeros_like(x))["output"].count_nonzero() == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_invalid_sensor_linear_samples_are_rejected(mapper_type, bad):
    with pytest.raises(ValueError):
        mapper_type("apple", backend=nn.Identity())(gray([bad]))


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_invalid_virtual_gain_is_rejected(mapper_type, bad):
    with pytest.raises(ValueError):
        mapper_type("samsung", backend=nn.Identity())(gray([.1]), virtual_gain=bad)


def test_subject_mask_is_validated_and_cannot_access_clean_target(mapper_type):
    mapper = mapper_type("samsung", backend=nn.Identity())
    x = gray([.1, .2, .3])
    out = mapper(x, subject_mask=torch.zeros(1, 1, 1, 3))
    assert out["output"].shape == x.shape
    with pytest.raises(ValueError):
        mapper(x, subject_mask=torch.ones(1, 1, 2, 3))
