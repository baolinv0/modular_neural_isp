"""Mechanism checks for edge-aware base/detail tone control."""

import pytest
import torch


def _operator(**kwargs):
    from tm_library.config import ModelConfig
    from tm_library.operators.base_detail import BaseDetailOperator

    return BaseDetailOperator(ModelConfig(algorithm="base_detail", width=8, **kwargs))


def _fixed_controls(operator, detail_bias=0.0, contrast_bias=0.0, shift_bias=0.0):
    with torch.no_grad():
        operator.base_head.weight.zero_()
        operator.base_head.bias.copy_(torch.tensor([contrast_bias, shift_bias]))
        operator.detail_head.weight.zero_()
        operator.detail_head.bias.fill_(detail_bias)


def test_detail_control_changes_texture_but_preserves_constant_fields():
    operator = _operator(filter_radius=2)
    x = torch.linspace(0.15, 0.7, 17).view(1, 1, 1, 17).expand(1, 3, 11, 17).clone()
    x[:, :, :, 1::2] *= 1.3
    constant = torch.full_like(x, 0.4)
    _fixed_controls(operator, detail_bias=-1.0)
    soft = operator(x, x)["image"]
    flat_soft = operator(constant, constant)["image"]
    _fixed_controls(operator, detail_bias=1.0)
    sharp = operator(x, x)["image"]
    flat_sharp = operator(constant, constant)["image"]
    assert (soft - sharp).abs().max() > 1e-4
    assert sharp.diff(dim=-1).abs().mean() > soft.diff(dim=-1).abs().mean()
    torch.testing.assert_close(flat_soft, flat_sharp, atol=2e-6, rtol=2e-6)


def test_base_curve_changes_large_scale_brightness_independently():
    operator = _operator()
    x = torch.full((2, 3, 9, 13), 0.3)
    _fixed_controls(operator, shift_bias=-0.7)
    dark = operator(x, x)
    _fixed_controls(operator, shift_bias=0.7)
    bright = operator(x, x)
    assert torch.all(bright["image"] > dark["image"])
    torch.testing.assert_close(dark["maps"]["detail_gain"], bright["maps"]["detail_gain"])


def test_reconstruction_matches_reported_base_and_detail_controls():
    from tm_library.common import apply_luminance

    operator = _operator()
    _fixed_controls(operator, detail_bias=0.8, contrast_bias=-0.4, shift_bias=0.2)
    rgb = torch.rand(2, 3, 13, 19) * 0.7 + 0.05
    result = operator(rgb, rgb * 0.9)
    maps = result["maps"]
    expected_log_y = maps["base_toned"] + maps["detail_gain"] * maps["detail"]
    expected = apply_luminance(rgb, expected_log_y.clamp(-30, 20).exp())
    torch.testing.assert_close(result["image"], expected)
    torch.testing.assert_close(maps["source_log"], maps["base"] + maps["detail"])
    # One scalar luminance change retains RGB ratios before pipeline gamut handling.
    torch.testing.assert_close(result["image"][:, 0] / result["image"][:, 1], rgb[:, 0] / rgb[:, 1])


def test_decomposition_preserves_step_edges_more_than_box_average():
    operator = _operator(filter_radius=3, filter_eps=0.001)
    x = torch.full((1, 3, 15, 21), 0.1)
    x[:, :, :, 11:] = 0.9
    maps = operator(x, x)["maps"]
    # Adjacent values at the edge should remain separated in the smooth base.
    assert (maps["base"][..., 11] - maps["base"][..., 10]).mean() > 1.8


@pytest.mark.parametrize("shape,value", [((2, 3, 1, 1), 0.0), ((2, 3, 1, 3), 1.0), ((2, 3, 7, 9), None)])
def test_black_white_tiny_odd_batches_have_finite_outputs_and_gradients(shape, value):
    operator = _operator()
    x = (torch.rand(shape) if value is None else torch.full(shape, value)).requires_grad_()
    result = operator(x, x)
    assert result["image"].shape == x.shape
    assert result["features"].shape[:2] == (shape[0], 8)
    assert torch.isfinite(result["image"]).all()
    assert all(torch.isfinite(value).all() for value in result["maps"].values())
    (result["image"].mean() + result["features"].square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in operator.parameters())
    if value == 0.0:
        assert torch.count_nonzero(result["image"]) == 0


def test_exposure_changes_output_and_semantic_confidence_gates_controls():
    torch.manual_seed(11)
    operator = _operator(semantic_mode="explicit")
    x = torch.rand(2, 3, 11, 15) * 0.7 + 0.1
    masks = torch.ones(2, 3, 11, 15)
    baseline = operator(x, x, torch.zeros_like(masks))["image"]
    changed = operator(x * 1.5, x, torch.zeros_like(masks))["image"]
    assert not torch.allclose(baseline, changed)
    with torch.no_grad():
        operator.base_head.weight.fill_(0.03)
        operator.detail_head.weight.fill_(0.03)
    conditioned = operator(x, x, masks, torch.ones(2, 1, 11, 15))["image"]
    unconditioned = operator(x, x, torch.zeros_like(masks))["image"]
    gated = operator(x, x, masks, torch.zeros(2, 1, 11, 15))["image"]
    assert not torch.allclose(conditioned, unconditioned)
    torch.testing.assert_close(gated, unconditioned)


def test_flat_equal_luminance_regions_receive_spatial_semantic_base_shift():
    operator = _operator(semantic_mode="explicit", analysis_size=17, filter_radius=1)
    # Route one semantic channel through the real CNN into only the base shift.
    # This fixture isolates spatial base control: detail remains zero everywhere.
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter.zero_()
        operator.encoder.net[0].weight[0, 6, 1, 1] = 1.
        operator.encoder.net[2].weight[0, 0, 1, 1] = 1.
        operator.base_head.weight[1, 0, 0, 0] = 1.
    gain = torch.full((1, 3, 17, 33), .4)
    semantics = torch.zeros(1, 3, 17, 33)
    semantics[:, 0, :, 17:] = 1.
    prediction = operator.predict(gain, gain, semantics)
    result = operator.render(gain, gain, prediction)
    assert prediction["base_shift"].shape == (1, 1, 17, 17)
    assert result["maps"]["base_shift"].shape == (1, 1, 17, 33)
    assert result["image"][..., 25:].mean() > result["image"][..., :8].mean() + .01
    torch.testing.assert_close(result["maps"]["detail"], torch.zeros_like(result["maps"]["detail"]), atol=2e-6, rtol=0)
    gated = operator(gain, gain, semantics, torch.zeros(1, 1, 17, 33))
    absent = operator(gain, gain)
    torch.testing.assert_close(gated["image"], absent["image"])
    assert result["maps"]["base_shift"][..., 25:].mean() > gated["maps"]["base_shift"][..., 25:].mean()


def test_staged_spatial_controls_match_forward_and_disable_diagnostics():
    operator = _operator(analysis_size=7)
    gain = torch.rand(2, 3, 9, 13).requires_grad_()
    predicted = operator.predict(gain, gain)
    direct = operator(gain, gain)
    staged = operator.render(gain, gain, predicted, return_maps=False)
    torch.testing.assert_close(staged["image"], direct["image"])
    torch.testing.assert_close(staged["features"], direct["features"])
    assert staged["maps"] == {}
    assert predicted["base_contrast"].shape == (2, 1, 7, 7)
    staged["image"].sum().backward()
    assert torch.isfinite(gain.grad).all()


def test_staged_and_forward_gradients_match_for_spatial_base_and_detail_controls():
    operator = _operator(analysis_size=7)
    gain = (torch.rand(1, 3, 5, 7) + .1).requires_grad_()
    parameters = tuple(operator.parameters())
    direct = operator(gain, gain, return_maps=False)["image"]
    staged = operator.render(gain, gain, operator.predict(gain, gain), return_maps=False)["image"]
    direct_grads = torch.autograd.grad(direct.square().mean(), (gain, *parameters))
    staged_grads = torch.autograd.grad(staged.square().mean(), (gain, *parameters))
    for actual, expected in zip(staged_grads, direct_grads):
        torch.testing.assert_close(actual, expected)
