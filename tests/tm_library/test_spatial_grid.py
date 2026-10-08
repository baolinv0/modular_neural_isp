"""Bilateral-grid mechanism checks, including its third (guidance) axis."""

import importlib

import pytest
import torch


def _implementation():
    try:
        return importlib.import_module("tm_library.operators.spatial_grid")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Spatial bilateral-grid implementation is missing: {exc}")


def _operator(**kwargs):
    from tm_library.config import ModelConfig

    config = ModelConfig(algorithm="spatial_grid", width=8, analysis_size=8,
                         grid_size=3, grid_depth=4, **kwargs)
    return _implementation().SpatialGridOperator(config)


def test_slice_coordinates_use_xy_and_guidance_depth():
    # Independent ramp: value = x + 10*y + 100*z at each lattice vertex.
    grid = torch.tensor([[[[[0., 1.], [10., 11.]],
                           [[100., 101.], [110., 111.]]]]])
    guide = torch.tensor([[[[0., .25], [.75, 1.]]]])
    actual = _implementation().slice_bilateral_grid(grid, guide)
    torch.testing.assert_close(actual, torch.tensor([[[[0., 26.], [85., 111.]]]]))


def test_depth_slice_has_nonzero_grid_and_guidance_gradients():
    grid = torch.tensor([[[[[0.]], [[1.]]]]], requires_grad=True)
    guide = torch.tensor([[[[.2, .7]]]], requires_grad=True)
    actual = _implementation().slice_bilateral_grid(grid, guide)
    torch.testing.assert_close(actual, guide)
    actual.sum().backward()
    torch.testing.assert_close(guide.grad, torch.ones_like(guide))
    torch.testing.assert_close(grid.grad.flatten(), torch.tensor([1.1, .9]))


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (17, 13)])
@pytest.mark.parametrize("value", [0., 1.])
def test_grid_handles_black_white_odd_tiny_and_batch_two(shape, value):
    op = _operator()
    gain = torch.full((2, 3, *shape), value)
    result = op(gain, gain)
    assert result["image"].shape == gain.shape
    assert result["features"].shape[:2] == (2, 8)
    maps = result["maps"]
    assert maps["grid"].shape == (2, 5, 4, 3, 3)
    assert maps["guidance"].shape == (2, 1, *shape)
    assert maps["coefficients"].shape == (2, 5, *shape)
    for tensor in [result["image"], result["features"], *maps.values()]:
        assert torch.isfinite(tensor).all()
    coefficients = maps["coefficients"]
    assert ((coefficients[:, :1] >= 0) & (coefficients[:, :1] <= 1)).all()
    assert (coefficients[:, 1:4] > 0).all()
    assert ((coefficients[:, 4:] >= .25) & (coefficients[:, 4:] <= 4)).all()


def test_gate_can_select_both_gtm_and_local_tone_endpoints():
    op = _operator()
    gain = torch.full((1, 3, 5, 7), .4)
    base = torch.full_like(gain, .8)
    with torch.no_grad():
        op.grid_head.weight.zero_()
        op.grid_head.bias.zero_()
        op.grid_head.bias[:op.grid_depth].fill_(-1000)
    torch.testing.assert_close(op(gain, base)["image"], base)
    with torch.no_grad():
        op.grid_head.bias[:op.grid_depth].fill_(1000)
    result = op(gain, base)
    assert torch.equal(result["maps"]["coefficients"][:, :1], torch.ones_like(base[:, :1]))
    assert not torch.allclose(result["image"], base)


def test_output_uses_local_gain_and_upstream_curve():
    from photofinishing.photofinishing_model import PhotofinishingModule

    op = _operator()
    gain = torch.full((1, 3, 5, 7), .2)
    base = torch.full_like(gain, .6)
    result = op(gain, base)
    w, a, b, c, local_gain = result["maps"]["coefficients"].split(1, dim=1)
    expected_local = PhotofinishingModule.apply_tm(gain * local_gain, a, b, c)
    torch.testing.assert_close(result["image"], (1 - w) * base + w * expected_local)


def test_render_loss_reaches_grid_and_guidance_heads():
    torch.manual_seed(13)
    op = _operator()
    gain = torch.rand(2, 3, 7, 9)
    result = op(gain, gain * .7)
    result["image"].square().mean().backward()
    for head in (op.grid_head, op.guidance_head):
        gradients = [p.grad for p in head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_explicit_semantics_influence_render_and_zero_confidence_suppresses_them():
    torch.manual_seed(17)
    op = _operator(semantic_mode="explicit")
    gain = torch.rand(2, 3, 9, 7)
    base = gain * .7
    masks = torch.ones(2, 3, 9, 7)
    zero = torch.zeros(2, 1, 9, 7)
    one = torch.ones_like(zero)
    absent = op(gain, base)["image"]
    suppressed = op(gain, base, masks, zero)["image"]
    conditioned = op(gain, base, masks, one)["image"]
    torch.testing.assert_close(absent, suppressed, rtol=0, atol=0)
    assert not torch.allclose(absent, conditioned, rtol=0, atol=1e-7)


def test_prediction_runs_feature_cnn_only_at_analysis_resolution():
    op = _operator()
    seen = []
    hooks = [layer.register_forward_pre_hook(lambda layer, inputs: seen.append(inputs[0].shape))
             for layer in op.encoder.modules() if isinstance(layer, torch.nn.Conv2d)]
    try:
        gain = torch.rand(2, 3, 37, 29)
        controls = op.predict(gain, gain * .7)
        assert controls["features"].shape == (2, 8, 8, 8)
        assert controls["guide_weight"].shape == (1, 6, 1, 1)
        assert controls["guide_bias"].shape == (1,)
        assert seen and all(shape[-2:] == (8, 8) for shape in seen)
        seen.clear()
        op.render(gain, gain * .7, controls)
        assert not seen
        assert sum(p.numel() for p in op.guidance_head.parameters()) == 7
    finally:
        for hook in hooks:
            hook.remove()


def test_staged_grid_render_matches_forward_outputs_and_gradients():
    import copy

    torch.manual_seed(29)
    direct_op = _operator()
    staged_op = copy.deepcopy(direct_op)
    direct_gain = torch.rand(2, 3, 11, 17, requires_grad=True)
    staged_gain = direct_gain.detach().clone().requires_grad_()
    direct = direct_op(direct_gain, direct_gain * .7)
    staged_base = staged_gain * .7
    controls = staged_op.predict(staged_gain, staged_base)
    staged = staged_op.render(staged_gain, staged_base, controls)
    torch.testing.assert_close(staged["image"], direct["image"], rtol=0, atol=0)
    for key in direct["maps"]:
        torch.testing.assert_close(staged["maps"][key], direct["maps"][key], rtol=0, atol=0)
    direct["image"].square().mean().backward()
    staged["image"].square().mean().backward()
    torch.testing.assert_close(staged_gain.grad, direct_gain.grad, rtol=0, atol=0)
    for direct_p, staged_p in zip(direct_op.parameters(), staged_op.parameters()):
        assert direct_p.grad is not None and staged_p.grad is not None
        torch.testing.assert_close(staged_p.grad, direct_p.grad, rtol=0, atol=0)
    assert staged_op.render(staged_gain, staged_base, controls, return_maps=False)["maps"] == {}
    without_maps = direct_op(direct_gain, direct_gain * .7, return_maps=False)
    assert without_maps["maps"] == {}
    torch.testing.assert_close(without_maps["image"], direct["image"], rtol=0, atol=0)
