"""Bounded scalar EV rendering and identity-safe staged learning."""

import importlib

import pytest
import torch

from tm_library.common import guided_upsample, luminance
from tm_library.config import ModelConfig


def operator(**kwargs):
    try:
        implementation = importlib.import_module("tm_library.operators.gain_residual")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Bounded EV operator implementation is missing: {exc}")
    torch.manual_seed(31)
    defaults = dict(algorithm="gain_residual", width=8, analysis_size=8)
    defaults.update(kwargs)
    return implementation.GainResidualOperator(ModelConfig(**defaults))


def test_initial_residual_is_exact_identity_for_all_semantic_modes():
    gain = torch.rand(2, 3, 9, 13) * 2
    base = torch.rand_like(gain)
    masks = torch.rand(2, 3, 9, 13)
    for mode in ("none", "train_only", "explicit"):
        op = operator(semantic_mode=mode)
        result = op(gain, base, masks, torch.ones(2, 1, 9, 13))
        assert torch.equal(result["image"], base)
        assert torch.count_nonzero(result["maps"]["lowres_ev"]) == 0
        assert torch.count_nonzero(result["maps"]["ev"]) == 0
        assert torch.equal(result["maps"]["gain"], torch.ones(2, 1, 9, 13))


@pytest.mark.parametrize("bias,scale", [(-100., .5), (100., 2.)])
def test_ev_endpoints_scale_all_rgb_channels_without_clipping(bias, scale):
    op = operator()
    with torch.no_grad():
        op.ev_head.bias.fill_(bias)
    base = torch.tensor([.2, .4, .8])[None, :, None, None].expand(2, 3, 7, 11)
    result = op(torch.full_like(base, .7), base)
    torch.testing.assert_close(result["image"], base * scale, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result["maps"]["gain"],
                               torch.full((2, 1, 7, 11), scale), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result["image"][:, 0] / result["image"][:, 2],
                               torch.full((2, 7, 11), .25))


def test_guided_filter_extrapolation_is_bounded_again_at_full_resolution():
    op = operator(analysis_size=2, filter_radius=1, filter_eps=1e-6, max_ev=.7)
    torch.manual_seed(8)
    gain = torch.rand(2, 3, 11, 13)
    base = torch.rand_like(gain)
    with torch.no_grad():
        op.ev_head.weight.normal_(0, 100)
        op.ev_head.bias.zero_()
    result = op(gain, base)
    low = result["maps"]["lowres_ev"]
    before_clamp = guided_upsample(low, luminance(base), radius=1, eps=1e-6)
    # Establish this fixture exercises extrapolation, so deleting the final
    # clamp cannot accidentally pass just because interpolation stayed bounded.
    assert before_clamp.abs().max() > .7
    assert low.abs().max() <= .7
    assert result["maps"]["ev"].abs().max() <= .7
    assert result["maps"]["gain"].min() >= 2 ** -.7
    assert result["maps"]["gain"].max() <= 2 ** .7


def test_zero_max_ev_remains_identity_even_with_nonzero_head():
    op = operator(max_ev=0)
    with torch.no_grad():
        op.ev_head.weight.fill_(10)
        op.ev_head.bias.fill_(10)
    base = torch.rand(2, 3, 5, 7)
    result = op(base * 2, base)
    assert torch.equal(result["image"], base)
    assert torch.count_nonzero(result["maps"]["ev"]) == 0


def test_first_optimizer_step_learns_head_and_next_backward_reaches_encoder():
    op = operator(semantic_mode="explicit")
    optimizer = torch.optim.SGD(op.parameters(), lr=.1)
    base = torch.rand(2, 3, 9, 13) * .5 + .1
    gain = base * 1.4
    masks = torch.rand(2, 3, 9, 13, requires_grad=True)
    target = base * 1.2
    initial_weight = op.ev_head.weight.detach().clone()
    ((op(gain, base, masks)["image"] - target).square().mean()).backward()
    assert op.ev_head.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in op.parameters())
    assert sum(p.grad.abs().sum() for p in op.encoder.parameters()) == 0
    optimizer.step()
    assert not torch.equal(initial_weight, op.ev_head.weight)
    optimizer.zero_grad()
    masks.grad = None
    ((op(gain, base, masks)["image"] - target).square().mean()).backward()
    assert sum(p.grad.abs().sum() for p in op.encoder.parameters()) > 0
    assert masks.grad is not None and torch.isfinite(masks.grad).all()
    assert masks.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["none", "train_only", "explicit"])
def test_semantic_mode_and_confidence_after_head_leaves_identity(mode):
    op = operator(semantic_mode=mode)
    with torch.no_grad():
        op.ev_head.weight.fill_(.2)
    base = torch.rand(2, 3, 9, 13) * .6 + .1
    masks = torch.ones(2, 3, 9, 13)
    confidence = torch.ones(2, 1, 9, 13)
    absent = op(base * 1.2, base)["image"]
    conditioned = op(base * 1.2, base, masks, confidence)["image"]
    gated = op(base * 1.2, base, masks, confidence * 0)["image"]
    assert torch.equal(absent, gated)
    if mode == "explicit":
        assert (conditioned - absent).abs().max() > 1e-6
    else:
        assert torch.equal(conditioned, absent)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 5), (9, 13)])
@pytest.mark.parametrize("value", [0., 1.])
def test_black_white_odd_tiny_and_batch_inputs_have_finite_maps_and_gradients(shape, value):
    op = operator()
    with torch.no_grad():
        op.ev_head.weight.fill_(.1)
    base = torch.full((2, 3, *shape), value, requires_grad=True)
    result = op(base * 1.5, base)
    assert result["image"].shape == base.shape
    assert result["features"].shape == (2, 8, 8, 8)
    assert result["maps"]["lowres_ev"].shape == (2, 1, 8, 8)
    for tensor in (result["image"], result["features"], *result["maps"].values()):
        assert torch.isfinite(tensor).all()
    result["image"].sum().backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()


def test_staged_ev_gate_scales_exposure_before_exponentiation_and_can_preserve_anchor():
    op = operator(max_ev=1.2)
    with torch.no_grad():
        op.ev_head.bias.fill_(.7)
    base = torch.rand(2, 3, 9, 13) + .1
    predicted = op.predict(base * 2, base)
    direct = op(base * 2, base)
    staged = op.render(base * 2, base, predicted, return_maps=False)
    torch.testing.assert_close(staged["image"], direct["image"])
    assert staged["maps"] == {}
    half = dict(predicted, ev_scale=torch.full((2, 1, 9, 13), .5))
    gated = op.render(base * 2, base, half, return_maps=False)
    torch.testing.assert_close(gated["image"] / base, (direct["image"] / base).sqrt())
    zero = dict(predicted, ev_scale=torch.zeros(2, 1, 9, 13))
    assert torch.equal(op.render(base * 2, base, zero, return_maps=False)["image"], base)


def test_staged_and_forward_gradients_match_after_residual_head_learns():
    op = operator()
    with torch.no_grad():
        op.ev_head.weight.fill_(.2)
    base = torch.rand(1, 3, 5, 7).requires_grad_()
    parameters = tuple(op.parameters())
    direct = op(base * 2, base, return_maps=False)["image"]
    staged = op.render(base * 2, base, op.predict(base * 2, base), return_maps=False)["image"]
    direct_grads = torch.autograd.grad(direct.square().mean(), (base, *parameters))
    staged_grads = torch.autograd.grad(staged.square().mean(), (base, *parameters))
    for actual, expected in zip(staged_grads, direct_grads):
        torch.testing.assert_close(actual, expected)
