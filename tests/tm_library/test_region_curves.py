"""Mechanism tests for spatial mixtures of shared-RGB monotonic curves."""

import pytest
import torch

from tm_library.config import ModelConfig
from tm_library.operators.region_curves import RegionCurveOperator


def make_operator():
    torch.manual_seed(7)
    return RegionCurveOperator(ModelConfig(
        algorithm="region_curves", semantic_mode="explicit", width=8,
        analysis_size=16, curve_bins=8, num_experts=3,
    ))


@pytest.mark.parametrize("shape,value", [((2, 3, 1, 1), 0.0),
                                          ((2, 3, 3, 5), 1.0),
                                          ((2, 3, 9, 13), 2.0)])
def test_extreme_tiny_odd_inputs_have_finite_normalized_monotonic_maps(shape, value):
    operator = make_operator()
    gain = torch.full(shape, value)
    result = operator(gain, gain.clamp(0, 1))
    curves, weights = result["maps"]["curves"], result["maps"]["weights"]
    assert curves.shape == (shape[0], 3, 8)
    assert result["image"].shape == shape
    assert result["features"].shape[:2] == (shape[0], 8)
    assert weights.shape == (shape[0], 3, *shape[-2:])
    assert torch.isfinite(result["image"]).all()
    assert torch.isfinite(curves).all()
    assert (curves[..., 1:] > curves[..., :-1]).all()
    torch.testing.assert_close(curves[..., 0], torch.zeros_like(curves[..., 0]))
    torch.testing.assert_close(curves[..., -1], torch.ones_like(curves[..., -1]))
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    if value in (0.0, 1.0):
        torch.testing.assert_close(result["image"], gain)


def test_shared_rgb_curve_preserves_neutral_pixels_and_applies_to_gain():
    operator = make_operator()
    gain = torch.full((2, 3, 7, 11), 0.35)
    result = operator(gain, torch.full_like(gain, 0.8))
    image = result["image"]
    torch.testing.assert_close(image[:, 0], image[:, 1])
    torch.testing.assert_close(image[:, 1], image[:, 2])
    assert not torch.allclose(image, torch.full_like(gain, 0.8))


def test_image_features_produce_spatially_varying_expert_weights():
    operator = make_operator()
    gain = torch.full((2, 3, 13, 15), 0.1)
    gain[..., 7:] = 0.8
    weights = operator(gain, gain)["maps"]["weights"]
    assert (weights[..., :4].mean((-2, -1)) - weights[..., 10:].mean((-2, -1))).abs().max() > 1e-5


def test_minority_semantic_region_changes_output_but_zero_confidence_suppresses_it():
    operator = make_operator()
    gain = torch.full((2, 3, 13, 15), 0.35)
    empty = torch.zeros(2, 3, 13, 15)
    minority = empty.clone()
    minority[:, 1, 4:9, 6:10] = 1.0
    confidence = torch.ones(2, 1, 13, 15)
    unconditioned = operator(gain, gain, empty, confidence)
    conditioned = operator(gain, gain, minority, confidence)
    assert (conditioned["image"] - unconditioned["image"]).abs().max() > 1e-6
    delta = conditioned["maps"]["weights"] - unconditioned["maps"]["weights"]
    assert delta[..., 4:9, 6:10].abs().max() > 1e-5
    assert delta.flatten(2).std(-1).max() > 1e-5
    gated = operator(gain, gain, minority, torch.zeros_like(confidence))
    absent = operator(gain, gain)
    torch.testing.assert_close(gated["image"], absent["image"])
    torch.testing.assert_close(gated["maps"]["weights"], absent["maps"]["weights"])


def test_curve_bank_and_mixture_weights_both_change_actual_output():
    operator = make_operator()
    gain = torch.full((2, 3, 7, 11), 0.4)
    with torch.no_grad():
        operator.weight_head.weight.zero_()
        operator.weight_head.bias.copy_(torch.tensor([8.0, -8.0, -8.0]))
    first = operator(gain, gain)["image"]
    with torch.no_grad():
        operator.weight_head.bias.copy_(torch.tensor([-8.0, -8.0, 8.0]))
    last = operator(gain, gain)["image"]
    assert (first - last).abs().max() > 0.01
    with torch.no_grad():
        operator.curve_logits[-1, :3].add_(2.0)
    changed = operator(gain, gain)["image"]
    assert (changed - last).abs().max() > 0.01


def test_image_loss_backpropagates_to_inputs_curves_weights_and_encoder():
    operator = make_operator()
    gain = (torch.rand(2, 3, 9, 13) * 0.8 + 0.1).requires_grad_()
    semantics = torch.rand(2, 3, 9, 13, requires_grad=True)
    result = operator(gain, gain * 0.8, semantics, torch.ones(2, 1, 9, 13))
    result["image"].square().mean().backward()
    for value in (gain.grad, semantics.grad, operator.curve_logits.grad,
                  operator.weight_head.weight.grad):
        assert value is not None and torch.isfinite(value).all()
        assert value.abs().sum() > 0
    encoder_grads = [p.grad for p in operator.encoder.parameters() if p.grad is not None]
    assert encoder_grads and sum(g.abs().sum() for g in encoder_grads) > 0
