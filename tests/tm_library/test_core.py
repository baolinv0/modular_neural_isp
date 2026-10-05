"""Shared contracts and original pretrained photofinishing compatibility."""
from pathlib import Path

import pytest
import torch

from tm_library.config import ModelConfig
from tm_library.common import (ConditionEncoder, apply_luminance, evaluate_curves,
                               guided_upsample, luminance, monotonic_curves,
                               tone_curve, total_variation)
from tm_library.model import TMModel
from photofinishing.photofinishing_model import PhotofinishingModule

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'photofinishing/models/photofinishing_s24-style-0.pth'


@pytest.mark.parametrize('kwargs', [
    {'algorithm': 'unknown'}, {'semantic_mode': 'invalid'},
    {'algorithm': 'baseline', 'semantic_mode': 'explicit'},
    {'width': 0}, {'width': True}, {'analysis_size': 0}, {'curve_bins': 1},
    {'num_experts': 0}, {'grid_depth': 1}, {'grid_size': 1},
    {'max_ev': -1}, {'max_ev': float('nan')}, {'exposures': []},
    {'exposures': [float('inf')]}, {'exposures': [0, 0]},
    {'pyramid_levels': 0}, {'filter_eps': 0}, {'filter_radius': -1},
    {'semantic_channels': 0}, {'freeze_backbone': 'yes'},
])
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ModelConfig(**kwargs)


def test_config_roundtrip():
    cfg = ModelConfig(algorithm='gain_residual', exposures=[-2, 0, 2])
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.exposures == (-2., 0., 2.)
    with pytest.raises(ValueError, match='unknown'):
        ModelConfig.from_dict({'typo': 5})


@pytest.mark.parametrize('max_ev', [16.0001, 128.])
def test_config_rejects_ev_limits_that_can_overflow_exposure_gains(max_ev):
    with pytest.raises(ValueError, match=r'max_ev.*\[0,16\]'):
        ModelConfig(algorithm='gain_residual', max_ev=max_ev)


@pytest.mark.parametrize('max_ev', [0., 16.])
def test_supported_ev_endpoints_render_finite_scalar_gains(max_ev):
    from tm_library.operators.gain_residual import GainResidualOperator

    operator = GainResidualOperator(ModelConfig(
        algorithm='gain_residual', max_ev=max_ev, width=4, analysis_size=4))
    with torch.no_grad():
        operator.ev_head.bias.fill_(100.)
    image = torch.ones(2, 3, 3, 5)
    result = operator(image, image)
    assert result['maps']['gain'].isfinite().all()
    assert result['image'].isfinite().all()
    torch.testing.assert_close(result['maps']['gain'],
                               torch.full((2, 1, 3, 5), 2 ** max_ev))


def test_curves_endpoints_monotonic_interpolation_and_gradients():
    logits = torch.randn(2, 3, 7, requires_grad=True)
    curves = monotonic_curves(logits)
    assert curves.shape == (2, 3, 8)
    assert torch.equal(curves[..., 0], torch.zeros(2, 3))
    assert torch.equal(curves[..., -1], torch.ones(2, 3))
    assert (curves.diff(dim=-1) > 0).all()
    x = torch.tensor([0., .5, 1.]).reshape(1, 1, 1, 3).expand(2, 3, 1, 3)
    result = evaluate_curves(x, curves)
    assert result.shape == (2, 3, 3, 1, 3)
    assert torch.equal(result[..., 0], torch.zeros_like(result[..., 0]))
    assert torch.equal(result[..., -1], torch.ones_like(result[..., -1]))
    result.sum().backward()
    assert logits.grad.isfinite().all() and logits.grad.abs().sum() > 0
    identity = torch.linspace(0, 1, 8).reshape(1, 1, 8).expand(2, 3, 8)
    torch.testing.assert_close(evaluate_curves(x, identity), x[:, None].expand(2, 3, 3, 1, 3))


def test_condition_encoder_confidence_and_modes():
    gain, base = torch.rand(2, 3, 7, 11), torch.rand(2, 3, 7, 11)
    masks, zero = torch.rand(2, 3, 7, 11), torch.zeros(2, 1, 7, 11)
    cfg = ModelConfig(algorithm='gain_residual', semantic_mode='explicit', analysis_size=6)
    encoder = ConditionEncoder(cfg)
    torch.testing.assert_close(encoder(gain, base), encoder(gain, base, masks, zero))
    assert encoder(gain, base).shape == (2, cfg.width, 6, 6)
    assert not torch.equal(encoder(gain, base), encoder(gain, base, masks))
    for mode in ('none', 'train_only'):
        encoder = ConditionEncoder(ModelConfig(algorithm='gain_residual', semantic_mode=mode))
        torch.testing.assert_close(encoder(gain, base), encoder(gain, base, masks))


@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (7, 11)])
def test_scalar_helpers_tiny_finite_and_gradients(shape):
    rgb = torch.rand(2, 3, *shape, requires_grad=True)
    y = luminance(rgb)
    restored = apply_luminance(rgb, y)
    torch.testing.assert_close(restored, rgb)
    low = torch.full((2, 2, 2, 3), .3, requires_grad=True)
    up = guided_upsample(low, y, radius=3)
    assert up.shape == (2, 2, *shape)
    torch.testing.assert_close(up, torch.full_like(up, .3), atol=1e-6, rtol=1e-5)
    loss = restored.mean() + up.mean() + total_variation(rgb)
    loss.backward()
    assert rgb.grad.isfinite().all() and low.grad.isfinite().all()
    black = apply_luminance(torch.zeros_like(rgb), torch.ones_like(y))
    assert black.isfinite().all()


def test_tone_curve_matches_upstream():
    x = torch.linspace(0, 2, 60).reshape(2, 3, 2, 5)
    a, b, c = (torch.rand(2, 1, 2, 5) + .1 for _ in range(3))
    torch.testing.assert_close(tone_curve(x, a, b, c), PhotofinishingModule.apply_tm(x, a, b, c))


def test_pretrained_baseline_equality_and_strict_loading():
    torch.manual_seed(20)
    model = TMModel(ModelConfig(algorithm='baseline')).eval()
    model.load_upstream(WEIGHTS)
    upstream = PhotofinishingModule(device=torch.device('cpu')).eval()
    upstream.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    x = torch.rand(2, 3, 25, 29)
    with torch.no_grad():
        actual, expected = model(x), upstream(x, training_mode=True)
    torch.testing.assert_close(actual['output'], expected['output'], rtol=0, atol=0)
    torch.testing.assert_close(actual['linear'], expected['lsrgb_ltm'], rtol=0, atol=0)
    state = upstream.state_dict()
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError):
        model.load_upstream(state)


def test_model_color_matrices_follow_dtype_and_checkpoint_roundtrip():
    model = TMModel(ModelConfig(algorithm='baseline')).double()
    assert model.backbone._rgb_to_ycbcr_matrix.dtype == torch.float64
    assert model.backbone._ycbcr_to_rgb_matrix.dtype == torch.float64
    assert not any('matrix' in key for key in model.state_dict())
    payload = model.checkpoint()
    clone = TMModel.from_checkpoint(payload)
    assert clone.config == model.config
    for key, value in clone.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key].float())


@pytest.mark.parametrize('mode', ['none', 'train_only'])
def test_s0_s1_renderer_and_inference_never_use_supplied_masks(mode):
    model = TMModel(ModelConfig(algorithm='region_curves', semantic_mode=mode,
                               width=4, analysis_size=8))
    model.load_upstream(WEIGHTS)
    image = torch.rand(2, 3, 9, 11)
    masks = torch.rand(2, 3, 9, 11)
    no_masks, supplied = model(image), model(image, masks, torch.ones(2, 1, 9, 11))
    torch.testing.assert_close(supplied['output'], no_masks['output'], rtol=0, atol=0)
    if mode == 'train_only':
        assert supplied['semantic_logits'].shape == masks.shape
        torch.testing.assert_close(supplied['semantic_logits'], no_masks['semantic_logits'], rtol=0, atol=0)
        model.zero_grad(set_to_none=True)
        supplied['semantic_logits'].square().mean().backward()
        assert model.operator.encoder.net[0].weight.grad.abs().sum() > 0
        assert model.semantic_head.weight.grad.abs().sum() > 0
    else:
        assert 'semantic_logits' not in supplied
    model.eval()
    with torch.no_grad():
        absent, present = model(image), model(image, masks)
    assert 'semantic_logits' not in absent
    torch.testing.assert_close(absent['output'], present['output'], rtol=0, atol=0)


def test_s2_semantics_change_rendering_and_zero_confidence_is_absent():
    model = TMModel(ModelConfig(algorithm='region_curves', semantic_mode='explicit',
                               width=4, analysis_size=8)).eval()
    model.load_upstream(WEIGHTS)
    image = torch.full((2, 3, 7, 9), .3)
    masks = torch.ones(2, 3, 7, 9)
    with torch.no_grad():
        absent = model(image)
        enabled = model(image, masks, torch.ones(2, 1, 7, 9))
        gated = model(image, masks, torch.zeros(2, 1, 7, 9))
    assert (absent['linear'] - enabled['linear']).abs().max() > 1e-7
    torch.testing.assert_close(gated['output'], absent['output'], rtol=0, atol=0)
    assert 'semantic_logits' not in enabled


@pytest.mark.parametrize('value', [0., 1.])
def test_baseline_tiny_extremes_are_finite(value):
    model = TMModel(ModelConfig()).eval()
    model.load_upstream(WEIGHTS)
    with torch.no_grad():
        result = model(torch.full((2, 3, 1, 3), value))
    assert result['output'].shape == (2, 3, 1, 3)
    assert result['output'].isfinite().all()


@pytest.mark.parametrize('bad_confidence', [torch.zeros(1, 3, 4, 5), torch.full((1, 1, 4, 5), float('nan'))])
def test_explicit_confidence_contract_also_applies_with_missing_semantics(bad_confidence):
    encoder = ConditionEncoder(ModelConfig(algorithm='region_curves', semantic_mode='explicit'))
    image = torch.rand(1, 3, 4, 5)
    with pytest.raises(ValueError, match='confidence'):
        encoder(image, image, confidence=bad_confidence)


@pytest.mark.parametrize('pretrained', [False, True])
@pytest.mark.parametrize('algorithm', ['baseline', 'region_curves', 'spatial_grid',
                                       'gain_residual', 'exposure_fusion', 'base_detail'])
def test_saturated_rgb_primaries_have_finite_real_backward(algorithm, pretrained):
    torch.manual_seed(3)
    model = TMModel(ModelConfig(algorithm=algorithm, width=4, analysis_size=8))
    if pretrained:
        model.load_upstream(WEIGHTS)
    primaries = torch.eye(3).reshape(3, 3, 1, 1).expand(3, 3, 21, 25)
    image = torch.cat((primaries, torch.zeros(1, 3, 21, 25),
                       torch.ones(1, 3, 21, 25))).clone().requires_grad_()
    result = model(image)
    assert result['output'].isfinite().all()
    result['output'].mean().backward()
    assert image.grad is not None and image.grad.isfinite().all()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    assert gradients
    assert all(gradient.isfinite().all() for gradient in gradients.values()), [
        name for name, gradient in gradients.items() if not gradient.isfinite().all()]


def test_histogram_zero_bin_backward_is_finite_and_forward_is_exact():
    from tm_library.model import _StableLuTNet
    from photofinishing.photofinishing_model import LuTNet
    cbcr = torch.tensor([-.5, .5]).reshape(1, 2, 1, 1).expand(1, 2, 3, 5).clone().requires_grad_()
    root = _StableLuTNet._differentiable_cbcr_histogram(cbcr)
    original = LuTNet._differentiable_cbcr_histogram(cbcr)
    assert (root == 0).any(), 'fixture must exercise actual float32 Gaussian underflow'
    torch.testing.assert_close(root, original, atol=0, rtol=0)
    root.sum().backward()
    assert cbcr.grad.isfinite().all()
    assert cbcr.grad.abs().sum() > 0
