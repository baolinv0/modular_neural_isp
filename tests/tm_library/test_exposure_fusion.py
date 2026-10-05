"""Mechanism and differentiation checks for genuine synthetic-exposure fusion."""

import importlib

import pytest
import torch


def fusion_module():
    return importlib.import_module("tm_library.operators.exposure_fusion")


@pytest.mark.parametrize("shape", [(2, 3, 17, 23), (2, 3, 1, 1), (2, 3, 2, 3), (1, 3, 1, 9)])
def test_laplacian_pyramid_reconstructs_original(shape):
    module = fusion_module()
    source = torch.rand(shape)
    pyramid = module.laplacian_pyramid(source, levels=7)
    rebuilt = module.reconstruct_pyramid(pyramid)
    torch.testing.assert_close(rebuilt, source, atol=2e-7, rtol=2e-6)
    assert len(pyramid) <= 7


@pytest.mark.parametrize("selected", [0, 1, 2])
def test_one_hot_weights_select_an_entire_exposure(selected):
    module = fusion_module()
    exposures = torch.rand(2, 3, 3, 19, 15)
    weights = torch.zeros(2, 3, 19, 15)
    weights[:, selected] = 1
    fused, scale_weights = module.multiscale_fusion(exposures, weights, levels=4)
    torch.testing.assert_close(fused, exposures[:, selected], atol=3e-7, rtol=2e-6)
    for scale in scale_weights:
        torch.testing.assert_close(scale.sum(dim=1), torch.ones_like(scale[:, 0]))
        assert (scale >= 0).all()


def test_structured_scene_fusion_is_different_from_pointwise_average():
    module = fusion_module()
    x = torch.linspace(0, 1, 33)[None, None, None, :].expand(1, 3, 29, 33)
    exposures = torch.stack((x.square(), x.sqrt()), dim=1)
    weights = torch.zeros(1, 2, 29, 33)
    weights[:, 0, :, :16] = 1
    weights[:, 1, :, 16:] = 1
    pointwise = (exposures * weights[:, :, None]).sum(dim=1)
    fused, scales = module.multiscale_fusion(exposures, weights, levels=4)
    assert len(scales) == 4
    assert (fused - pointwise).abs().max() > 0.01
    assert torch.isfinite(fused).all()


def make_operator(**kwargs):
    from tm_library.config import ModelConfig

    return fusion_module().ExposureFusionOperator(ModelConfig(algorithm="exposure_fusion", **kwargs))


def test_exposures_preserve_above_one_input_and_apply_ev_before_compression():
    operator = make_operator(width=8, analysis_size=16)
    gain = torch.full((2, 3, 9, 11), 2.0)
    result = operator(gain, gain / (1 + gain))
    exposure_ev = torch.tensor(operator.config.exposures)
    radiance = gain[:, None] * torch.exp2(exposure_ev)[None, :, None, None, None]
    torch.testing.assert_close(result["maps"]["exposures"], radiance / (1 + radiance))
    brighter = operator(gain * 2, gain / (1 + gain))["maps"]["exposures"]
    assert (brighter > result["maps"]["exposures"]).all()


@pytest.mark.parametrize("shape", [(17, 23), (1, 1), (2, 3), (1, 9)])
def test_black_white_batch_two_has_finite_output_and_gradients(shape):
    operator = make_operator(width=8, analysis_size=16, pyramid_levels=6)
    gain = torch.stack((torch.zeros(3, *shape), torch.ones(3, *shape))).requires_grad_()
    base = gain / (1 + gain)
    result = operator(gain, base)
    assert result["image"].shape == gain.shape
    assert result["features"].shape[:2] == (2, 8)
    assert torch.isfinite(result["image"]).all()
    assert result["image"][0].abs().max() == 0
    assert result["image"][1].min() >= 0
    assert result["image"][1].max() <= 1
    torch.testing.assert_close(result["maps"]["weights"].sum(dim=1), torch.ones_like(gain[:, 0]))
    for key, scale in result["maps"].items():
        if not key.startswith("weights_scale_"):
            continue
        torch.testing.assert_close(scale.sum(dim=1), torch.ones_like(scale[:, 0]))
    result["image"].square().mean().backward()
    assert torch.isfinite(gain.grad).all()
    gradients = [p.grad for p in operator.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_explicit_semantics_change_weights_and_zero_confidence_suppresses_them():
    torch.manual_seed(3)
    operator = make_operator(width=8, analysis_size=16, semantic_mode="explicit")
    gain = torch.rand(2, 3, 13, 15)
    base = gain / (1 + gain)
    masks = torch.ones(2, 3, 13, 15)
    blank = operator(gain, base, torch.zeros_like(masks))["maps"]["weights"]
    masked = operator(gain, base, masks)["maps"]["weights"]
    suppressed = operator(gain, base, masks, torch.zeros(2, 1, 13, 15))["maps"]["weights"]
    assert not torch.allclose(blank, masked)
    torch.testing.assert_close(blank, suppressed)


def test_operator_renders_pyramid_result_and_reports_only_tensor_maps():
    torch.manual_seed(11)
    operator = make_operator(width=8, analysis_size=32, pyramid_levels=4)
    x = torch.linspace(0, 2, 33)[None, None, None, :].expand(2, 3, 29, 33)
    result = operator(x, x / (1 + x))
    expected, scales = fusion_module().multiscale_fusion(result["maps"]["exposures"], result["maps"]["weights"], 4)
    torch.testing.assert_close(result["image"], expected)
    pointwise = (result["maps"]["exposures"] * result["maps"]["weights"][:, :, None]).sum(dim=1)
    assert (result["image"] - pointwise).abs().max() > 1e-6
    assert all(isinstance(value, torch.Tensor) for value in result["maps"].values())
    for index, scale in enumerate(scales):
        torch.testing.assert_close(result["maps"][f"weights_scale_{index}"], scale)
