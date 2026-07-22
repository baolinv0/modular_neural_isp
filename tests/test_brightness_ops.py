import torch

from photofinishing.brightness_ops import (
    apply_luminance_scale,
    linear_rgb_to_luminance,
    srgb_eotf,
    srgb_oetf,
)


def test_srgb_oetf_breakpoint_and_roundtrip():
    x = torch.tensor([0.0, 0.0031308, 0.18, 1.0])
    y = srgb_oetf(x)
    assert torch.isclose(y[1], torch.tensor(12.92 * 0.0031308), atol=1e-6)
    assert torch.all(y[1:] >= y[:-1])
    assert torch.allclose(srgb_eotf(y), x, atol=2e-6)


def test_apply_luminance_scale_preserves_rgb_ratios_without_clipping():
    rgb = torch.tensor([[[[0.10]], [[0.20]], [[0.30]]]])
    target_y = linear_rgb_to_luminance(rgb) * 1.5
    out = apply_luminance_scale(rgb, target_y, scale_min=0.1, scale_max=4.0)
    assert torch.allclose(out[:, 0] / out[:, 1], rgb[:, 0] / rgb[:, 1], atol=1e-6)
    assert torch.allclose(out[:, 2] / out[:, 1], rgb[:, 2] / rgb[:, 1], atol=1e-6)
