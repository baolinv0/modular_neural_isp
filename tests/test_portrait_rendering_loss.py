from __future__ import annotations

import importlib

import pytest
import torch


def _mod():
    try:
        return importlib.import_module("portrait_rendering_loss")
    except ModuleNotFoundError as exc:
        pytest.fail(f"portrait_rendering_loss module is not implemented yet: {exc}")


def _masks(h: int = 8, w: int = 8):
    face = torch.zeros(1, 1, h, w)
    face[:, :, 2:6, 2:6] = 1.0
    skin = face.clone()
    background = 1.0 - face
    return face, skin, background


def _constant_rgb(rgb: tuple[float, float, float], h: int = 8, w: int = 8):
    image = torch.empty(1, 3, h, w)
    for c, value in enumerate(rgb):
        image[:, c] = value
    return image


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def test_identity_has_near_zero_all_losses():
    m = _mod()
    face, skin, background = _masks()
    image = _constant_rgb((0.45, 0.40, 0.35))
    loss_fn = m.PortraitRenderingLoss(quantile_mode="hard")

    out = loss_fn(image, image.clone(), face, skin, background)

    for key in ("loss", "loss_exp", "loss_color", "loss_tone", "loss_fb"):
        assert out[key].item() == pytest.approx(0.0, abs=1e-6)


def test_global_face_exposure_shift_mainly_changes_exposure_not_tone():
    m = _mod()
    face, skin, background = _masks()
    reference = _constant_rgb((0.35, 0.35, 0.35))
    candidate = reference.clone()
    candidate = torch.where(face.bool().expand_as(candidate), torch.full_like(candidate, 0.48), candidate)

    out = m.PortraitRenderingLoss(quantile_mode="hard")(
        candidate, reference, face, skin, background
    )

    assert out["loss_exp"].item() > 0.2
    assert out["loss_tone"].item() < 1e-5


def test_chromaticity_shift_with_preserved_linear_luminance_targets_color():
    m = _mod()
    face, skin, background = _masks()

    ref_lin = torch.tensor([0.21404114, 0.21404114, 0.21404114])
    cand_lin = ref_lin.clone()
    cand_lin[0] += 0.05
    cand_lin[2] -= 0.05
    cand_lin[1] -= (0.2126 * 0.05 - 0.0722 * 0.05) / 0.7152
    ref_rgb = _linear_to_srgb(ref_lin)
    cand_rgb = _linear_to_srgb(cand_lin)

    reference = _constant_rgb(tuple(float(x) for x in ref_rgb))
    candidate = reference.clone()
    face_rgb = cand_rgb.view(1, 3, 1, 1).expand_as(candidate)
    candidate = torch.where(face.bool().expand_as(candidate), face_rgb, candidate)

    out = m.PortraitRenderingLoss(quantile_mode="hard")(
        candidate, reference, face, skin, background
    )

    assert out["loss_color"].item() > 1.0
    assert out["loss_exp"].item() < 1e-4
    assert out["loss_tone"].item() < 1e-5


def test_face_tone_flattening_changes_tone_with_median_exposure_preserved():
    m = _mod()
    face, skin, background = _masks()
    reference = _constant_rgb((0.30, 0.30, 0.30))

    levels_ref = torch.tensor([0.18, 0.28, 0.42, 0.62])
    levels_cand = torch.tensor([0.30, 0.35, 0.40, 0.45])
    for row, (rv, cv) in enumerate(zip(levels_ref, levels_cand, strict=True), start=2):
        reference[:, :, row, 2:6] = rv

    # Shift candidate log-luminance so its face median matches reference.
    candidate = reference.clone()
    for row, cv in enumerate(levels_cand, start=2):
        candidate[:, :, row, 2:6] = cv

    # Explicitly normalize candidate face luminance to the reference face median in log2 space.
    ref_y = m.rgb_to_luminance(reference)
    cand_y = m.rgb_to_luminance(candidate)
    ref_med = m.masked_quantile(torch.log2(ref_y + 1e-6), face, 0.5, mode="hard")
    cand_med = m.masked_quantile(torch.log2(cand_y + 1e-6), face, 0.5, mode="hard")
    scale = torch.pow(2.0, ref_med - cand_med).view(1, 1, 1, 1)
    cand_linear = m.srgb_to_linear(candidate) * scale
    candidate = m.linear_to_srgb(cand_linear.clamp(0.0, 1.0))

    out = m.PortraitRenderingLoss(quantile_mode="hard")(
        candidate, reference, face, skin, background
    )

    assert out["loss_exp"].item() < 1e-4
    assert out["loss_tone"].item() > 0.15


def test_background_only_shift_targets_face_background_relation():
    m = _mod()
    face, skin, background = _masks()
    reference = _constant_rgb((0.40, 0.40, 0.40))
    candidate = reference.clone()
    darker_bg = torch.full_like(candidate, 0.22)
    candidate = torch.where(background.bool().expand_as(candidate), darker_bg, candidate)

    out = m.PortraitRenderingLoss(quantile_mode="hard")(
        candidate, reference, face, skin, background
    )

    assert out["loss_fb"].item() > 0.5
    assert out["loss_exp"].item() < 1e-6
    assert out["loss_color"].item() < 1e-6
    assert out["loss_tone"].item() < 1e-6


def test_total_loss_is_weighted_sum():
    m = _mod()
    face, skin, background = _masks()
    reference = _constant_rgb((0.35, 0.40, 0.45))
    candidate = _constant_rgb((0.45, 0.38, 0.32))

    loss_fn = m.PortraitRenderingLoss(
        w_exp=0.25,
        w_color=0.25,
        w_tone=0.30,
        w_fb=0.20,
        quantile_mode="hard",
    )
    out = loss_fn(candidate, reference, face, skin, background)
    expected = (
        0.25 * out["loss_exp"]
        + 0.25 * out["loss_color"]
        + 0.30 * out["loss_tone"]
        + 0.20 * out["loss_fb"]
    )
    assert torch.allclose(out["loss"], expected, atol=1e-7)


def test_soft_mode_supports_finite_gradients():
    m = _mod()
    face, skin, background = _masks()
    reference = torch.rand(2, 3, 12, 12) * 0.7 + 0.1
    candidate = (reference + 0.03 * torch.randn_like(reference)).clamp(0.01, 0.99).requires_grad_()
    face = torch.zeros(2, 1, 12, 12)
    face[:, :, 2:10, 2:10] = 1
    skin = face.clone()
    background = 1 - face

    out = m.PortraitRenderingLoss(quantile_mode="soft", soft_quantile_temperature=0.08)(
        candidate, reference, face, skin, background
    )
    out["loss"].backward()

    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum().item() > 0


def test_masks_must_have_pixels_for_every_sample():
    m = _mod()
    image = torch.full((1, 3, 8, 8), 0.4)
    empty = torch.zeros(1, 1, 8, 8)
    face, skin, background = _masks()

    with pytest.raises(ValueError, match="face_mask"):
        m.PortraitRenderingLoss()(image, image, empty, skin, background)
