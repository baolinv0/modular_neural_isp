import torch

from photofinishing.unpaired_style_losses import (
    chroma_histogram_loss,
    chroma_moment_loss,
    log_exposure_loss,
    luma_distribution_loss,
    luminance_preservation_loss,
    lut_delta_regularization,
)


def _image(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((2, 3, 12, 10), generator=generator)


def test_identical_images_have_zero_distribution_losses():
    image = _image()
    assert log_exposure_loss(image, image).item() < 1e-7
    assert luma_distribution_loss(image, image).item() < 1e-7
    assert chroma_histogram_loss(image, image).item() < 1e-7
    assert chroma_moment_loss(image, image).item() < 1e-7


def test_luma_distribution_is_spatial_order_invariant():
    image = _image()
    shuffled = image.flatten(2)[:, :, torch.randperm(image.shape[2] * image.shape[3])].reshape_as(image)
    assert luma_distribution_loss(image, shuffled).item() < 1e-6


def test_exposure_loss_backpropagates():
    prediction = (_image() * 0.5).requires_grad_()
    reference = (_image() * 0.8).detach()
    loss = log_exposure_loss(prediction, reference)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_y_preservation_uses_fixed_stage1_target():
    stage1 = _image()
    stage2 = (stage1 + 0.1).clamp(0, 1).requires_grad_()
    loss = luminance_preservation_loss(stage2, stage1)
    loss.backward()
    assert loss.item() > 0
    assert stage2.grad is not None
    assert stage1.grad is None


def test_lut_regularization_is_zero_for_unchanged_lut():
    lut = torch.rand(2, 2, 8, 8)
    anchor, smooth = lut_delta_regularization(lut, lut.clone())
    assert anchor.item() == 0
    assert smooth.item() == 0
