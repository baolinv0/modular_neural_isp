import torch

from photofinishing.reference_style_losses import (
  chroma_style_loss,
  luminance_style_loss,
)


def _shuffle_spatial(image):
  batch, channels, height, width = image.shape
  index = torch.randperm(height * width)
  return image.flatten(2)[:, :, index].reshape(batch, channels, height, width)


def test_luminance_loss_is_spatially_permutation_invariant():
  torch.manual_seed(4)
  image = torch.rand(1, 3, 32, 32)
  shuffled = _shuffle_spatial(image)
  loss, _ = luminance_style_loss(image, shuffled)
  assert float(loss) < 1e-5


def test_luminance_loss_detects_brightness_shift():
  image = torch.full((1, 3, 32, 32), 0.2)
  reference = torch.full((1, 3, 32, 32), 0.6)
  loss, _ = luminance_style_loss(image, reference)
  assert float(loss) > 0.5


def test_chroma_loss_is_non_spatial_but_detects_color_shift():
  torch.manual_seed(8)
  image = torch.rand(1, 3, 32, 32)
  shuffled = _shuffle_spatial(image)
  same_loss, _ = chroma_style_loss(image, shuffled)
  shifted = image.clone()
  shifted[:, 0] = (shifted[:, 0] + 0.25).clamp(0, 1)
  shift_loss, _ = chroma_style_loss(image, shifted)
  assert float(same_loss) < 1e-5
  assert float(shift_loss) > float(same_loss) + 0.01
