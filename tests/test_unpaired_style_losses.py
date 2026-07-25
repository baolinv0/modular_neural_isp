import torch

from photofinishing.unpaired_style.contracts import Stage1LossWeights, Stage2LossWeights
from photofinishing.unpaired_style.losses import (
    Stage1UnpairedLoss,
    Stage2UnpairedLoss,
    cdf_distance,
    chroma_histogram_loss,
    rgb_to_luminance,
)


def _image(value=0.4):
  return torch.full((2, 3, 16, 16), value, dtype=torch.float32)


def test_luminance_cdf_is_lower_for_matching_distribution():
  dark = rgb_to_luminance(_image(0.2))
  bright = rgb_to_luminance(_image(0.8))
  same = cdf_distance(dark, dark, bins=16, sigma=0.05)
  different = cdf_distance(dark, bright, bins=16, sigma=0.05)
  assert torch.all(same < different)


def test_stage1_loss_backpropagates_to_output_and_anchors_content():
  output = _image(0.25).requires_grad_(True)
  reference = _image(0.65)
  baseline = _image(0.25)
  loss_fn = Stage1UnpairedLoss(Stage1LossWeights(), bins=16, sigma=0.05)
  loss, details = loss_fn(output, reference, baseline, torch.ones(2))
  loss.backward()
  assert torch.isfinite(loss)
  assert output.grad is not None
  assert output.grad.abs().sum() > 0
  assert set(details) >= {"exposure", "luminance_cdf", "edge_anchor"}


def test_chroma_histogram_detects_color_shift():
  neutral = torch.zeros((1, 2, 16, 16))
  shifted = neutral.clone()
  shifted[:, 0] = 0.2
  same = chroma_histogram_loss(neutral, neutral, bins=8, sigma=0.05)
  different = chroma_histogram_loss(neutral, shifted, bins=8, sigma=0.05)
  assert same.item() < different.item()


def test_stage2_luminance_preserve_uses_frozen_anchor():
  output = _image(0.5).requires_grad_(True)
  reference = output.detach().clone()
  anchor = _image(0.3).requires_grad_(True)
  lut = torch.zeros((2, 2, 8, 8), requires_grad=True)
  identity = torch.zeros((1, 2, 8, 8))
  weights = Stage2LossWeights(
    chroma_histogram=0.0, chroma_moments=0.0, saturation_cdf=0.0,
    semantic_regions=0.0, luminance_preserve=1.0, edge_anchor=0.0,
    lut_identity=0.0, lut_total_variation=0.0, lut_bound=0.0)
  loss, _ = Stage2UnpairedLoss(weights, bins=8, sigma=0.05)(
    output, reference, anchor, lut, identity, torch.ones(2))
  loss.backward()
  assert output.grad is not None and output.grad.abs().sum() > 0
  assert anchor.grad is None
