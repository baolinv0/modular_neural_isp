import torch
import torch.nn as nn

from photofinishing.reference_style_losses import (
  ChromaReferenceLoss,
  ToneReferenceLoss,
)
from photofinishing.reference_style_training import (
  assert_only_expected_gradients,
  build_anchor_model,
  configure_trainable_stage,
  extract_model_output,
)


class ScalarStage(nn.Module):
  def __init__(self, initial=0.0):
    super().__init__()
    self.value = nn.Parameter(torch.tensor(float(initial)))


class FakePhotofinishing(nn.Module):
  def __init__(self):
    super().__init__()
    self._gain_net = ScalarStage(0.0)
    self._gtm_net = ScalarStage(0.0)
    self._ltm_net = ScalarStage(0.0)
    self._lut_net = ScalarStage(0.0)
    self._gamma_net = ScalarStage(0.0)
    self._3d_lut = None

  def forward(self, x, training_mode=True):
    brightness = torch.sigmoid(self._gain_net.value + self._gtm_net.value)
    out = (x * (0.5 + brightness)).clamp(0, 1)
    chroma = self._lut_net.value
    out = torch.cat([
      (out[:, 0:1] + chroma).clamp(0, 1),
      out[:, 1:2],
      (out[:, 2:3] - chroma).clamp(0, 1),
    ], dim=1)
    return {"output": out}


def test_tone_stage_unfreezes_only_gain_and_global():
  model = FakePhotofinishing()
  names = configure_trainable_stage(model, "tone")
  assert names == ("_gain_net.value", "_gtm_net.value")
  assert not model._ltm_net.value.requires_grad
  assert not model._lut_net.value.requires_grad
  assert not model._gamma_net.value.requires_grad


def test_chroma_stage_freezes_gain_global_local_and_gamma():
  model = FakePhotofinishing()
  names = configure_trainable_stage(model, "chroma")
  assert names == ("_lut_net.value",)
  assert not model._gain_net.value.requires_grad
  assert not model._gtm_net.value.requires_grad
  assert not model._ltm_net.value.requires_grad
  assert not model._gamma_net.value.requires_grad


def test_gradients_follow_selected_stage():
  model = FakePhotofinishing()
  configure_trainable_stage(model, "tone")
  output = extract_model_output(model, torch.full((2, 3, 16, 16), 0.2))
  loss, _ = ToneReferenceLoss()(output, torch.full_like(output, 0.6))
  loss.backward()
  assert_only_expected_gradients(model, "tone")
  assert model._gain_net.value.grad is not None
  assert model._gtm_net.value.grad is not None
  assert model._lut_net.value.grad is None


def test_chroma_loss_uses_frozen_tone_anchor():
  model = FakePhotofinishing()
  anchor = build_anchor_model(model)
  configure_trainable_stage(model, "chroma")
  inputs = torch.rand(2, 3, 16, 16)
  output = extract_model_output(model, inputs)
  with torch.no_grad():
    anchor_output = extract_model_output(anchor, inputs)
  reference = output.clone()
  reference[:, 0] = (reference[:, 0] + 0.1).clamp(0, 1)
  loss, details = ChromaReferenceLoss()(output, reference, anchor_output)
  loss.backward()
  assert model._lut_net.value.grad is not None
  assert model._gain_net.value.grad is None
  assert float(details["luma_anchor"].detach()) == 0.0
