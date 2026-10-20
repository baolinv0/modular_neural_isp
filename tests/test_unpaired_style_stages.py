import torch
import torch.nn as nn

from photofinishing.unpaired_style.stages import configure_trainable_stage


class TinyPhotofinishing(nn.Module):
  def __init__(self):
    super().__init__()
    self._gain_net = nn.Conv2d(3, 3, 1)
    self._gtm_net = nn.Conv2d(3, 3, 1)
    self._ltm_net = nn.Conv2d(3, 3, 1)
    self._lut_net = nn.Conv2d(3, 3, 1)
    self._gamma_net = nn.Conv2d(3, 3, 1)
    self._3d_lut = nn.Parameter(torch.ones(1))


def test_stage1_exposes_only_gain_and_gtm():
  model = TinyPhotofinishing()
  names = configure_trainable_stage(model, "stage1")
  assert names
  assert all(name.startswith(("_gain_net.", "_gtm_net.")) for name in names)
  assert all(not parameter.requires_grad for parameter in model._ltm_net.parameters())
  assert not model._3d_lut.requires_grad


def test_stage2_exposes_only_chroma_lut():
  model = TinyPhotofinishing()
  names = configure_trainable_stage(model, "stage2")
  assert names
  assert all(name.startswith("_lut_net.") for name in names)
  assert all(not parameter.requires_grad for parameter in model._gain_net.parameters())
