from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from photofinishing.unpaired_style.contracts import TwoStageTrainingConfig
from photofinishing.unpaired_style.trainer import train_two_stage


class ScalarNet(nn.Module):
  def __init__(self, channels=1):
    super().__init__()
    self.value = nn.Parameter(torch.zeros(channels))


class TinyLutNet(ScalarNet):
  def __init__(self):
    super().__init__(2)
    axis = torch.linspace(-0.5, 0.5, 8)
    cb, cr = torch.meshgrid(axis, axis, indexing="ij")
    self.register_buffer("identity_grid", torch.stack([cb, cr], dim=0).unsqueeze(0))


class TinyPhotofinishing(nn.Module):
  def __init__(self):
    super().__init__()
    self._gain_net = ScalarNet()
    self._gtm_net = ScalarNet()
    self._ltm_net = ScalarNet()
    self._lut_net = TinyLutNet()
    self._gamma_net = ScalarNet()
    self._3d_lut = None

  def forward(self, x, training_mode=False):
    scale = torch.exp(self._gain_net.value + self._gtm_net.value).view(1, 1, 1, 1)
    base = (x * scale).clamp(0, 1)
    color = self._lut_net.value.view(1, 2, 1, 1)
    out = base.clone()
    out[:, 0:1] = (out[:, 0:1] + color[:, 0:1] * 0.1).clamp(0, 1)
    out[:, 2:3] = (out[:, 2:3] + color[:, 1:2] * 0.1).clamp(0, 1)
    lut = self._lut_net.identity_grid + color
    return {"output": out, "cbcr_lut": lut.expand(x.shape[0], -1, -1, -1)}


class TinyDataset(Dataset):
  def __len__(self):
    return 2

  def __getitem__(self, index):
    input_image = torch.full((3, 12, 12), 0.25)
    reference = torch.stack([
      torch.full((12, 12), 0.55),
      torch.full((12, 12), 0.45),
      torch.full((12, 12), 0.35),
    ])
    return {
      "input": input_image,
      "reference": reference,
      "confidence": torch.tensor(1.0),
      "input_masks": torch.zeros((0, 1, 12, 12)),
      "reference_masks": torch.zeros((0, 1, 12, 12)),
      "region_valid": torch.zeros((0,)),
      "region_weights": torch.zeros((0,)),
      "sample_id": str(index),
      "scene_group": str(index),
      "region_names": [],
    }


def test_two_stage_training_updates_only_intended_modules(tmp_path: Path):
  model = TinyPhotofinishing()
  ltm_before = model._ltm_net.value.detach().clone()
  gamma_before = model._gamma_net.value.detach().clone()
  gain_before = model._gain_net.value.detach().clone()
  lut_before = model._lut_net.value.detach().clone()
  loader = DataLoader(TinyDataset(), batch_size=2)
  config = TwoStageTrainingConfig(
    stage1_epochs=1,
    stage2_epochs=1,
    stage1_learning_rate=5e-2,
    stage2_learning_rate=5e-2,
    histogram_bins=8,
    chroma_bins=8,
    histogram_sigma=0.08,
    chroma_sigma=0.08,
  )
  report = train_two_stage(
    model=model,
    train_loader=loader,
    validation_loader=loader,
    config=config,
    output_dir=str(tmp_path),
    device=torch.device("cpu"),
  )
  assert not torch.equal(gain_before, model._gain_net.value.detach())
  assert not torch.equal(lut_before, model._lut_net.value.detach())
  assert torch.equal(ltm_before, model._ltm_net.value.detach())
  assert torch.equal(gamma_before, model._gamma_net.value.detach())
  assert (tmp_path / "stage1_best.pth").is_file()
  assert (tmp_path / "stage2_best.pth").is_file()
  assert set(report["stages"]) == {"stage1", "stage2"}
