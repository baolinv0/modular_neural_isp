import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from photofinishing.train_unpaired_style import train_epoch
from photofinishing.unpaired_stage_control import AdaptationStage, ParameterAnchor, configure_trainable_scope
from photofinishing.unpaired_style_losses import Stage1UnpairedLoss, Stage2UnpairedLoss


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.full((3, 8, 8), 0.25 + 0.05 * index)
        base = 0.5 + 0.05 * index
        reference = torch.stack([
            torch.full((8, 8), base + 0.10),
            torch.full((8, 8), base),
            torch.full((8, 8), base - 0.10),
        ])
        return {"sample_id": str(index), "input_image": image, "reference_image": reference}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._gain_net = nn.Conv2d(3, 3, 1, bias=False)
        self._gtm_net = nn.Conv2d(3, 3, 1, bias=False)
        self._ltm_net = nn.Identity()
        self._lut_net = nn.Conv2d(3, 2, 1, bias=False)
        self._gamma_net = nn.Identity()
        nn.init.eye_(self._gain_net.weight[:, :, 0, 0])
        nn.init.eye_(self._gtm_net.weight[:, :, 0, 0])
        nn.init.zeros_(self._lut_net.weight)

    def forward(self, x, training_mode=False):
        gain = self._gain_net(x)
        gtm = self._gtm_net(gain)
        lut_delta = self._lut_net(gtm).mean(dim=(2, 3), keepdim=True)
        output = torch.clamp(gtm + torch.cat([lut_delta, lut_delta[:, :1]], dim=1), 0, 1)
        lut = lut_delta.expand(-1, -1, 4, 4)
        return {"output": output, "cbcr_lut": lut}


def test_stage1_epoch_updates_gain_gtm_only():
    model = TinyModel()
    configure_trainable_scope(model, AdaptationStage.LUMINANCE)
    before_lut = model._lut_net.weight.detach().clone()
    anchor = ParameterAnchor(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    metrics = train_epoch(
        model,
        DataLoader(TinyDataset(), batch_size=1),
        optimizer,
        AdaptationStage.LUMINANCE,
        Stage1UnpairedLoss(),
        torch.device("cpu"),
        parameter_anchor=anchor,
    )
    assert metrics["total"] > 0
    assert torch.equal(before_lut, model._lut_net.weight)


def test_stage2_epoch_updates_lut_only_and_uses_frozen_reference():
    model = TinyModel()
    reference_model = copy.deepcopy(model)
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    configure_trainable_scope(model, AdaptationStage.CHROMA)
    before_gain = model._gain_net.weight.detach().clone()
    before_lut = model._lut_net.weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    metrics = train_epoch(
        model,
        DataLoader(TinyDataset(), batch_size=1),
        optimizer,
        AdaptationStage.CHROMA,
        Stage2UnpairedLoss(),
        torch.device("cpu"),
        frozen_stage1_model=reference_model,
    )
    assert metrics["total"] > 0
    assert torch.equal(before_gain, model._gain_net.weight)
    assert not torch.equal(before_lut, model._lut_net.weight)
