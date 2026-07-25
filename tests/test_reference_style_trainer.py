import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "photofinishing"))

import torch
import torch.nn as nn

from reference_style.contracts import TrainingStage
from reference_style.losses import UnalignedReferenceStyleLoss
from reference_style.trainer import TwoStageReferenceStyleTrainer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._gain_net = nn.Conv2d(3, 3, 1)
        self._gtm_net = nn.Conv2d(3, 3, 1)
        self._ltm_net = nn.Identity()
        self._gamma_net = nn.Identity()
        self._lut_net = nn.Conv2d(3, 3, 1)
        self._3d_lut = None

    def forward(self, x, training_mode=True):
        x = torch.sigmoid(self._gain_net(x))
        x = torch.sigmoid(self._gtm_net(x))
        x = torch.sigmoid(self._lut_net(x))
        return {"output": x}


def _loader():
    return [{
        "sample_id": "s1",
        "source": torch.rand(1, 3, 12, 16),
        "reference": torch.rand(1, 3, 9, 11),
    }]


def test_two_stage_training_writes_checkpoints(tmp_path):
    trainer = TwoStageReferenceStyleTrainer(ToyModel(), UnalignedReferenceStyleLoss(bins=8), torch.device("cpu"))
    luma = trainer.train_stage(TrainingStage.LUMA, _loader(), 1, 1e-3, tmp_path)
    chroma = trainer.train_stage(TrainingStage.CHROMA, _loader(), 1, 1e-3, tmp_path)
    assert Path(luma.checkpoint).is_file()
    assert Path(chroma.checkpoint).is_file()
    assert luma.mean_loss >= 0
    assert chroma.mean_loss >= 0
