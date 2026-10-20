import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "photofinishing"))

import torch.nn as nn

from reference_style.contracts import TrainingStage
from reference_style.stage_control import configure_training_stage, trainable_parameter_names


class ToyPhotofinishing(nn.Module):
    def __init__(self):
        super().__init__()
        self._gain_net = nn.Conv2d(3, 3, 1)
        self._gtm_net = nn.Conv2d(3, 3, 1)
        self._ltm_net = nn.Conv2d(3, 3, 1)
        self._gamma_net = nn.Conv2d(3, 3, 1)
        self._lut_net = nn.Conv2d(3, 3, 1)
        self._3d_lut = nn.Conv2d(3, 3, 1)


def test_luma_stage_only_unfreezes_gain_and_gtm():
    model = ToyPhotofinishing()
    configure_training_stage(model, TrainingStage.LUMA)
    names = trainable_parameter_names(model)
    assert names
    assert all(name.startswith(("_gain_net", "_gtm_net")) for name in names)


def test_chroma_stage_freezes_gain_global_local_gamma():
    model = ToyPhotofinishing()
    configure_training_stage(model, TrainingStage.CHROMA, train_3d_lut=False)
    names = trainable_parameter_names(model)
    assert names
    assert all(name.startswith("_lut_net") for name in names)
