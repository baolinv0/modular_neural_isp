import torch
import torch.nn as nn

from photofinishing.unpaired_stage_control import (
    AdaptationStage,
    ParameterAnchor,
    assert_trainable_scope,
    configure_trainable_scope,
    set_stage_train_mode,
)


class FakePhotofinishing(nn.Module):
    def __init__(self):
        super().__init__()
        self._gain_net = nn.Conv2d(3, 3, 1)
        self._gtm_net = nn.Conv2d(3, 3, 1)
        self._ltm_net = nn.Conv2d(3, 3, 1)
        self._lut_net = nn.Conv2d(3, 3, 1)
        self._gamma_net = nn.Conv2d(3, 3, 1)


def _trainable_roots(model):
    return {name.split(".")[0] for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_luminance_stage_trains_only_gain_and_gtm():
    model = FakePhotofinishing()
    configure_trainable_scope(model, AdaptationStage.LUMINANCE)
    assert_trainable_scope(model, AdaptationStage.LUMINANCE)
    assert _trainable_roots(model) == {"_gain_net", "_gtm_net"}


def test_chroma_stage_trains_only_lut():
    model = FakePhotofinishing()
    configure_trainable_scope(model, AdaptationStage.CHROMA)
    assert_trainable_scope(model, AdaptationStage.CHROMA)
    assert _trainable_roots(model) == {"_lut_net"}


def test_parameter_anchor_detects_change():
    model = FakePhotofinishing()
    configure_trainable_scope(model, AdaptationStage.LUMINANCE)
    anchor = ParameterAnchor(model)
    assert anchor.loss(model).item() == 0
    with torch.no_grad():
        model._gain_net.weight.add_(0.1)
    assert anchor.loss(model).item() > 0


def test_stage_train_mode_keeps_frozen_modules_in_eval():
    model = FakePhotofinishing()
    configure_trainable_scope(model, AdaptationStage.LUMINANCE)
    set_stage_train_mode(model, AdaptationStage.LUMINANCE)
    assert model._gain_net.training
    assert model._gtm_net.training
    assert not model._ltm_net.training
    assert not model._lut_net.training
    assert not model._gamma_net.training
