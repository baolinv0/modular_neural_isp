import torch
import torch.nn as nn
import pytest

from photofinishing.train_unpaired_style import _validate_arguments, build_parser
from photofinishing.unpaired_chroma_heads import ChromaHead, FrozenLUTAffineResidual, configure_chroma_head
from photofinishing.unpaired_stage_control import (
    AdaptationStage,
    assert_trainable_scope,
    configure_trainable_scope,
    set_stage_train_mode,
)


class TinyLUT(nn.Module):
    def __init__(self, lut_size: int = 4):
        super().__init__()
        self._lut_size = lut_size
        self.weight = nn.Parameter(torch.ones(2, 2))

    def get_cbcr_lut_size(self) -> int:
        return self._lut_size

    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        return self.weight.mean() * torch.zeros(
            ycbcr.shape[0], 2, self._lut_size, self._lut_size, device=ycbcr.device
        )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._gain_net = nn.Linear(1, 1)
        self._gtm_net = nn.Linear(1, 1)
        self._ltm_net = nn.Linear(1, 1)
        self._lut_net = TinyLUT()
        self._gamma_net = nn.Linear(1, 1)


def _trainable_names(model: nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_stage2_cli_defaults_to_full_lut():
    args = build_parser().parse_args([
        "--stage", "chroma",
        "--manifest", "manifest.csv",
        "--load", "stage1.pth",
        "--output-dir", "run",
    ])
    assert args.chroma_head == ChromaHead.FULL_LUT.value


def test_stage2_cli_accepts_affine_residual():
    args = build_parser().parse_args([
        "--stage", "chroma",
        "--chroma-head", "affine_residual",
        "--manifest", "manifest.csv",
        "--load", "stage1.pth",
        "--output-dir", "run",
    ])
    _validate_arguments(args)
    assert args.chroma_head == ChromaHead.AFFINE_RESIDUAL.value


def test_luminance_stage_rejects_affine_chroma_head():
    args = build_parser().parse_args([
        "--stage", "luminance",
        "--chroma-head", "affine_residual",
        "--manifest", "manifest.csv",
        "--load", "source.pth",
        "--output-dir", "run",
    ])
    with pytest.raises(ValueError, match="only valid for chroma"):
        _validate_arguments(args)


def test_full_lut_mode_preserves_original_lut_and_trains_it():
    model = TinyModel()
    original = model._lut_net
    configure_chroma_head(model, ChromaHead.FULL_LUT)
    configure_trainable_scope(model, AdaptationStage.CHROMA, ChromaHead.FULL_LUT)
    assert_trainable_scope(model, AdaptationStage.CHROMA, ChromaHead.FULL_LUT)
    assert model._lut_net is original
    assert _trainable_names(model) == {"_lut_net.weight"}


def test_affine_mode_wraps_lut_and_trains_exactly_six_scalars():
    model = TinyModel()
    original = model._lut_net
    configure_chroma_head(model, ChromaHead.AFFINE_RESIDUAL)
    configure_trainable_scope(model, AdaptationStage.CHROMA, ChromaHead.AFFINE_RESIDUAL)
    assert_trainable_scope(model, AdaptationStage.CHROMA, ChromaHead.AFFINE_RESIDUAL)
    assert isinstance(model._lut_net, FrozenLUTAffineResidual)
    assert model._lut_net.base_lut_net is original
    assert _trainable_names(model) == {"_lut_net.matrix_raw", "_lut_net.bias_raw"}
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 6


def test_affine_stage_train_mode_keeps_base_lut_in_eval():
    model = TinyModel()
    configure_chroma_head(model, ChromaHead.AFFINE_RESIDUAL)
    configure_trainable_scope(model, AdaptationStage.CHROMA, ChromaHead.AFFINE_RESIDUAL)
    set_stage_train_mode(model, AdaptationStage.CHROMA, ChromaHead.AFFINE_RESIDUAL)
    assert model._lut_net.training
    assert not model._lut_net.base_lut_net.training
