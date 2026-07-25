import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "photofinishing"))

import torch

from reference_style.contracts import ReferenceStyleLossWeights, TrainingStage
from reference_style.losses import UnalignedReferenceStyleLoss


def _gradient(batch=1, h=32, w=40, scale=1.0):
    y = torch.linspace(0.05, 0.85, h * w).view(1, 1, h, w).repeat(batch, 3, 1, 1)
    return (y * scale).clamp(0, 1)


def test_luma_reference_can_have_different_resolution():
    loss_fn = UnalignedReferenceStyleLoss()
    result = loss_fn(_gradient(), _gradient(h=19, w=27, scale=1.2), _gradient(), TrainingStage.LUMA)
    assert torch.isfinite(result.total)
    assert result.total > 0


def test_chroma_loss_preserves_stage1_luma():
    loss_fn = UnalignedReferenceStyleLoss(ReferenceStyleLossWeights(luma_preservation=5.0))
    anchor = _gradient()
    output = anchor.clone()
    output[:, 0] = (output[:, 0] + 0.1).clamp(0, 1)
    result = loss_fn(output, torch.flip(anchor, dims=[1]), anchor, TrainingStage.CHROMA)
    assert result.terms["luma_preservation"] > 0


def test_non_finite_input_fails_closed():
    loss_fn = UnalignedReferenceStyleLoss()
    out = _gradient()
    out[..., 0, 0] = float("nan")
    try:
        loss_fn(out, _gradient(), _gradient(), TrainingStage.LUMA)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("expected fail-closed")
