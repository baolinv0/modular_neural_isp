"""Trainable-scope controls for the two-stage unpaired experiment."""
from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

try:
    from .unpaired_chroma_heads import ChromaHead, FrozenLUTAffineResidual
except ImportError:
    from unpaired_chroma_heads import ChromaHead, FrozenLUTAffineResidual


class AdaptationStage(str, Enum):
    LUMINANCE = "luminance"
    CHROMA = "chroma"


_LUMINANCE_MODULES = ("_gain_net", "_gtm_net")


def configure_trainable_scope(
    model: nn.Module,
    stage: AdaptationStage | str,
    chroma_head: ChromaHead | str = ChromaHead.FULL_LUT,
) -> List[str]:
    stage = AdaptationStage(stage)
    chroma_head = ChromaHead(chroma_head)
    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage is AdaptationStage.LUMINANCE:
        for module_name in _LUMINANCE_MODULES:
            module = getattr(model, module_name, None)
            if module is None:
                raise AttributeError(f"Model missing required module {module_name}")
            for parameter in module.parameters():
                parameter.requires_grad = True
    else:
        lut_net = getattr(model, "_lut_net", None)
        if lut_net is None:
            raise AttributeError("Model missing required module _lut_net")
        if chroma_head is ChromaHead.AFFINE_RESIDUAL:
            if not isinstance(lut_net, FrozenLUTAffineResidual):
                raise TypeError("affine_residual requires FrozenLUTAffineResidual")
            lut_net.matrix_raw.requires_grad = True
            lut_net.bias_raw.requires_grad = True
        else:
            if isinstance(lut_net, FrozenLUTAffineResidual):
                raise TypeError("full_lut cannot use an affine-residual wrapper")
            for parameter in lut_net.parameters():
                parameter.requires_grad = True

    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError(f"No trainable parameters configured for stage={stage.value}")
    return names


def assert_trainable_scope(
    model: nn.Module,
    stage: AdaptationStage | str,
    chroma_head: ChromaHead | str = ChromaHead.FULL_LUT,
) -> None:
    stage = AdaptationStage(stage)
    chroma_head = ChromaHead(chroma_head)
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    if stage is AdaptationStage.LUMINANCE:
        unexpected = [name for name in actual if not name.startswith(_LUMINANCE_MODULES)]
        if unexpected:
            raise AssertionError(f"Unexpected trainable parameters for luminance: {unexpected[:8]}")
        for prefix in _LUMINANCE_MODULES:
            if not any(name.startswith(prefix) for name in actual):
                raise AssertionError(f"No trainable parameters under {prefix}")
        return

    if chroma_head is ChromaHead.AFFINE_RESIDUAL:
        expected = {"_lut_net.matrix_raw", "_lut_net.bias_raw"}
        if actual != expected:
            raise AssertionError(f"Affine chroma trainable parameters must be {sorted(expected)}, got {sorted(actual)}")
        lut_net = getattr(model, "_lut_net", None)
        if not isinstance(lut_net, FrozenLUTAffineResidual):
            raise AssertionError("Affine chroma head wrapper is missing")
        if any(parameter.requires_grad for parameter in lut_net.base_lut_net.parameters()):
            raise AssertionError("Frozen base LuTNet contains trainable parameters")
        return

    if not actual or any(not name.startswith("_lut_net") for name in actual):
        raise AssertionError(f"Unexpected trainable parameters for full_lut: {sorted(actual)[:8]}")
    if isinstance(getattr(model, "_lut_net", None), FrozenLUTAffineResidual):
        raise AssertionError("full_lut mode cannot use affine wrapper")


def set_stage_train_mode(
    model: nn.Module,
    stage: AdaptationStage | str,
    chroma_head: ChromaHead | str = ChromaHead.FULL_LUT,
) -> None:
    """Keeps frozen modules in eval mode and trains only the active stage."""
    stage = AdaptationStage(stage)
    chroma_head = ChromaHead(chroma_head)
    model.eval()
    if stage is AdaptationStage.LUMINANCE:
        for module_name in _LUMINANCE_MODULES:
            getattr(model, module_name).train()
    else:
        lut_net = getattr(model, "_lut_net")
        if chroma_head is ChromaHead.AFFINE_RESIDUAL and not isinstance(lut_net, FrozenLUTAffineResidual):
            raise TypeError("affine_residual requires FrozenLUTAffineResidual")
        lut_net.train()


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


class ParameterAnchor:
    """Anchors the selected trainable parameter set to its initial checkpoint."""

    def __init__(self, model: nn.Module) -> None:
        self._reference: Dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self._reference:
            raise ValueError("Cannot build ParameterAnchor without trainable parameters")

    def loss(self, model: nn.Module) -> torch.Tensor:
        losses = []
        current = dict(model.named_parameters())
        for name, reference in self._reference.items():
            if name not in current:
                raise KeyError(f"Anchored parameter disappeared: {name}")
            losses.append((current[name] - reference.to(current[name].device)).square().mean())
        return torch.stack(losses).mean()
