"""Trainable-scope controls for the two-stage unpaired experiment."""
from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class AdaptationStage(str, Enum):
    LUMINANCE = "luminance"
    CHROMA = "chroma"


_STAGE_MODULES = {
    AdaptationStage.LUMINANCE: ("_gain_net", "_gtm_net"),
    AdaptationStage.CHROMA: ("_lut_net",),
}


def configure_trainable_scope(model: nn.Module, stage: AdaptationStage | str) -> List[str]:
    stage = AdaptationStage(stage)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in _STAGE_MODULES[stage]:
        module = getattr(model, module_name, None)
        if module is None:
            raise AttributeError(f"Model missing required module {module_name}")
        for parameter in module.parameters():
            parameter.requires_grad = True
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError(f"No trainable parameters configured for stage={stage.value}")
    return names


def assert_trainable_scope(model: nn.Module, stage: AdaptationStage | str) -> None:
    stage = AdaptationStage(stage)
    allowed_prefixes = _STAGE_MODULES[stage]
    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise AssertionError(f"Unexpected trainable parameters for {stage.value}: {unexpected[:8]}")
    for prefix in allowed_prefixes:
        if not any(name.startswith(prefix) and parameter.requires_grad for name, parameter in model.named_parameters()):
            raise AssertionError(f"No trainable parameters under {prefix}")


def set_stage_train_mode(model: nn.Module, stage: AdaptationStage | str) -> None:
    """Keeps frozen modules in eval mode and trains only the active stage."""
    stage = AdaptationStage(stage)
    model.eval()
    for module_name in _STAGE_MODULES[stage]:
        getattr(model, module_name).train()


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
