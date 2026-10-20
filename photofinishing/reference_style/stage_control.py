"""Freeze policy for the two-stage reference-style experiment."""
from __future__ import annotations

import torch.nn as nn

from .contracts import TrainingStage

_STAGE_MODULES = {
    TrainingStage.LUMA: ("_gain_net", "_gtm_net"),
    TrainingStage.CHROMA: ("_lut_net", "_3d_lut"),
}
_ALL_MODULES = ("_gain_net", "_gtm_net", "_ltm_net", "_gamma_net", "_lut_net", "_3d_lut")


def configure_training_stage(model: nn.Module, stage: TrainingStage, train_3d_lut: bool = False) -> list[nn.Parameter]:
    if stage not in _STAGE_MODULES:
        raise ValueError(f"unsupported stage: {stage}")
    for name in _ALL_MODULES:
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    names = list(_STAGE_MODULES[stage])
    if stage == TrainingStage.CHROMA and not train_3d_lut:
        names.remove("_3d_lut")
    parameters: list[nn.Parameter] = []
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue
        module.requires_grad_(True)
        module.train()
        parameters.extend(p for p in module.parameters() if p.requires_grad)
    if not parameters:
        raise RuntimeError(f"stage {stage.value} has no trainable parameters")
    return parameters


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]
