"""Strict module freezing for the two-stage adaptation protocol."""

from __future__ import annotations

import copy
from typing import Iterable, List

import torch
import torch.nn as nn


_STAGE_MODULES = {
  "stage1": ("_gain_net", "_gtm_net"),
  "stage2": ("_lut_net",),
}


def _require_modules(model: nn.Module) -> None:
  required = {"_gain_net", "_gtm_net", "_ltm_net", "_lut_net", "_gamma_net"}
  missing = [name for name in sorted(required) if not hasattr(model, name)]
  if missing:
    raise AttributeError(f"model is missing photofinishing modules: {missing}")


def configure_trainable_stage(model: nn.Module, stage: str) -> List[str]:
  """Freeze the whole model, then enable exactly the requested modules."""
  _require_modules(model)
  if stage not in _STAGE_MODULES:
    raise ValueError(f"unsupported stage: {stage}")
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  for module_name in _STAGE_MODULES[stage]:
    module = getattr(model, module_name)
    for parameter in module.parameters():
      parameter.requires_grad_(True)
  names = trainable_parameter_names(model)
  if not names:
    raise RuntimeError(f"stage {stage} exposed no trainable parameters")
  allowed_prefixes = tuple(f"{name}." for name in _STAGE_MODULES[stage])
  unexpected = [name for name in names if not name.startswith(allowed_prefixes)]
  if unexpected:
    raise RuntimeError(f"unexpected trainable parameters in {stage}: {unexpected}")
  return names


def trainable_parameter_names(model: nn.Module) -> List[str]:
  return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
  return (parameter for parameter in model.parameters() if parameter.requires_grad)


def clone_frozen_model(model: nn.Module) -> nn.Module:
  clone = copy.deepcopy(model)
  clone.eval()
  for parameter in clone.parameters():
    parameter.requires_grad_(False)
  return clone


def assert_frozen_modules_unchanged(
    before: dict,
    model: nn.Module,
    allowed_prefixes: tuple[str, ...],
) -> None:
  for name, parameter in model.named_parameters():
    if name.startswith(allowed_prefixes):
      continue
    if not torch.equal(before[name], parameter.detach().cpu()):
      raise RuntimeError(f"frozen parameter changed: {name}")
