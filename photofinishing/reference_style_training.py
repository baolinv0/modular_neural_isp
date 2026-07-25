"""Staged fine-tuning helpers for unaligned reference-style experiments."""
from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
  from photofinishing.reference_style_losses import ChromaReferenceLoss, ToneReferenceLoss
except ImportError:
  from reference_style_losses import ChromaReferenceLoss, ToneReferenceLoss


_STAGE_MODULES = {
  "tone": ("_gain_net", "_gtm_net"),
  "chroma": ("_lut_net",),
}


@dataclass(frozen=True)
class StageResult:
  stage: str
  best_validation_loss: float
  best_checkpoint: str
  last_checkpoint: str
  trainable_parameter_names: Tuple[str, ...]


def set_deterministic_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _get_module(model: torch.nn.Module, name: str):
  if not hasattr(model, name):
    raise AttributeError(f"Model does not expose required photofinishing stage '{name}'")
  return getattr(model, name)


def configure_trainable_stage(model: torch.nn.Module, stage: str) -> Tuple[str, ...]:
  """Freezes all stages, then enables exactly the modules allowed by the experiment."""
  if stage not in _STAGE_MODULES:
    raise ValueError(f"Unsupported stage '{stage}'")
  for parameter in model.parameters():
    parameter.requires_grad = False
  for module_name in _STAGE_MODULES[stage]:
    module = _get_module(model, module_name)
    if module is None:
      raise ValueError(f"Stage '{stage}' requires module '{module_name}'")
    for parameter in module.parameters():
      parameter.requires_grad = True
  names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
  if not names:
    raise RuntimeError(f"Stage '{stage}' has no trainable parameters")
  expected_prefixes = _STAGE_MODULES[stage]
  unexpected = [name for name in names if not name.startswith(expected_prefixes)]
  if unexpected:
    raise RuntimeError(f"Unexpected trainable parameters for stage '{stage}': {unexpected}")
  return names


def set_stage_train_mode(model: torch.nn.Module, stage: str) -> None:
  """Keeps frozen stages in eval mode while training only the selected stage."""
  model.eval()
  for module_name in _STAGE_MODULES[stage]:
    _get_module(model, module_name).train()


def assert_only_expected_gradients(model: torch.nn.Module, stage: str) -> None:
  expected_prefixes = _STAGE_MODULES[stage]
  violations = []
  for name, parameter in model.named_parameters():
    if parameter.grad is None:
      continue
    if not bool(torch.isfinite(parameter.grad).all()):
      raise FloatingPointError(f"Non-finite gradient in parameter '{name}'")
    if not name.startswith(expected_prefixes):
      violations.append(name)
  if violations:
    raise RuntimeError(f"Frozen parameters received gradients: {violations}")


def extract_model_output(model: torch.nn.Module, input_image: torch.Tensor) -> torch.Tensor:
  result = model(input_image, training_mode=True)
  if isinstance(result, Mapping):
    if "output" not in result:
      raise KeyError("Photofinishing model output dictionary is missing 'output'")
    output = result["output"]
  elif torch.is_tensor(result):
    output = result
  else:
    raise TypeError(f"Unsupported model output type: {type(result)}")
  if not torch.isfinite(output).all():
    raise FloatingPointError("Non-finite model output")
  return output


def _save_checkpoint(path: str, model: torch.nn.Module, stage: str, epoch: int,
                     validation_loss: float, trainable_names: Sequence[str],
                     source_checkpoint: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save({
    "schema_version": 1,
    "stage": stage,
    "epoch": int(epoch),
    "validation_loss": float(validation_loss),
    "trainable_parameter_names": list(trainable_names),
    "source_checkpoint": os.path.abspath(source_checkpoint),
    "model_state_dict": model.state_dict(),
  }, path)


def _run_epoch(model: torch.nn.Module, loader: DataLoader, stage: str,
               device: torch.device, loss_module: torch.nn.Module,
               optimizer: Optional[torch.optim.Optimizer] = None,
               anchor_model: Optional[torch.nn.Module] = None) -> float:
  training = optimizer is not None
  if training:
    set_stage_train_mode(model, stage)
  else:
    model.eval()
  total = 0.0
  count = 0
  context = torch.enable_grad() if training else torch.no_grad()
  with context:
    for batch in loader:
      input_image = batch["input"].to(device=device, non_blocking=True)
      reference = batch["reference"].to(device=device, non_blocking=True)
      weight = batch["weight"].to(device=device, non_blocking=True)
      if training:
        optimizer.zero_grad(set_to_none=True)
      output = extract_model_output(model, input_image)
      if stage == "tone":
        loss, _ = loss_module(output, reference, weight)
      else:
        if anchor_model is None:
          raise ValueError("Chroma stage requires a frozen tone-stage anchor model")
        with torch.no_grad():
          anchor_output = extract_model_output(anchor_model, input_image)
        loss, _ = loss_module(output, reference, anchor_output, weight)
      if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite {stage} loss")
      if training:
        loss.backward()
        assert_only_expected_gradients(model, stage)
        optimizer.step()
      batch_size = int(input_image.shape[0])
      total += float(loss.detach()) * batch_size
      count += batch_size
  if count == 0:
    raise ValueError("Empty dataloader")
  return total / count


def train_stage(model: torch.nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, stage: str, epochs: int,
                learning_rate: float, weight_decay: float, device: torch.device,
                output_dir: str, source_checkpoint: str,
                anchor_model: Optional[torch.nn.Module] = None) -> StageResult:
  if epochs <= 0:
    raise ValueError("epochs must be positive")
  if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise ValueError("learning_rate must be finite and positive")
  trainable_names = configure_trainable_stage(model, stage)
  trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
  optimizer = torch.optim.Adam(
    trainable_parameters,
    lr=learning_rate,
    weight_decay=weight_decay,
  )
  loss_module = ToneReferenceLoss() if stage == "tone" else ChromaReferenceLoss()
  loss_module.to(device)
  best_loss = float("inf")
  best_path = os.path.join(output_dir, f"{stage}_best.pth")
  last_path = os.path.join(output_dir, f"{stage}_last.pth")
  for epoch in range(1, epochs + 1):
    _run_epoch(
      model,
      train_loader,
      stage,
      device,
      loss_module,
      optimizer,
      anchor_model,
    )
    validation_loss = _run_epoch(
      model,
      val_loader,
      stage,
      device,
      loss_module,
      None,
      anchor_model,
    )
    if validation_loss < best_loss:
      best_loss = validation_loss
      _save_checkpoint(
        best_path,
        model,
        stage,
        epoch,
        validation_loss,
        trainable_names,
        source_checkpoint,
      )
    _save_checkpoint(
      last_path,
      model,
      stage,
      epoch,
      validation_loss,
      trainable_names,
      source_checkpoint,
    )
  if not math.isfinite(best_loss):
    raise FloatingPointError("No finite validation loss was observed")
  return StageResult(stage, best_loss, best_path, last_path, trainable_names)


def load_model_state(path: str, map_location: torch.device) -> Dict[str, torch.Tensor]:
  payload = torch.load(path, map_location=map_location)
  if isinstance(payload, Mapping) and "model_state_dict" in payload:
    state = dict(payload["model_state_dict"])
  elif isinstance(payload, Mapping) and "state_dict" in payload:
    state = dict(payload["state_dict"])
  elif isinstance(payload, Mapping):
    state = dict(payload)
  else:
    raise TypeError("Unsupported checkpoint payload")
  if state and all(name.startswith("module.") for name in state):
    state = {name[len("module."):]: value for name, value in state.items()}
  return state


def build_anchor_model(model: torch.nn.Module) -> torch.nn.Module:
  anchor = copy.deepcopy(model).eval()
  for parameter in anchor.parameters():
    parameter.requires_grad = False
  return anchor
