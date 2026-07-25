"""Two-stage training loop for same-scene non-aligned style adaptation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .contracts import TwoStageTrainingConfig
from .losses import Stage1UnpairedLoss, Stage2UnpairedLoss
from .stages import clone_frozen_model, configure_trainable_stage, trainable_parameters


def set_deterministic_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _load_state_dict(path: str) -> Mapping[str, torch.Tensor]:
  payload = torch.load(path, map_location="cpu", weights_only=True)
  if isinstance(payload, Mapping):
    for key in ("model_state_dict", "state_dict", "model"):
      if key in payload and isinstance(payload[key], Mapping):
        return payload[key]
  if not isinstance(payload, Mapping):
    raise ValueError("checkpoint does not contain a state dict")
  return payload


def load_checkpoint_strict(model: nn.Module, path: str) -> None:
  state_dict = _load_state_dict(path)
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if missing or unexpected:
    raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def _forward_training(model: nn.Module, image: torch.Tensor) -> Dict[str, torch.Tensor]:
  output = model(image, training_mode=True)
  required = {"output", "cbcr_lut"}
  if not isinstance(output, dict) or not required.issubset(output):
    raise RuntimeError("PhotofinishingModule training output is missing output/cbcr_lut")
  if not torch.isfinite(output["output"]).all():
    raise RuntimeError("NON_FINITE_MODEL_OUTPUT")
  if not torch.isfinite(output["cbcr_lut"]).all():
    raise RuntimeError("NON_FINITE_CHROMA_LUT")
  return output


def _move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
  moved: Dict[str, object] = {}
  for key, value in batch.items():
    moved[key] = value.to(device) if torch.is_tensor(value) else value
  return moved


def _finite_loss(loss: torch.Tensor, details: Mapping[str, torch.Tensor]) -> None:
  if not torch.isfinite(loss):
    raise RuntimeError("NON_FINITE_TRAINING_LOSS")
  for name, value in details.items():
    if not torch.isfinite(value):
      raise RuntimeError(f"NON_FINITE_LOSS_COMPONENT:{name}")


def _run_epoch(
    *,
    model: nn.Module,
    anchor_model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_module: nn.Module,
    stage: str,
    device: torch.device,
    gradient_clip_norm: float,
) -> Dict[str, float]:
  training = optimizer is not None
  model.train(training)
  anchor_model.eval()
  totals: Dict[str, float] = {"total": 0.0}
  sample_count = 0
  for raw_batch in loader:
    batch = _move_batch(raw_batch, device)
    images = batch["input"]
    references = batch["reference"]
    with torch.no_grad():
      anchor = _forward_training(anchor_model, images)["output"]
    output_dict = _forward_training(model, images)
    if stage == "stage1":
      loss, details = loss_module(
        output_dict["output"], references, anchor, batch["confidence"],
        batch["input_masks"], batch["reference_masks"],
        batch["region_valid"], batch["region_weights"],
      )
    elif stage == "stage2":
      identity_lut = model._lut_net.identity_grid
      loss, details = loss_module(
        output_dict["output"], references, anchor, output_dict["cbcr_lut"],
        identity_lut, batch["confidence"], batch["input_masks"],
        batch["reference_masks"], batch["region_valid"], batch["region_weights"],
      )
    else:
      raise ValueError(f"unknown stage: {stage}")
    _finite_loss(loss, details)
    if training:
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), gradient_clip_norm)
      optimizer.step()
    batch_size = images.shape[0]
    sample_count += batch_size
    totals["total"] += float(loss.detach()) * batch_size
    for name, value in details.items():
      totals[name] = totals.get(name, 0.0) + float(value) * batch_size
  if sample_count == 0:
    raise RuntimeError("empty data loader")
  return {name: value / sample_count for name, value in totals.items()}


def _save_checkpoint(model: nn.Module, path: Path, *, stage: str, epoch: int, metrics: Mapping[str, float]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "model_state_dict": model.state_dict(),
    "stage": stage,
    "epoch": epoch,
    "metrics": dict(metrics),
  }, path)


def train_two_stage(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TwoStageTrainingConfig,
    output_dir: str,
    device: torch.device,
    run_stage1: bool = True,
    run_stage2: bool = True,
) -> Dict[str, object]:
  """Run Stage 1 (Gain/GTM) and Stage 2 (Chroma LUT) sequentially."""
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  model.to(device)
  source_anchor = clone_frozen_model(model).to(device)
  report: Dict[str, object] = {"config": config.to_dict(), "stages": {}}

  if run_stage1:
    configure_trainable_stage(model, "stage1")
    stage1_loss = Stage1UnpairedLoss(config.stage1, config.histogram_bins, config.histogram_sigma)
    optimizer = Adam(list(trainable_parameters(model)), lr=config.stage1_learning_rate,
                     weight_decay=config.weight_decay)
    best = math.inf
    history = []
    for epoch in range(1, config.stage1_epochs + 1):
      train_metrics = _run_epoch(
        model=model, anchor_model=source_anchor, loader=train_loader, optimizer=optimizer,
        loss_module=stage1_loss, stage="stage1", device=device,
        gradient_clip_norm=config.gradient_clip_norm)
      with torch.no_grad():
        validation_metrics = _run_epoch(
          model=model, anchor_model=source_anchor, loader=validation_loader, optimizer=None,
          loss_module=stage1_loss, stage="stage1", device=device,
          gradient_clip_norm=config.gradient_clip_norm)
      history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
      _save_checkpoint(model, output_path / "stage1_last.pth", stage="stage1", epoch=epoch,
                       metrics=validation_metrics)
      if validation_metrics["total"] < best:
        best = validation_metrics["total"]
        _save_checkpoint(model, output_path / "stage1_best.pth", stage="stage1", epoch=epoch,
                         metrics=validation_metrics)
    load_checkpoint_strict(model, str(output_path / "stage1_best.pth"))
    report["stages"]["stage1"] = {"best_validation_loss": best, "history": history}
  elif run_stage2:
    raise ValueError("stage2 requires an already loaded Stage-1 checkpoint when stage1 is skipped")

  if run_stage2:
    stage1_anchor = clone_frozen_model(model).to(device)
    configure_trainable_stage(model, "stage2")
    stage2_loss = Stage2UnpairedLoss(config.stage2, config.chroma_bins, config.chroma_sigma)
    optimizer = Adam(list(trainable_parameters(model)), lr=config.stage2_learning_rate,
                     weight_decay=config.weight_decay)
    best = math.inf
    history = []
    for epoch in range(1, config.stage2_epochs + 1):
      train_metrics = _run_epoch(
        model=model, anchor_model=stage1_anchor, loader=train_loader, optimizer=optimizer,
        loss_module=stage2_loss, stage="stage2", device=device,
        gradient_clip_norm=config.gradient_clip_norm)
      with torch.no_grad():
        validation_metrics = _run_epoch(
          model=model, anchor_model=stage1_anchor, loader=validation_loader, optimizer=None,
          loss_module=stage2_loss, stage="stage2", device=device,
          gradient_clip_norm=config.gradient_clip_norm)
      history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
      _save_checkpoint(model, output_path / "stage2_last.pth", stage="stage2", epoch=epoch,
                       metrics=validation_metrics)
      if validation_metrics["total"] < best:
        best = validation_metrics["total"]
        _save_checkpoint(model, output_path / "stage2_best.pth", stage="stage2", epoch=epoch,
                         metrics=validation_metrics)
    load_checkpoint_strict(model, str(output_path / "stage2_best.pth"))
    report["stages"]["stage2"] = {"best_validation_loss": best, "history": history}

  report_path = output_path / "training_report.json"
  report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  return report
