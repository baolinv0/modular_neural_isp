"""Two-stage non-pixel-aligned photofinishing trainer."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from .contracts import ReferenceStyleBatch, TrainingStage
from .losses import LossResult, UnalignedReferenceStyleLoss
from .stage_control import configure_training_stage


@dataclass
class StageSummary:
    stage: str
    epochs: int
    mean_loss: float
    checkpoint: str


class TwoStageReferenceStyleTrainer:
    def __init__(self, model: nn.Module, loss_fn: UnalignedReferenceStyleLoss,
                 device: torch.device, train_3d_lut: bool = False):
        self.model = model.to(device)
        self.anchor_model = copy.deepcopy(model).to(device).eval()
        self.anchor_model.requires_grad_(False)
        self.loss_fn = loss_fn
        self.device = device
        self.train_3d_lut = train_3d_lut

    @staticmethod
    def _extract_output(model_output: object) -> torch.Tensor:
        if isinstance(model_output, dict):
            output = model_output.get("output")
            if output is None:
                raise KeyError("model output dictionary lacks 'output'")
            return output
        if not torch.is_tensor(model_output):
            raise TypeError("model must return a tensor or {'output': tensor}")
        return model_output

    def _forward(self, model: nn.Module, source: torch.Tensor) -> torch.Tensor:
        try:
            result = model(source, training_mode=True)
        except TypeError:
            result = model(source)
        return self._extract_output(result)

    def train_stage(self, stage: TrainingStage, loader: Iterable[dict[str, object]], epochs: int,
                    lr: float, output_dir: str | Path) -> StageSummary:
        if epochs <= 0 or lr <= 0:
            raise ValueError("epochs and lr must be positive")
        params = configure_training_stage(self.model, stage, train_3d_lut=self.train_3d_lut)
        optimizer = torch.optim.Adam(params, lr=lr)
        losses: list[float] = []
        for _ in range(epochs):
            self.model.train()
            for raw_batch in loader:
                batch = ReferenceStyleBatch(
                    source=raw_batch["source"].to(self.device),
                    reference=raw_batch["reference"].to(self.device),
                    metadata={"sample_id": raw_batch.get("sample_id")},
                )
                batch.validate()
                with torch.no_grad():
                    anchor = self._forward(self.anchor_model, batch.source)
                output = self._forward(self.model, batch.source)
                result: LossResult = self.loss_fn(output, batch.reference, anchor, stage)
                optimizer.zero_grad(set_to_none=True)
                result.total.backward()
                optimizer.step()
                losses.append(float(result.total.detach().cpu()))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / f"reference_style_{stage.value}.pth"
        torch.save({"stage": stage.value, "model_state_dict": self.model.state_dict()}, checkpoint)
        if stage == TrainingStage.LUMA:
            self.anchor_model.load_state_dict(self.model.state_dict())
            self.anchor_model.eval().requires_grad_(False)
        return StageSummary(stage=stage.value, epochs=epochs,
                            mean_loss=sum(losses) / max(len(losses), 1), checkpoint=str(checkpoint))
