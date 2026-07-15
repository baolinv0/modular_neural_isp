from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from .schemas import TeacherRecord
from .teacher_manifest import load_teacher_manifest


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DistillationLossConfig:
    lambda_luminance: float = 1.0
    lambda_gradient: float = 0.20
    lambda_chroma: float = 0.10

    def __post_init__(self) -> None:
        if min(self.lambda_luminance, self.lambda_gradient, self.lambda_chroma) < 0:
            raise ValueError("Distillation loss weights must be non-negative")
        if self.lambda_luminance + self.lambda_gradient + self.lambda_chroma <= 0:
            raise ValueError("At least one distillation loss weight must be positive")


@dataclass(frozen=True)
class BaselineDistillationConfig:
    teacher_manifest: Path
    baseline_checkpoint: Path
    output_dir: Path
    train_modules: tuple[str, ...] = ("gain", "gtm")
    image_size: int = 512
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    parameter_anchor_weight: float = 1e-5
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False
    loss: DistillationLossConfig = DistillationLossConfig()

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("image_size, batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.parameter_anchor_weight < 0:
            raise ValueError("Invalid optimizer or anchor configuration")
        if self.amp and not str(self.device).startswith("cuda"):
            raise ValueError("AMP requires a CUDA device")


def _srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))


def _luminance(linear_rgb: torch.Tensor) -> torch.Tensor:
    weights = linear_rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (linear_rgb * weights).sum(dim=1, keepdim=True)


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    sample_weights = sample_weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if values.shape != sample_weights.shape:
        raise ValueError(f"Per-sample values {values.shape} do not match weights {sample_weights.shape}")
    denominator = sample_weights.sum().clamp_min(1e-8)
    return (values * sample_weights).sum() / denominator


def weighted_distillation_loss(
    prediction_srgb: torch.Tensor,
    target_srgb: torch.Tensor,
    sample_weights: torch.Tensor,
    config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction_srgb.shape != target_srgb.shape or prediction_srgb.ndim != 4 or prediction_srgb.shape[1] != 3:
        raise ValueError("prediction and target must have matching NCHW RGB shapes")
    prediction = _srgb_to_linear(prediction_srgb)
    target = _srgb_to_linear(target_srgb)
    pred_luma = _luminance(prediction)
    target_luma = _luminance(target)

    luminance_per_sample = (pred_luma - target_luma).abs().mean(dim=(1, 2, 3))
    pred_dx = pred_luma[..., :, 1:] - pred_luma[..., :, :-1]
    pred_dy = pred_luma[..., 1:, :] - pred_luma[..., :-1, :]
    target_dx = target_luma[..., :, 1:] - target_luma[..., :, :-1]
    target_dy = target_luma[..., 1:, :] - target_luma[..., :-1, :]
    gradient_per_sample = 0.5 * (
        (pred_dx - target_dx).abs().mean(dim=(1, 2, 3))
        + (pred_dy - target_dy).abs().mean(dim=(1, 2, 3))
    )
    pred_sum = prediction.sum(dim=1, keepdim=True).clamp_min(1e-6)
    target_sum = target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pred_chroma = prediction[:, (0, 2)] / pred_sum
    target_chroma = target[:, (0, 2)] / target_sum
    chroma_per_sample = (pred_chroma - target_chroma).abs().mean(dim=(1, 2, 3))

    luminance = _weighted_mean(luminance_per_sample, sample_weights)
    gradient = _weighted_mean(gradient_per_sample, sample_weights)
    chroma = _weighted_mean(chroma_per_sample, sample_weights)
    total = (
        config.lambda_luminance * luminance
        + config.lambda_gradient * gradient
        + config.lambda_chroma * chroma
    )
    return total, {
        "luminance": float(luminance.detach()),
        "gradient": float(gradient.detach()),
        "chroma": float(chroma.detach()),
        "total": float(total.detach()),
    }


def select_trainable_baseline_modules(model: nn.Module, modules: Sequence[str]) -> dict[str, int]:
    mapping = {
        "gain": "_gain_net",
        "gtm": "_gtm_net",
        "ltm": "_ltm_net",
    }
    normalized = tuple(dict.fromkeys(str(item).lower() for item in modules))
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown baseline module(s): {unknown}")
    if not normalized:
        raise ValueError("At least one baseline module must be trainable")
    for parameter in model.parameters():
        parameter.requires_grad = False
    report: dict[str, int] = {}
    for name, attribute in mapping.items():
        module = getattr(model, attribute, None)
        if not isinstance(module, nn.Module):
            raise AttributeError(f"Baseline model is missing module {attribute}")
        enabled = name in normalized
        for parameter in module.parameters():
            parameter.requires_grad = enabled
        report[name] = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return report


def _read_image(path: str | Path, *, require_uint16: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected readable three-channel image: {path}")
    if require_uint16 and image.dtype != np.uint16:
        raise TypeError(f"Linear input must be uint16: {path}")
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"Unsupported image dtype {image.dtype}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / scale


def _fit_square(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / max(height, 1), size / max(width, 1))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    pad_top = (size - new_height) // 2
    pad_bottom = size - new_height - pad_top
    pad_left = (size - new_width) // 2
    pad_right = size - new_width - pad_left
    return np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")


class TeacherDistillationDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Iterable[TeacherRecord], *, image_size: int) -> None:
        self.records = list(records)
        self.image_size = int(image_size)
        if not self.records:
            raise ValueError("Teacher distillation dataset is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source = _fit_square(_read_image(record.input_path, require_uint16=True), self.image_size)
        target = _fit_square(_read_image(record.target_path, require_uint16=False), self.image_size)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(source)).permute(2, 0, 1),
            "target": torch.from_numpy(np.ascontiguousarray(target)).permute(2, 0, 1),
            "weight": torch.tensor(float(record.sample_weight), dtype=torch.float32),
            "scene_id": record.scene_id,
            "status": record.status,
        }


def _parameter_anchor(model: nn.Module, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            terms.append((parameter - reference[name].to(parameter.device)).square().mean())
    if not terms:
        raise RuntimeError("No trainable baseline parameters")
    return torch.stack(terms).mean()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: DistillationLossConfig,
    optimizer: torch.optim.Optimizer | None,
    parameter_reference: dict[str, torch.Tensor],
    parameter_anchor_weight: float,
    amp: bool,
    scaler: Any | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.eval()
    if training:
        for module in model.modules():
            if module is model:
                continue
            if any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
                module.train()
    totals = {"loss": 0.0, "luminance": 0.0, "gradient": 0.0, "chroma": 0.0, "parameter_anchor": 0.0}
    sample_count = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        weights = batch["weight"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=amp):
            output = model(inputs)["output"]
            distill_loss, parts = weighted_distillation_loss(output, targets, weights, loss_config)
            anchor = _parameter_anchor(model, parameter_reference)
            loss = distill_loss + float(parameter_anchor_weight) * anchor
        if training:
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
        batch_size = int(inputs.shape[0])
        sample_count += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["luminance"] += parts["luminance"] * batch_size
        totals["gradient"] += parts["gradient"] * batch_size
        totals["chroma"] += parts["chroma"] * batch_size
        totals["parameter_anchor"] += float(anchor.detach()) * batch_size
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def distill_fixed_baseline(config: BaselineDistillationConfig) -> dict[str, Any]:
    """Fine-tune one fixed automatic baseline checkpoint from IQA-selected teachers."""

    from adjustTM.model import LuminanceOnlyBaseline, load_baseline_checkpoint

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    records = load_teacher_manifest(config.teacher_manifest)
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if not train_records:
        raise ValueError("Teacher manifest contains no training records")

    model = LuminanceOnlyBaseline(device=device).to(device)
    load_baseline_checkpoint(model, config.baseline_checkpoint, map_location=device)
    module_report = select_trainable_baseline_modules(model, config.train_modules)
    parameter_reference = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    train_loader = DataLoader(
        TeacherDistillationDataset(train_records, image_size=config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = (
        DataLoader(
            TeacherDistillationDataset(validation_records, image_size=config.image_size),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        if validation_records
        else None
    )
    optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best_fixed_baseline.pth"
    for epoch in range(config.epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            loss_config=config.loss,
            optimizer=optimizer,
            parameter_reference=parameter_reference,
            parameter_anchor_weight=config.parameter_anchor_weight,
            amp=config.amp,
            scaler=scaler,
        )
        with torch.no_grad():
            validation_metrics = (
                _run_epoch(
                    model,
                    validation_loader,
                    device=device,
                    loss_config=config.loss,
                    optimizer=None,
                    parameter_reference=parameter_reference,
                    parameter_anchor_weight=config.parameter_anchor_weight,
                    amp=config.amp,
                )
                if validation_loader is not None
                else train_metrics
            )
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "version": 1,
                    "type": "iqa_teacher_fixed_baseline",
                    "state_dict": model.state_dict(),
                    "trainable_modules": list(config.train_modules),
                    "teacher_manifest": str(config.teacher_manifest),
                    "epoch": best_epoch,
                    "validation_loss": best_loss,
                    "module_report": module_report,
                },
                checkpoint_path,
            )
    reloaded = LuminanceOnlyBaseline(device=device).to(device)
    load_baseline_checkpoint(reloaded, checkpoint_path, map_location=device)
    frozen_drifts = [
        float((parameter.detach().cpu() - parameter_reference[name]).abs().max())
        for name, parameter in reloaded.named_parameters()
        if name not in trainable_names
    ]
    frozen_max_drift = max(frozen_drifts, default=0.0)
    if frozen_max_drift != 0.0:
        raise RuntimeError(f"Frozen baseline parameter drift detected: {frozen_max_drift}")

    report = {
        "version": 1,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train_scene_count": len(train_records),
        "validation_scene_count": len(validation_records),
        "train_improved_count": sum(record.status == "improved" for record in train_records),
        "train_anchor_count": sum(record.status != "improved" for record in train_records),
        "module_report": module_report,
        "frozen_parameter_max_drift": frozen_max_drift,
        "baseline_checkpoint_sha256": _sha256_file(config.baseline_checkpoint),
        "teacher_manifest_sha256": _sha256_file(config.teacher_manifest),
        "config": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items() if key != "loss"},
            "loss": asdict(config.loss),
        },
        "history": history,
    }
    (output_dir / "distillation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
