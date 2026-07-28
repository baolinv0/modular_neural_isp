"""Two-stage same-scene non-pixel-aligned photofinishing adaptation.

Stage 1 trains only GainNet + GlobalToneMappingNet using luminance statistics.
Stage 2 loads the Stage-1 checkpoint, freezes the luminance path, and either
fine-tunes the full LuTNet or trains a six-parameter affine chroma residual.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

try:
    from .unpaired_chroma_heads import ChromaHead, FrozenLUTAffineResidual, configure_chroma_head
    from .unpaired_reference_data import ReferenceStyleDataset
    from .unpaired_stage_control import (
        AdaptationStage, ParameterAnchor, assert_trainable_scope, configure_trainable_scope, set_stage_train_mode,
        trainable_parameters,
    )
    from .unpaired_style_losses import Stage1LossWeights, Stage1UnpairedLoss, Stage2LossWeights, Stage2UnpairedLoss
except ImportError:  # direct execution from photofinishing/
    from unpaired_chroma_heads import ChromaHead, FrozenLUTAffineResidual, configure_chroma_head
    from unpaired_reference_data import ReferenceStyleDataset
    from unpaired_stage_control import (
        AdaptationStage, ParameterAnchor, assert_trainable_scope, configure_trainable_scope, set_stage_train_mode,
        trainable_parameters,
    )
    from unpaired_style_losses import Stage1LossWeights, Stage1UnpairedLoss, Stage2LossWeights, Stage2UnpairedLoss


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _move_batch(batch: Dict[str, object], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_image = batch["input_image"]
    reference_image = batch["reference_image"]
    if not isinstance(input_image, torch.Tensor) or not isinstance(reference_image, torch.Tensor):
        raise TypeError("Dataset batch must contain tensor input_image/reference_image")
    return input_image.to(device), reference_image.to(device)


def _forward_training(model: torch.nn.Module, input_image: torch.Tensor) -> Dict[str, torch.Tensor]:
    output = model(input_image, training_mode=True)
    if not isinstance(output, dict) or "output" not in output or "cbcr_lut" not in output:
        raise RuntimeError("Photofinishing model must return output and cbcr_lut in training_mode")
    return output


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    stage: AdaptationStage,
    loss_fn: torch.nn.Module,
    device: torch.device,
    parameter_anchor: Optional[ParameterAnchor] = None,
    frozen_stage1_model: Optional[torch.nn.Module] = None,
    chroma_head: ChromaHead | str = ChromaHead.FULL_LUT,
) -> Dict[str, float]:
    set_stage_train_mode(model, stage, chroma_head)
    totals: Dict[str, float] = {"total": 0.0}
    count = 0
    for batch in loader:
        input_image, reference = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        result = _forward_training(model, input_image)
        if stage is AdaptationStage.LUMINANCE:
            if parameter_anchor is None:
                raise ValueError("Stage 1 requires ParameterAnchor")
            loss, terms = loss_fn(result["output"], reference, parameter_anchor.loss(model))
        else:
            if frozen_stage1_model is None:
                raise ValueError("Stage 2 requires a frozen Stage-1 model")
            with torch.no_grad():
                frozen = _forward_training(frozen_stage1_model, input_image)
            loss, terms = loss_fn(
                result["output"], reference, frozen["output"], result["cbcr_lut"], frozen["cbcr_lut"]
            )
        loss.backward()
        optimizer.step()
        count += 1
        totals["total"] += float(loss.detach())
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    if count == 0:
        raise RuntimeError("Empty training loader")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    stage: AdaptationStage,
    loss_fn: torch.nn.Module,
    device: torch.device,
    parameter_anchor: Optional[ParameterAnchor] = None,
    frozen_stage1_model: Optional[torch.nn.Module] = None,
    chroma_head: ChromaHead | str = ChromaHead.FULL_LUT,
) -> Dict[str, float]:
    model.eval()
    if stage is AdaptationStage.CHROMA and ChromaHead(chroma_head) is ChromaHead.AFFINE_RESIDUAL:
        if not isinstance(getattr(model, "_lut_net", None), FrozenLUTAffineResidual):
            raise TypeError("affine_residual requires FrozenLUTAffineResidual")
        model._lut_net.base_lut_net.eval()
    totals: Dict[str, float] = {"total": 0.0}
    count = 0
    for batch in loader:
        input_image, reference = _move_batch(batch, device)
        result = _forward_training(model, input_image)
        if stage is AdaptationStage.LUMINANCE:
            if parameter_anchor is None:
                raise ValueError("Stage 1 requires ParameterAnchor")
            loss, terms = loss_fn(result["output"], reference, parameter_anchor.loss(model))
        else:
            if frozen_stage1_model is None:
                raise ValueError("Stage 2 requires frozen Stage-1 model")
            frozen = _forward_training(frozen_stage1_model, input_image)
            loss, terms = loss_fn(
                result["output"], reference, frozen["output"], result["cbcr_lut"], frozen["cbcr_lut"]
            )
        count += 1
        totals["total"] += float(loss)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    if count == 0:
        raise RuntimeError("Empty validation loader")
    return {name: value / count for name, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-scene non-pixel-aligned photofinishing adaptation")
    parser.add_argument("--stage", choices=[stage.value for stage in AdaptationStage], required=True)
    parser.add_argument(
        "--chroma-head",
        choices=[head.value for head in ChromaHead],
        default=ChromaHead.FULL_LUT.value,
        help="Stage-2 capacity: full adaptive LuTNet or frozen LuTNet plus six-parameter affine residual",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--load", required=True, help="Source checkpoint for luminance; Stage-1 checkpoint for chroma")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--input-mode", choices=["linear_srgb", "raw_metadata"], default="linear_srgb")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--use-3d-lut", action="store_true")
    parser.add_argument(
        "--stage1-run-config", default=None,
        help="Stage-1 run_config.json; required for chroma unless it is beside --load",
    )

    parser.add_argument("--exposure-weight", type=float, default=1.0)
    parser.add_argument("--percentile-weight", type=float, default=1.0)
    parser.add_argument("--luma-distribution-weight", type=float, default=1.0)
    parser.add_argument("--parameter-anchor-weight", type=float, default=1e-4)

    parser.add_argument("--chroma-histogram-weight", type=float, default=1.0)
    parser.add_argument("--chroma-moment-weight", type=float, default=0.5)
    parser.add_argument("--saturation-weight", type=float, default=0.25)
    parser.add_argument("--y-preserve-weight", type=float, default=1.0)
    parser.add_argument("--lut-anchor-weight", type=float, default=0.1)
    parser.add_argument("--lut-smoothness-weight", type=float, default=0.05)
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _load_model(checkpoint: str, device: torch.device, use_3d_lut: bool) -> torch.nn.Module:
    try:
        from .photofinishing_model import PhotofinishingModule
    except ImportError:
        from photofinishing_model import PhotofinishingModule

    model = PhotofinishingModule(device=device, use_3d_lut=use_3d_lut)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stage1_provenance(checkpoint: str, configured_path: Optional[str]) -> Path:
    config_path = Path(configured_path).resolve() if configured_path else Path(checkpoint).resolve().parent / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Chroma stage requires the Stage-1 run_config.json for provenance; "
            f"not found: {config_path}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("stage") != AdaptationStage.LUMINANCE.value:
        raise ValueError(f"Chroma source is not a luminance-stage run: {config_path}")
    expected_hashes = {
        payload.get("best_checkpoint_sha256"),
        payload.get("last_checkpoint_sha256"),
    } - {None, ""}
    if not expected_hashes:
        raise ValueError(f"Stage-1 run config does not bind output checkpoint hashes: {config_path}")
    actual_hash = _sha256_file(checkpoint)
    if actual_hash not in expected_hashes:
        raise ValueError(
            "Loaded Stage-1 checkpoint does not match the hashes recorded in "
            f"{config_path}: {actual_hash}"
        )
    return config_path


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("epochs, batch-size, and image-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    weight_names = [name for name in vars(args) if name.endswith("_weight")]
    negative = [name for name in weight_names if getattr(args, name) < 0]
    if negative:
        raise ValueError(f"Loss weights must be non-negative: {negative}")
    if args.stage == AdaptationStage.LUMINANCE.value:
        if args.chroma_head != ChromaHead.FULL_LUT.value:
            raise ValueError("--chroma-head is only valid for chroma stage")
        if args.exposure_weight + args.percentile_weight + args.luma_distribution_weight <= 0:
            raise ValueError("Luminance stage requires at least one style loss")
    else:
        if args.chroma_histogram_weight + args.chroma_moment_weight + args.saturation_weight <= 0:
            raise ValueError("Chroma stage requires at least one chroma style loss")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_arguments(args)
    set_deterministic(args.seed)
    device = _resolve_device(args.device)
    stage = AdaptationStage(args.stage)
    chroma_head = ChromaHead(args.chroma_head)

    stage1_config_path = None
    if stage is AdaptationStage.CHROMA:
        stage1_config_path = _validate_stage1_provenance(args.load, args.stage1_run_config)
    model = _load_model(args.load, device=device, use_3d_lut=args.use_3d_lut)
    frozen_stage1_model = None
    if stage is AdaptationStage.CHROMA:
        frozen_stage1_model = copy.deepcopy(model).eval()
        for parameter in frozen_stage1_model.parameters():
            parameter.requires_grad = False
        configure_chroma_head(model, chroma_head)

    configure_trainable_scope(model, stage, chroma_head)
    assert_trainable_scope(model, stage, chroma_head)
    parameter_anchor = ParameterAnchor(model) if stage is AdaptationStage.LUMINANCE else None

    train_set = ReferenceStyleDataset(args.manifest, args.train_split, args.image_size, args.input_mode)
    validation_set = ReferenceStyleDataset(args.manifest, args.validation_split, args.image_size, args.input_mode)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda"
    )

    if stage is AdaptationStage.LUMINANCE:
        weights = Stage1LossWeights(
            exposure=args.exposure_weight,
            percentiles=args.percentile_weight,
            distribution=args.luma_distribution_weight,
            parameter_anchor=args.parameter_anchor_weight,
        )
        loss_fn: torch.nn.Module = Stage1UnpairedLoss(weights)
    else:
        weights = Stage2LossWeights(
            chroma_histogram=args.chroma_histogram_weight,
            chroma_moments=args.chroma_moment_weight,
            saturation=args.saturation_weight,
            y_preserve=args.y_preserve_weight,
            lut_anchor=args.lut_anchor_weight,
            lut_smoothness=args.lut_smoothness_weight,
        )
        loss_fn = Stage2UnpairedLoss(weights)

    optimizer = torch.optim.AdamW(
        list(trainable_parameters(model)), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, stage, loss_fn, device, parameter_anchor, frozen_stage1_model, chroma_head
        )
        validation_metrics = validate_epoch(
            model, validation_loader, stage, loss_fn, device, parameter_anchor, frozen_stage1_model, chroma_head
        )
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        logging.info(
            "epoch=%d train=%.6f val=%.6f", epoch, train_metrics["total"], validation_metrics["total"]
        )
        torch.save(model.state_dict(), output_dir / "last.pth")
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            torch.save(model.state_dict(), output_dir / "best.pth")
        _write_json(output_dir / "history.json", {"records": history})

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    head_config: Dict[str, object] = {"chroma_head": chroma_head.value}
    if isinstance(getattr(model, "_lut_net", None), FrozenLUTAffineResidual):
        head_config.update({
            "affine_matrix_limit": model._lut_net.matrix_limit,
            "affine_bias_limit": model._lut_net.bias_limit,
        })
    _write_json(
        output_dir / "run_config.json",
        {
            "stage": stage.value,
            **head_config,
            "source_checkpoint": str(Path(args.load).resolve()),
            "source_checkpoint_sha256": _sha256_file(args.load),
            "stage1_run_config": str(stage1_config_path) if stage1_config_path else None,
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": _sha256_file(args.manifest),
            "train_split": args.train_split,
            "validation_split": args.validation_split,
            "input_mode": args.input_mode,
            "image_size": args.image_size,
            "use_3d_lut": args.use_3d_lut,
            "seed": args.seed,
            "loss_weights": asdict(weights),
            "trainable_parameters": trainable_names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "best_validation_loss": best_loss,
            "best_checkpoint_sha256": _sha256_file(output_dir / "best.pth"),
            "last_checkpoint_sha256": _sha256_file(output_dir / "last.pth"),
        },
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
