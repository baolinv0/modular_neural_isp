"""Train photofinishing from same-scene, non-pixel-aligned references."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from photofinishing_model import PhotofinishingModule
from reference_style.contracts import ReferenceStyleLossWeights, TrainingStage
from reference_style.dataset import UnalignedReferenceDataset, unaligned_collate
from reference_style.losses import UnalignedReferenceStyleLoss
from reference_style.trainer import TwoStageReferenceStyleTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL with sample_id/source/reference")
    parser.add_argument("--model-path", required=True, help="pretrained photofinishing state_dict")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--luma-epochs", type=int, default=5)
    parser.add_argument("--chroma-epochs", type=int, default=5)
    parser.add_argument("--luma-lr", type=float, default=1e-5)
    parser.add_argument("--chroma-lr", type=float, default=1e-5)
    parser.add_argument("--use-3d-lut", action="store_true")
    parser.add_argument("--train-3d-lut", action="store_true")
    return parser


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state_dict")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    dataset = UnalignedReferenceDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=unaligned_collate)
    model = PhotofinishingModule(device=device, use_3d_lut=args.use_3d_lut)
    model.load_state_dict(_load_state_dict(args.model_path), strict=True)
    trainer = TwoStageReferenceStyleTrainer(
        model=model,
        loss_fn=UnalignedReferenceStyleLoss(ReferenceStyleLossWeights()),
        device=device,
        train_3d_lut=args.train_3d_lut,
    )
    output_dir = Path(args.output_dir)
    luma = trainer.train_stage(TrainingStage.LUMA, loader, args.luma_epochs, args.luma_lr, output_dir)
    chroma = trainer.train_stage(TrainingStage.CHROMA, loader, args.chroma_epochs, args.chroma_lr, output_dir)
    report = {"luma": luma.__dict__, "chroma": chroma.__dict__, "reference_supervision": "non_pixel_aligned"}
    (output_dir / "reference_style_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
