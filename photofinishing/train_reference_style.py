"""Two-stage non-pixel-aligned reference-style fine-tuning experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
  sys.path.append(REPO_ROOT)

try:
  from photofinishing.reference_style_data import ReferenceStyleDataset
  from photofinishing.reference_style_training import (
    build_anchor_model,
    load_model_state,
    set_deterministic_seed,
    train_stage,
  )
except ImportError:
  from reference_style_data import ReferenceStyleDataset
  from reference_style_training import (
    build_anchor_model,
    load_model_state,
    set_deterministic_seed,
    train_stage,
  )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Fine-tune Gain/GTM, then the CbCr LUT, from same-scene "
      "non-pixel-aligned references."
    )
  )
  parser.add_argument("--manifest", required=True, help="JSONL manifest with train/val rows")
  parser.add_argument("--base-checkpoint", required=True, help="Pretrained photofinishing checkpoint")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--image-size", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--tone-epochs", type=int, default=10)
  parser.add_argument("--chroma-epochs", type=int, default=10)
  parser.add_argument("--tone-lr", type=float, default=1e-5)
  parser.add_argument("--chroma-lr", type=float, default=1e-5)
  parser.add_argument("--weight-decay", type=float, default=1e-6)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--use-3d-lut",
    action="store_true",
    help="Instantiate an optional 3D LUT; it remains frozen in both stages",
  )
  parser.add_argument("--skip-chroma", action="store_true", help="Run only Gain/GTM fine-tuning")
  return parser


def _build_loader(manifest: str, split: str, image_size: int,
                  batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
  dataset = ReferenceStyleDataset(manifest, split=split, image_size=image_size)
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
  )


def main(argv=None) -> int:
  args = build_parser().parse_args(argv)
  if args.image_size <= 0 or args.batch_size <= 0 or args.num_workers < 0:
    raise ValueError("Invalid dataloader configuration")
  if args.tone_epochs <= 0:
    raise ValueError("tone-epochs must be positive")
  if not args.skip_chroma and args.chroma_epochs <= 0:
    raise ValueError("chroma-epochs must be positive unless --skip-chroma is used")
  set_deterministic_seed(args.seed)
  device = torch.device(args.device)
  os.makedirs(args.output_dir, exist_ok=True)

  train_loader = _build_loader(
    args.manifest,
    "train",
    args.image_size,
    args.batch_size,
    args.num_workers,
    shuffle=True,
  )
  val_loader = _build_loader(
    args.manifest,
    "val",
    args.image_size,
    args.batch_size,
    args.num_workers,
    shuffle=False,
  )

  try:
    from photofinishing.photofinishing_model import PhotofinishingModule
  except ImportError:
    from photofinishing_model import PhotofinishingModule
  model = PhotofinishingModule(
    device=device,
    use_3d_lut=args.use_3d_lut,
  ).to(device)
  model.load_state_dict(load_model_state(args.base_checkpoint, device), strict=True)

  tone_result = train_stage(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    stage="tone",
    epochs=args.tone_epochs,
    learning_rate=args.tone_lr,
    weight_decay=args.weight_decay,
    device=device,
    output_dir=args.output_dir,
    source_checkpoint=args.base_checkpoint,
  )
  model.load_state_dict(load_model_state(tone_result.best_checkpoint, device), strict=True)

  summary = {"tone": asdict(tone_result), "chroma": None}
  if not args.skip_chroma:
    tone_anchor = build_anchor_model(model).to(device)
    chroma_result = train_stage(
      model=model,
      train_loader=train_loader,
      val_loader=val_loader,
      stage="chroma",
      epochs=args.chroma_epochs,
      learning_rate=args.chroma_lr,
      weight_decay=args.weight_decay,
      device=device,
      output_dir=args.output_dir,
      source_checkpoint=tone_result.best_checkpoint,
      anchor_model=tone_anchor,
    )
    summary["chroma"] = asdict(chroma_result)

  summary.update({
    "schema_version": 1,
    "manifest": os.path.abspath(args.manifest),
    "base_checkpoint": os.path.abspath(args.base_checkpoint),
    "reference_supervision": "non_pixel_aligned_distributional",
    "tone_trainable_modules": ["_gain_net", "_gtm_net"],
    "chroma_trainable_modules": ["_lut_net"] if not args.skip_chroma else [],
    "frozen_modules_during_chroma": [
      "_gain_net",
      "_gtm_net",
      "_ltm_net",
      "_gamma_net",
      "_3d_lut",
    ],
  })
  with open(
    os.path.join(args.output_dir, "training_summary.json"),
    "w",
    encoding="utf-8",
  ) as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
