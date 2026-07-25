"""CLI for two-stage non-pixel-aligned photofinishing adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .contracts import TwoStageTrainingConfig, load_manifest, validate_disjoint_manifests
from .data import UnpairedStyleDataset
from .trainer import load_checkpoint_strict, set_deterministic_seed, train_two_stage


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Unpaired same-scene photofinishing style adaptation")
  parser.add_argument("--train-manifest", required=True)
  parser.add_argument("--validation-manifest", required=True)
  parser.add_argument("--checkpoint", required=True, help="Source PhotofinishingModule checkpoint")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--config", default=None, help="Optional JSON training config")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--image-size", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--use-3d-lut", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  return parser


def _load_config(path: str | None) -> TwoStageTrainingConfig:
  if path is None:
    return TwoStageTrainingConfig()
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  return TwoStageTrainingConfig.from_dict(payload)


def main(argv=None) -> int:
  args = build_parser().parse_args(argv)
  set_deterministic_seed(args.seed)
  train_records = load_manifest(args.train_manifest)
  validation_records = load_manifest(args.validation_manifest)
  if any(record.split != "train" for record in train_records):
    raise ValueError("train manifest may only contain split=train")
  if any(record.split != "validation" for record in validation_records):
    raise ValueError("validation manifest may only contain split=validation")
  validate_disjoint_manifests(train_records, validation_records)
  train_dataset = UnpairedStyleDataset(train_records, image_size=args.image_size)
  validation_dataset = UnpairedStyleDataset(validation_records, image_size=args.image_size)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
  validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
  from photofinishing.photofinishing_model import PhotofinishingModule
  device = torch.device(args.device)
  model = PhotofinishingModule(device=device, use_3d_lut=args.use_3d_lut)
  load_checkpoint_strict(model, args.checkpoint)
  config = _load_config(args.config)
  if args.dry_run:
    first = next(iter(train_loader))
    with torch.no_grad():
      output = model(first["input"].to(device), training_mode=True)
    print(json.dumps({
      "status": "DRY_RUN_PASS",
      "train_samples": len(train_dataset),
      "validation_samples": len(validation_dataset),
      "train_regions": train_dataset.region_names,
      "validation_regions": validation_dataset.region_names,
      "output_shape": list(output["output"].shape),
    }, indent=2))
    return 0
  report = train_two_stage(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    config=config,
    output_dir=args.output_dir,
    device=device,
  )
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
