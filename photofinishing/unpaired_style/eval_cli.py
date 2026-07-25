"""CLI for scene-disjoint non-pixel holdout evaluation."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .contracts import load_manifest
from .data import UnpairedStyleDataset
from .evaluation import evaluate_models
from .trainer import load_checkpoint_strict, set_deterministic_seed


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Evaluate unpaired Photofinishing style adaptation")
  parser.add_argument("--manifest", required=True, help="JSONL manifest containing split=test")
  parser.add_argument("--source-checkpoint", required=True)
  parser.add_argument("--adapted-checkpoint", required=True)
  parser.add_argument("--output", required=True, help="Output JSON report")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--image-size", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--use-3d-lut", action="store_true")
  return parser


def main(argv=None) -> int:
  args = build_parser().parse_args(argv)
  set_deterministic_seed(args.seed)
  records = load_manifest(args.manifest)
  if any(record.split != "test" for record in records):
    raise ValueError("evaluation manifest may only contain split=test")
  dataset = UnpairedStyleDataset(records, image_size=args.image_size)
  loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers)
  from photofinishing.photofinishing_model import PhotofinishingModule
  device = torch.device(args.device)
  baseline_model = PhotofinishingModule(device=device, use_3d_lut=args.use_3d_lut)
  adapted_model = PhotofinishingModule(device=device, use_3d_lut=args.use_3d_lut)
  load_checkpoint_strict(baseline_model, args.source_checkpoint)
  load_checkpoint_strict(adapted_model, args.adapted_checkpoint)
  report = evaluate_models(
    baseline_model=baseline_model,
    adapted_model=adapted_model,
    loader=loader,
    device=device,
    output_path=args.output,
  )
  print(json.dumps(report["aggregate"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
