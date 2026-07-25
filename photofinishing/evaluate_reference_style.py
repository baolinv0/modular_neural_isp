"""Evaluate staged checkpoints using non-pixel-aligned style evidence."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
  sys.path.append(REPO_ROOT)

try:
  from photofinishing.reference_style_data import ReferenceStyleDataset
  from photofinishing.reference_style_losses import (
    chroma_style_loss,
    luminance_style_loss,
    rgb_to_luma,
  )
  from photofinishing.reference_style_training import extract_model_output, load_model_state
except ImportError:
  from reference_style_data import ReferenceStyleDataset
  from reference_style_losses import chroma_style_loss, luminance_style_loss, rgb_to_luma
  from reference_style_training import extract_model_output, load_model_state


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Evaluate base, tone, and chroma checkpoints on unaligned references"
  )
  parser.add_argument("--manifest", required=True)
  parser.add_argument("--base-checkpoint", required=True)
  parser.add_argument("--tone-checkpoint", required=True)
  parser.add_argument("--chroma-checkpoint", required=True)
  parser.add_argument("--output-json", required=True)
  parser.add_argument("--split", default="test", choices=("train", "val", "test"))
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--image-size", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--num-workers", type=int, default=2)
  parser.add_argument("--use-3d-lut", action="store_true")
  return parser


def _load_model(checkpoint: str, device: torch.device, use_3d_lut: bool):
  try:
    from photofinishing.photofinishing_model import PhotofinishingModule
  except ImportError:
    from photofinishing_model import PhotofinishingModule
  model = PhotofinishingModule(device=device, use_3d_lut=use_3d_lut).to(device)
  model.load_state_dict(load_model_state(checkpoint, device), strict=True)
  model.eval()
  for parameter in model.parameters():
    parameter.requires_grad = False
  return model


def _scalar(value: torch.Tensor) -> float:
  return float(value.detach().cpu())


def main(argv=None) -> int:
  args = build_parser().parse_args(argv)
  device = torch.device(args.device)
  dataset = ReferenceStyleDataset(
    args.manifest,
    split=args.split,
    image_size=args.image_size,
  )
  loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
  )
  models = {
    "base": _load_model(args.base_checkpoint, device, args.use_3d_lut),
    "tone": _load_model(args.tone_checkpoint, device, args.use_3d_lut),
    "final": _load_model(args.chroma_checkpoint, device, args.use_3d_lut),
  }
  rows = []
  with torch.no_grad():
    for batch in loader:
      inputs = batch["input"].to(device)
      reference = batch["reference"].to(device)
      outputs = {
        name: extract_model_output(model, inputs)
        for name, model in models.items()
      }
      for batch_index, sample_id in enumerate(batch["sample_id"]):
        ref = reference[batch_index:batch_index + 1]
        metrics: Dict[str, object] = {"sample_id": sample_id}
        for name, output in outputs.items():
          current = output[batch_index:batch_index + 1]
          luma, _ = luminance_style_loss(current, ref)
          chroma, _ = chroma_style_loss(current, ref)
          metrics[f"{name}_luma_style_loss"] = _scalar(luma)
          metrics[f"{name}_chroma_style_loss"] = _scalar(chroma)
        metrics["final_luma_change_from_tone"] = _scalar(
          torch.nn.functional.l1_loss(
            rgb_to_luma(outputs["final"][batch_index:batch_index + 1]),
            rgb_to_luma(outputs["tone"][batch_index:batch_index + 1]),
          )
        )
        rows.append(metrics)
  if not rows:
    raise ValueError("No evaluation samples")
  summary = {
    "schema_version": 1,
    "split": args.split,
    "num_samples": len(rows),
    "mean_base_luma_style_loss": (
      sum(row["base_luma_style_loss"] for row in rows) / len(rows)
    ),
    "mean_tone_luma_style_loss": (
      sum(row["tone_luma_style_loss"] for row in rows) / len(rows)
    ),
    "mean_base_chroma_style_loss": (
      sum(row["base_chroma_style_loss"] for row in rows) / len(rows)
    ),
    "mean_final_chroma_style_loss": (
      sum(row["final_chroma_style_loss"] for row in rows) / len(rows)
    ),
    "mean_final_luma_change_from_tone": (
      sum(row["final_luma_change_from_tone"] for row in rows) / len(rows)
    ),
    "samples": rows,
  }
  output_dir = os.path.dirname(os.path.abspath(args.output_json))
  os.makedirs(output_dir, exist_ok=True)
  with open(args.output_json, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
