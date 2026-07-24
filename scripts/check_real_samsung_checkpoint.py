#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from photofinishing.photofinishing_model import PhotofinishingModule


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("photofinishing/models/photofinishing_s24-style-0.pth"),
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        print(json.dumps({"status": "UNAVAILABLE", "reason": "checkpoint_missing"}, sort_keys=True))
        return 2
    model = PhotofinishingModule(device=torch.device("cpu"), use_3d_lut=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image = torch.linspace(0.02, 0.72, 3 * 64 * 64).view(1, 3, 64, 64).requires_grad_(True)
    result = model(image, training_mode=True)
    output = result["output"]
    output.mean().backward()
    report = {
        "status": "PASS",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "strict_load": True,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "input_gradient_finite": bool(image.grad is not None and torch.isfinite(image.grad).all().item()),
        "input_gradient_nonzero": bool(image.grad is not None and image.grad.abs().sum().item() > 0.0),
        "trainable_model_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "real_checkpoint_interface_verified": True,
        "real_data_effectiveness_verified": False,
        "model_qualified_for_cross_camera_use": False,
    }
    passed = (
        report["output_finite"]
        and report["input_gradient_finite"]
        and report["input_gradient_nonzero"]
        and report["trainable_model_parameters"] == 0
    )
    report["status"] = "PASS" if passed else "FAIL"
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
