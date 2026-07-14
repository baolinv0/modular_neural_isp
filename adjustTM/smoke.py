from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from .constants import CONTROL_METHODS
from .dataset import read_linear_png16
from .model import ControlledBrightnessISP, load_baseline_checkpoint, load_control_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test all adjustTM methods against a real baseline checkpoint")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--input", default=None, help="Optional 16-bit linear RGB PNG")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="adjustTM/smoke_results.json")
    return parser.parse_args()


def _input_tensor(path: str | None, image_size: int, device: torch.device) -> torch.Tensor:
    if path is None:
        generator = torch.Generator().manual_seed(123)
        return torch.rand((1, 3, image_size, image_size), generator=generator, device=device)
    image = read_linear_png16(path).unsqueeze(0).to(device)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=True)
    return image


def run_method(method: str, checkpoint: str, x: torch.Tensor, device: torch.device) -> dict:
    model = ControlledBrightnessISP(method, device=device).to(device)
    load_baseline_checkpoint(model.baseline, checkpoint, map_location=device)
    model.freeze_baseline()
    model.assert_baseline_frozen()
    model.train()

    baseline_before = {key: value.detach().clone() for key, value in model.baseline.state_dict().items()}
    control_before = {key: value.detach().clone() for key, value in model.control_state_dict().items()}
    zero = model(x, torch.zeros(x.shape[0], device=device), training_mode=True)["output"]
    with torch.no_grad():
        baseline_zero = model.forward_baseline(x, training_mode=True)["output"]
    zero_drift = float((zero.detach() - baseline_zero).abs().max())
    if zero_drift != 0.0:
        raise RuntimeError(f"{method}: alpha-zero drift is {zero_drift}")

    optimizer = torch.optim.Adam(model.control_parameters(), lr=1e-3)
    negative = model(x, -torch.ones(x.shape[0], device=device), training_mode=True)["output"]
    positive = model(x, torch.ones(x.shape[0], device=device), training_mode=True)["output"]
    loss = negative.mean() - positive.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(parameter.grad is not None for parameter in model.baseline.parameters()):
        raise RuntimeError(f"{method}: frozen baseline received gradients")
    optimizer.step()

    control_changed = any(
        not torch.equal(control_before[key], value) for key, value in model.control_state_dict().items()
    )
    baseline_unchanged = all(
        torch.equal(baseline_before[key], value) for key, value in model.baseline.state_dict().items()
    )
    if not control_changed or not baseline_unchanged:
        raise RuntimeError(
            f"{method}: control_changed={control_changed}, baseline_unchanged={baseline_unchanged}"
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{method}.pth"
        torch.save({"control_method": method, "control_state_dict": model.control_state_dict()}, path)
        restored = ControlledBrightnessISP(method, device=device).to(device)
        load_baseline_checkpoint(restored.baseline, checkpoint, map_location=device)
        restored.freeze_baseline()
        load_control_checkpoint(restored, path, map_location=device)
        round_trip = all(
            torch.equal(model.control_state_dict()[key], restored.control_state_dict()[key])
            for key in model.control_state_dict()
        )
    if not round_trip:
        raise RuntimeError(f"{method}: control checkpoint round trip failed")

    return {
        "alpha_zero_max_drift": zero_drift,
        "loss": float(loss.detach()),
        "control_changed": control_changed,
        "baseline_unchanged": baseline_unchanged,
        "checkpoint_round_trip": round_trip,
        "parameter_report": model.parameter_report(),
        "negative_output_mean": float(negative.detach().mean()),
        "positive_output_mean": float(positive.detach().mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    x = _input_tensor(args.input, args.image_size, device)
    results = {
        method: run_method(method, args.baseline_checkpoint, x, device) for method in CONTROL_METHODS
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
