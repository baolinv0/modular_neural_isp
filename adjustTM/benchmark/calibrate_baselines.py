from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from adjustTM.constants import LEVELS
from .baselines import log_luma_objective, project_level_parameters
from .image_io import read_srgb_png, resize_like
from .transforms import exposure_transform, luminance_gamma_transform
from .schemas import canonical_hash, read_json, write_json_atomic


def calibrate_from_manifest(
    manifest,
    baseline_output_root: str | Path,
    *,
    kind: str,
    minimum: float,
    maximum: float,
    steps: int,
    device: str,
):
    baseline_root = Path(baseline_output_root)
    transform = exposure_transform if kind == "exposure" else luminance_gamma_transform
    parameters = {}
    candidates = np.linspace(minimum, maximum, steps)
    for level_name, _ in LEVELS:
        if level_name == "a_000":
            parameters[level_name] = 0.0 if kind == "exposure" else 1.0
            continue
        scores = np.zeros(len(candidates), dtype=np.float64)
        for scene in manifest["scenes"]:
            scene_id = scene["scene_id"]
            baseline_path = baseline_root / "a_000" / scene_id
            if not baseline_path.is_file():
                baseline_path = baseline_root / scene_id
            baseline = read_srgb_png(baseline_path, device=device)
            target = resize_like(read_srgb_png(scene["gt"][level_name]["path"], device=device), baseline)
            with torch.no_grad():
                for index, candidate in enumerate(candidates):
                    scores[index] += float(log_luma_objective(transform(baseline, float(candidate)), target))
        best = int(np.argmin(scores))
        parameters[level_name] = float(candidates[best])
    return project_level_parameters(parameters, kind=kind)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate deployable global exposure and gamma baselines")
    parser.add_argument("--manifest", required=True, help="Calibration/train manifest; never use the test manifest")
    parser.add_argument("--baseline-output-root", required=True, help="Frozen baseline outputs, a_000/scene.png or scene.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exposure-min", type=float, default=-4.0)
    parser.add_argument("--exposure-max", type=float, default=4.0)
    parser.add_argument("--exposure-steps", type=int, default=161)
    parser.add_argument("--gamma-min", type=float, default=0.25)
    parser.add_argument("--gamma-max", type=float, default=4.0)
    parser.add_argument("--gamma-steps", type=int, default=161)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    exposure = calibrate_from_manifest(
        manifest, args.baseline_output_root, kind="exposure", minimum=args.exposure_min,
        maximum=args.exposure_max, steps=args.exposure_steps, device=args.device
    )
    gamma = calibrate_from_manifest(
        manifest, args.baseline_output_root, kind="gamma", minimum=args.gamma_min,
        maximum=args.gamma_max, steps=args.gamma_steps, device=args.device
    )
    payload = {
        "calibration_manifest_sha256": manifest.get("content_sha256", canonical_hash(manifest)),
        "calibration_scene_count": len(manifest["scenes"]),
        "objective": "mean_pixel_log_luma_mae",
        "exposure_global": exposure,
        "gamma_global": gamma,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
