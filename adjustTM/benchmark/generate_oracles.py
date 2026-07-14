from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adjustTM.constants import LEVELS
from .baselines import search_best_parameter
from .image_io import read_srgb_png, resize_like, write_srgb_png16
from .schemas import read_json
from .transforms import exposure_transform, luminance_gamma_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostic per-image exposure/gamma oracle outputs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parameter-output", required=True)
    parser.add_argument("--methods", nargs="+", choices=["exposure", "gamma"], default=["exposure", "gamma"])
    parser.add_argument("--steps", type=int, default=161)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    baseline_root = Path(args.baseline_output_root)
    output_root = Path(args.output_root)
    records = []
    for scene in manifest["scenes"]:
        scene_id = scene["scene_id"]
        baseline_path = baseline_root / "a_000" / scene_id
        if not baseline_path.is_file():
            baseline_path = baseline_root / scene_id
        baseline = read_srgb_png(baseline_path, device=args.device)
        for level_name, alpha in LEVELS:
            target = resize_like(read_srgb_png(scene["gt"][level_name]["path"], device=args.device), baseline)
            for method in args.methods:
                if level_name == "a_000":
                    parameter = 0.0 if method == "exposure" else 1.0
                    objective = 0.0
                else:
                    result = search_best_parameter(
                        baseline, target, kind=method,
                        minimum=-4.0 if method == "exposure" else 0.25,
                        maximum=4.0, steps=args.steps,
                    )
                    parameter, objective = result.parameter, result.objective
                output = exposure_transform(baseline, parameter) if method == "exposure" else luminance_gamma_transform(baseline, parameter)
                method_name = f"{method}_oracle"
                write_srgb_png16(output_root / method_name / level_name / scene_id, output)
                records.append({
                    "scene_id": scene_id, "level": level_name, "alpha": alpha,
                    "method": method_name, "parameter": parameter, "objective": objective,
                })
    path = Path(args.parameter_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
