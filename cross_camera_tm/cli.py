from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .canary import run_synthetic_canary
from .certification import CertificationProfile
from .config import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-camera Samsung-style adaptation v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True, type=Path)
    canary = subparsers.add_parser("synthetic-canary")
    canary.add_argument("--config", required=True, type=Path)
    canary.add_argument("--output-dir", required=True, type=Path)
    real = subparsers.add_parser("real-run")
    real.add_argument("--config", required=True, type=Path)
    real.add_argument("--calibration-profile", type=Path)
    real.add_argument("--calibration-manifest", type=Path)
    real.add_argument("--adapter-checkpoint", type=Path)
    real.add_argument("--input", type=Path)
    real.add_argument("--metadata", type=Path)
    real.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PipelineConfig.from_yaml(args.config)
        if args.command == "validate-config":
            print(json.dumps({"status": "valid", "config_sha256": config.sha256}, sort_keys=True))
            return 0
        if args.command == "real-run":
            if config.mode != "real":
                raise ValueError("real-run requires mode=real")
            required = {
                "calibration_profile": args.calibration_profile,
                "calibration_manifest": args.calibration_manifest,
                "adapter_checkpoint": args.adapter_checkpoint,
                "input": args.input,
                "metadata": args.metadata,
            }
            missing = [name for name, path in required.items() if path is None or not path.is_file()]
            if missing:
                raise ValueError("real-run missing required local artifacts: " + ",".join(missing))
            if config.models.samsung_checkpoint is None or not Path(config.models.samsung_checkpoint).is_file():
                raise ValueError("real-run Samsung checkpoint is unavailable")
            profile_payload = json.loads(args.calibration_profile.read_text(encoding="utf-8"))
            profile = CertificationProfile.from_mapping(profile_payload)
            if not profile.real_calibrated:
                raise ValueError("real-run requires a non-synthetic calibrated gate profile")
            raise RuntimeError(
                "real-run preflight passed, but execution requires a project-specific trained adapter artifact loader; no fallback is allowed"
            )
        report = run_synthetic_canary(config=config, output_dir=args.output_dir)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
