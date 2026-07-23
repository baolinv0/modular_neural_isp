from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from .canary import run_synthetic_canary
from .canonicalization import DeviceCanonicalizer
from .config import PipelineConfig
from .contracts import LinearMetadata, canonical_tensor_sha256
from .phase1 import FrozenSamsungTM
from .phase1_data import file_sha256, load_source_manifest, manifest_sha256
from .phase1_protocol import Phase1TrainingConfig, train_phase1
from .phase1_remediation import (
    DEFAULT_ALIGNMENT_POLICY,
    evaluate_hardened_phase1_artifact,
    load_calibration_manifest_strict,
    load_hardened_phase1_artifact,
    run_hardened_phase1_inference,
    seal_phase1_artifact,
)


class _SamsungTrainingForward(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        return self.model(image, training_mode=True)


def _load_samsung_tm(checkpoint: Path) -> tuple[FrozenSamsungTM, str]:
    if not checkpoint.is_file():
        raise ValueError("Samsung checkpoint is unavailable")
    from photofinishing.photofinishing_model import PhotofinishingModule

    model = PhotofinishingModule(device=torch.device("cpu"), use_3d_lut=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return FrozenSamsungTM(_SamsungTrainingForward(model)), file_sha256(checkpoint)


def _load_input_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and set(payload) == {"tensor"}:
        payload = payload["tensor"]
    if not torch.is_tensor(payload):
        raise ValueError("input artifact must contain a tensor")
    image = payload.detach().to(dtype=torch.float32, device="cpu")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("input tensor must have shape [1,3,H,W]")
    if not torch.isfinite(image).all() or image.min().item() < 0:
        raise ValueError("input tensor must contain finite non-negative linear RGB")
    return image


def _load_metadata(path: Path) -> LinearMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("metadata must be a JSON object")
    for field in (
        "is_normalized",
        "black_level_corrected",
        "white_balanced",
        "awb_gains_comparable",
        "metadata_complete",
    ):
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"metadata field {field} must be a JSON boolean")
    return LinearMetadata.from_mapping(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-camera Samsung-style adaptation v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True, type=Path)
    canary = subparsers.add_parser("synthetic-canary")
    canary.add_argument("--config", required=True, type=Path)
    canary.add_argument("--output-dir", required=True, type=Path)

    train = subparsers.add_parser("train-phase1")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--source-manifest", required=True, type=Path)
    train.add_argument("--calibration-manifest", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--solver-steps", type=int, default=24)

    evaluate = subparsers.add_parser("evaluate-phase1")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--calibration-manifest", required=True, type=Path)
    evaluate.add_argument("--adapter-checkpoint", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    real = subparsers.add_parser("real-run")
    real.add_argument("--config", required=True, type=Path)
    real.add_argument("--adapter-checkpoint", required=True, type=Path)
    real.add_argument("--input", required=True, type=Path)
    real.add_argument("--metadata", required=True, type=Path)
    real.add_argument("--output-dir", required=True, type=Path)
    return parser


def _require_real_config(config: PipelineConfig) -> Path:
    if config.mode != "real":
        raise ValueError("this command requires mode=real")
    if config.models.samsung_checkpoint is None:
        raise ValueError("real mode requires an explicit Samsung checkpoint")
    return Path(config.models.samsung_checkpoint)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PipelineConfig.from_yaml(args.config)
        if args.command == "validate-config":
            print(json.dumps({"status": "valid", "config_sha256": config.sha256}, sort_keys=True))
            return 0
        if args.command == "synthetic-canary":
            report = run_synthetic_canary(config=config, output_dir=args.output_dir)
            print(json.dumps(report, sort_keys=True))
            return 0

        checkpoint = _require_real_config(config)
        frozen_tm, model_sha = _load_samsung_tm(checkpoint)
        canonicalizer = DeviceCanonicalizer(config.canonicalization)
        alignment_policy = DEFAULT_ALIGNMENT_POLICY

        if args.command == "train-phase1":
            source_examples = load_source_manifest(args.source_manifest)
            calibration_examples = load_calibration_manifest_strict(
                args.calibration_manifest,
                policy=alignment_policy,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = args.output_dir / "phase1_adapter.pt"
            result = train_phase1(
                source_examples=source_examples,
                calibration_examples=calibration_examples,
                frozen_tm=frozen_tm,
                samsung_model_sha256=model_sha,
                source_manifest_sha256=manifest_sha256(args.source_manifest),
                calibration_manifest_sha256=manifest_sha256(args.calibration_manifest),
                artifact_path=artifact_path,
                config=Phase1TrainingConfig(
                    solver_steps=args.solver_steps,
                    seed=config.seed,
                    data_mode="real",
                ),
                canonicalizer=canonicalizer,
            )
            hardened = seal_phase1_artifact(
                artifact_path,
                calibration_examples=calibration_examples,
                canonicalizer=canonicalizer,
                alignment_policy=alignment_policy,
                frozen_tm=frozen_tm,
                expected_model_sha256=model_sha,
            )
            report_payload = {
                **asdict(result.report),
                "canonicalization_sha256": hardened.canonicalization_sha256,
                "alignment_policy_sha256": hardened.alignment_policy_sha256,
                "max_support_distance": hardened.max_support_distance,
                "minimum_parameter_bound_margin": hardened.minimum_parameter_bound_margin,
                "real_phase1_calibration_accepted": hardened.real_phase1_calibration_accepted,
                "real_source_replay_verified": hardened.real_source_replay_verified,
                "real_target_effectiveness_verified": hardened.real_target_effectiveness_verified,
            }
            (args.output_dir / "phase1_training_report.json").write_text(
                json.dumps(report_payload, sort_keys=True, indent=2), encoding="utf-8"
            )
            print(json.dumps(report_payload, sort_keys=True))
            return 0 if result.report.passed else 3

        artifact = load_hardened_phase1_artifact(
            args.adapter_checkpoint,
            expected_model_sha256=model_sha,
            expected_canonicalization_sha256=canonicalizer.config.sha256,
            expected_alignment_policy_sha256=alignment_policy.sha256,
        )
        if args.command == "evaluate-phase1":
            calibration_sha = manifest_sha256(args.calibration_manifest)
            if calibration_sha != artifact.calibration_manifest_sha256:
                raise ValueError("calibration manifest does not match the Phase 1 artifact")
            calibration_examples = load_calibration_manifest_strict(
                args.calibration_manifest,
                policy=artifact.alignment_policy,
            )
            report = evaluate_hardened_phase1_artifact(
                calibration_examples=calibration_examples,
                frozen_tm=frozen_tm,
                artifact=artifact,
                canonicalizer=canonicalizer,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "phase1_evaluation_report.json").write_text(
                json.dumps(report, sort_keys=True, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, sort_keys=True))
            passed = (
                report["phase1_artifact_passed"]
                and report["locked_median_improvement"] > 0.0
                and report["minimum_parameter_bound_margin"] > 0.0
            )
            return 0 if passed else 3

        image = _load_input_tensor(args.input)
        metadata = _load_metadata(args.metadata)
        output, run_manifest = run_hardened_phase1_inference(
            image=image,
            metadata=metadata,
            frozen_tm=frozen_tm,
            artifact=artifact,
            canonicalizer=canonicalizer,
            require_real_artifact=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "phase1_output.pt"
        torch.save(output, output_path)
        run_manifest.update(
            {
                "config_sha256": config.sha256,
                "adapter_artifact_sha256": file_sha256(args.adapter_checkpoint),
                "input_sha256": canonical_tensor_sha256(image),
                "output_sha256": canonical_tensor_sha256(output),
                "phase2_status": "PHASE2_NOT_IMPLEMENTED",
            }
        )
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        print(json.dumps(run_manifest, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
