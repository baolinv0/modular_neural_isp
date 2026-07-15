from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .data_prescription import build_data_prescription
from .dataset_gate import DatasetGate, DatasetGateConfig
from .qwen_tmqa_adapter import (
    QwenTMQAConfig,
    QwenTMQARunner,
    load_qwen_tmqa_results,
    write_runtime_config,
)
from .runner import AdjustTMCommandConfig, ClosedLoopRunner


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _add_common_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qwen-config", default="adjustTM_closed_loop/configs/qwen_tmqa.yaml")
    parser.add_argument("--qwen-executable", default="qwen-tmqa")
    parser.add_argument("--primary-url")
    parser.add_argument("--primary-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--arbiter-url")
    parser.add_argument("--arbiter-model", default="OpenGVLab/InternVL3_5-38B")
    parser.add_argument("--enable-segmentation", action="store_true")
    parser.add_argument("--segmentation-device", default="auto")
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--amp", action="store_true")


def prepare(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_report = DatasetGate(DatasetGateConfig()).run(args.input_dir, args.gt_root)
    gate_path = output_dir / "dataset_gate.json"
    _write_json(gate_path, gate_report.to_dict())

    runtime_config = write_runtime_config(
        base_config=args.qwen_config,
        output_path=output_dir / "qwen_tmqa_runtime.yaml",
        source_dir=args.input_dir,
        primary_enabled=bool(args.primary_url),
        arbiter_enabled=bool(args.arbiter_url),
    )
    train_command = ClosedLoopRunner.build_train_command(
        AdjustTMCommandConfig(
            input_dir=Path(args.input_dir),
            gt_root=Path(args.gt_root),
            baseline_checkpoint=Path(args.baseline_checkpoint),
            output_dir=output_dir / "train",
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            image_size=args.image_size,
            amp=args.amp,
        )
    )
    qwen_config = QwenTMQAConfig(
        executable=args.qwen_executable,
        config_path=runtime_config,
        primary_url=args.primary_url,
        primary_model=args.primary_model if args.primary_url else None,
        arbiter_url=args.arbiter_url,
        arbiter_model=args.arbiter_model if args.arbiter_url else None,
        enable_segmentation=args.enable_segmentation,
        segmentation_device=args.segmentation_device,
    )
    qwen_command = QwenTMQARunner(qwen_config).build_command(args.gt_root, output_dir / "qwen_tmqa_dataset")
    manifest = ClosedLoopRunner.write_handoff_bundle(
        output_dir=output_dir,
        train_command=train_command,
        qwen_command=qwen_command,
        dataset_gate=gate_report.to_dict(),
        prescription={"tasks": []},
    )
    (output_dir / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + shlex.join(qwen_command) + "\n"
        + shlex.join(train_command) + "\n",
        encoding="utf-8",
    )
    return manifest


def prescribe(args: argparse.Namespace) -> Path:
    results = load_qwen_tmqa_results(args.qwen_output)
    prescription = build_data_prescription(results.scenes)
    output = Path(args.output)
    _write_json(output, prescription.to_dict())
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adjusttm-closed-loop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Validate Dataset V1 and create training/IQA commands")
    _add_common_prepare_args(prepare_parser)
    prescribe_parser = subparsers.add_parser("prescribe", help="Convert Qwen-TMQA failures into training-data tasks")
    prescribe_parser.add_argument("--qwen-output", required=True)
    prescribe_parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        print(prepare(args))
        return 0
    if args.command == "prescribe":
        print(prescribe(args))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
