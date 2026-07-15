from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdjustTMCommandConfig:
    input_dir: Path
    gt_root: Path
    baseline_checkpoint: Path
    output_dir: Path
    batch_size: int = 18
    epochs: int = 30
    seed: int = 42
    image_size: int = 512
    amp: bool = False


class ClosedLoopRunner:
    @staticmethod
    def build_train_command(config: AdjustTMCommandConfig) -> list[str]:
        if config.batch_size <= 0 or config.batch_size % 18 != 0:
            raise ValueError("batch_size must be a positive multiple of 18")
        command = [
            "python", "-m", "adjustTM.train",
            "--input-dir", str(config.input_dir),
            "--gt-root", str(config.gt_root),
            "--baseline-checkpoint", str(config.baseline_checkpoint),
            "--control-method", "param_residual",
            "--output-dir", str(config.output_dir),
            "--batch-size", str(config.batch_size),
            "--epochs", str(config.epochs),
            "--seed", str(config.seed),
            "--image-size", str(config.image_size),
        ]
        if config.amp:
            command.append("--amp")
        return command

    @staticmethod
    def write_handoff_bundle(
        output_dir: str | Path,
        train_command: list[str],
        qwen_command: list[str],
        dataset_gate: dict[str, Any],
        prescription: dict[str, Any],
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "closed_loop_manifest.json"
        payload = {
            "version": 1,
            "scope": "controllable_brightness_tm_v1",
            "commands": {"train": train_command, "qwen_tmqa": qwen_command},
            "dataset_gate": dataset_gate,
            "data_prescription": prescription,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
