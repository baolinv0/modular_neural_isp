from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import QwenSceneResult, QwenTMQAResults


@dataclass(frozen=True)
class QwenTMQAConfig:
    executable: str = "qwen-tmqa"
    config_path: Path = Path("adjustTM_closed_loop/configs/qwen_tmqa.yaml")
    primary_url: str | None = None
    primary_model: str | None = None
    arbiter_url: str | None = None
    arbiter_model: str | None = None
    enable_segmentation: bool = False
    segmentation_device: str = "auto"


class QwenTMQARunner:
    def __init__(self, config: QwenTMQAConfig) -> None:
        self.config = config

    def build_command(self, dataset_root: str | Path, output_dir: str | Path) -> list[str]:
        command = [
            self.config.executable,
            "evaluate",
            "--root", str(Path(dataset_root)),
            "--output", str(Path(output_dir)),
            "--config", str(self.config.config_path),
        ]
        if self.config.enable_segmentation:
            command.extend(["--enable-segmentation", "--segmentation-device", self.config.segmentation_device])
        if self.config.primary_url:
            if not self.config.primary_model:
                raise ValueError("primary_model is required when primary_url is set")
            command.extend(["--primary-url", self.config.primary_url, "--primary-model", self.config.primary_model])
        if self.config.arbiter_url:
            if not self.config.arbiter_model:
                raise ValueError("arbiter_model is required when arbiter_url is set")
            command.extend(["--arbiter-url", self.config.arbiter_url, "--arbiter-model", self.config.arbiter_model])
        return command

    def run(self, dataset_root: str | Path, output_dir: str | Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.build_command(dataset_root, output_dir),
            check=check,
            text=True,
            capture_output=True,
        )


def _extract_reasons(payload: dict[str, Any]) -> tuple[str, ...]:
    direct = payload.get("failure_reasons")
    if isinstance(direct, list):
        return tuple(str(item) for item in direct)
    hard_gate = payload.get("hard_gate")
    if isinstance(hard_gate, dict) and isinstance(hard_gate.get("reasons"), list):
        return tuple(str(item) for item in hard_gate["reasons"])
    decision = payload.get("decision")
    if isinstance(decision, dict) and isinstance(decision.get("reasons"), list):
        return tuple(str(item) for item in decision["reasons"])
    return ()


def load_qwen_tmqa_results(output_dir: str | Path) -> QwenTMQAResults:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Qwen-TMQA summary.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scene_dir = output_dir / "scenes"
    scenes: dict[str, QwenSceneResult] = {}
    if scene_dir.is_dir():
        for path in sorted(scene_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            scene_id = str(payload.get("scene_id") or payload.get("scene_name") or path.name.removesuffix(".json"))
            action = str(payload.get("action") or payload.get("final_action") or payload.get("decision", {}).get("action") or "REVIEW")
            scenes[scene_id] = QwenSceneResult(scene_id, action.upper(), _extract_reasons(payload), payload)
    return QwenTMQAResults(summary=summary, scenes=scenes)


def write_runtime_config(
    *,
    base_config: str | Path,
    output_path: str | Path,
    source_dir: str | Path,
    primary_enabled: bool,
    arbiter_enabled: bool,
) -> Path:
    base_config = Path(base_config)
    output_path = Path(output_path)
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    dataset = payload.setdefault("dataset", {})
    dataset["source_dir"] = str(Path(source_dir))
    vlm = payload.setdefault("vlm", {})
    vlm.setdefault("primary", {})["enabled"] = bool(primary_enabled)
    vlm.setdefault("arbiter", {})["enabled"] = bool(arbiter_enabled)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path
