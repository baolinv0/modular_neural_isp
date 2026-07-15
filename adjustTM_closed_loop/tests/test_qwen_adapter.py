from __future__ import annotations

import json
from pathlib import Path

import pytest

from adjustTM_closed_loop.qwen_tmqa_adapter import QwenTMQAConfig, QwenTMQARunner, load_qwen_tmqa_results


def test_build_command_uses_openai_compatible_models(tmp_path: Path) -> None:
    config = QwenTMQAConfig(
        executable="qwen-tmqa",
        config_path=Path("cfg.yaml"),
        primary_url="http://127.0.0.1:8000/v1",
        primary_model="Qwen/Qwen3-VL-8B-Instruct",
        arbiter_url="http://127.0.0.1:8001/v1",
        arbiter_model="OpenGVLab/InternVL3_5-38B",
    )
    command = QwenTMQARunner(config).build_command(tmp_path / "dataset", tmp_path / "out")
    assert command[:2] == ["qwen-tmqa", "evaluate"]
    assert "--primary-url" in command
    assert "Qwen/Qwen3-VL-8B-Instruct" in command
    assert "--arbiter-url" in command
    assert "OpenGVLab/InternVL3_5-38B" in command


def test_command_omits_vlm_flags_when_disabled(tmp_path: Path) -> None:
    config = QwenTMQAConfig(executable="qwen-tmqa", config_path=Path("cfg.yaml"))
    command = QwenTMQARunner(config).build_command(tmp_path / "dataset", tmp_path / "out")
    assert "--primary-url" not in command
    assert "--arbiter-url" not in command


def test_load_results_normalizes_scene_actions_and_failures(tmp_path: Path) -> None:
    output = tmp_path / "qwen"
    (output / "scenes").mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps({"scene_count": 2}), encoding="utf-8")
    (output / "scenes" / "scene_a.png.json").write_text(
        json.dumps({"scene_id": "scene_a.png", "action": "KEEP", "failure_reasons": []}), encoding="utf-8"
    )
    (output / "scenes" / "scene_b.png.json").write_text(
        json.dumps({"scene_id": "scene_b.png", "final_action": "REGENERATE", "hard_gate": {"reasons": ["max_luminance_clip_ratio", "color_drift"]}}),
        encoding="utf-8",
    )
    result = load_qwen_tmqa_results(output)
    assert result.summary["scene_count"] == 2
    assert result.scenes["scene_a.png"].action == "KEEP"
    assert result.scenes["scene_b.png"].action == "REGENERATE"
    assert result.scenes["scene_b.png"].failure_reasons == ("max_luminance_clip_ratio", "color_drift")


def test_load_results_requires_summary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="summary.json"):
        load_qwen_tmqa_results(tmp_path)
