from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from adjustTM_closed_loop.bootstrap_qwen_tmqa import extract_qwen_tmqa_archive
from adjustTM_closed_loop.cli import main
from adjustTM_closed_loop.qwen_tmqa_adapter import write_runtime_config


def test_runtime_config_injects_linear_source_and_vlm_settings(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("dataset:\n  source_dir: null\nvlm:\n  primary:\n    enabled: false\n", encoding="utf-8")
    output = tmp_path / "runtime.yaml"
    write_runtime_config(
        base_config=base,
        output_path=output,
        source_dir=tmp_path / "linear",
        primary_enabled=True,
        arbiter_enabled=False,
    )
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["dataset"]["source_dir"] == str(tmp_path / "linear")
    assert payload["vlm"]["primary"]["enabled"] is True
    assert payload["vlm"]["arbiter"]["enabled"] is False


def test_bootstrap_extracts_nested_qwen_tmqa_root(tmp_path: Path) -> None:
    archive = tmp_path / "qwen.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("qwen_tmqa/pyproject.toml", "[project]\nname='qwen-tmqa'\n")
        handle.writestr("qwen_tmqa/src/qwen_tmqa/__init__.py", "")
    extracted = extract_qwen_tmqa_archive(archive, tmp_path / "vendor")
    assert extracted == tmp_path / "vendor" / "qwen_tmqa"
    assert (extracted / "pyproject.toml").is_file()


def test_cli_prepare_writes_dataset_gate_and_commands(tmp_path: Path, monkeypatch) -> None:
    from adjustTM_closed_loop.tests.test_dataset_gate import _make_dataset

    input_dir, gt_root = _make_dataset(tmp_path / "data", [0.08, 0.11, 0.15, 0.20, 0.26, 0.34, 0.44, 0.56, 0.70])
    base_config = tmp_path / "qwen.yaml"
    base_config.write_text("dataset: {source_dir: null}\nvlm: {primary: {enabled: false}, arbiter: {enabled: false}}\n", encoding="utf-8")
    output = tmp_path / "out"
    argv = [
        "adjusttm-closed-loop", "prepare",
        "--input-dir", str(input_dir), "--gt-root", str(gt_root),
        "--baseline-checkpoint", str(tmp_path / "base.pth"),
        "--output-dir", str(output), "--qwen-config", str(base_config),
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    manifest = json.loads((output / "closed_loop_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_gate"]["summary"]["counts"]["clean"] == 1
    assert manifest["commands"]["train"][0:3] == ["python", "-m", "adjustTM.train"]
    assert manifest["commands"]["qwen_tmqa"][0:2] == ["qwen-tmqa", "evaluate"]
    assert (output / "dataset_gate.json").is_file()
    assert (output / "qwen_tmqa_runtime.yaml").is_file()
