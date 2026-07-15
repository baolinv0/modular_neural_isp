from __future__ import annotations

import json
from pathlib import Path

from adjustTM_closed_loop.runner import AdjustTMCommandConfig, ClosedLoopRunner


def test_param_residual_train_command_is_frozen_scope(tmp_path: Path) -> None:
    config = AdjustTMCommandConfig(
        input_dir=tmp_path / "input",
        gt_root=tmp_path / "gt",
        baseline_checkpoint=tmp_path / "base.pth",
        output_dir=tmp_path / "run",
    )
    command = ClosedLoopRunner.build_train_command(config)
    assert command[:3] == ["python", "-m", "adjustTM.train"]
    assert command[command.index("--control-method") + 1] == "param_residual"
    assert command[command.index("--batch-size") + 1] == "18"
    assert "--allow-parameter-mismatch" not in command


def test_write_handoff_bundle_contains_commands_and_prescription(tmp_path: Path) -> None:
    runner = ClosedLoopRunner()
    bundle = runner.write_handoff_bundle(
        output_dir=tmp_path,
        train_command=["python", "-m", "adjustTM.train"],
        qwen_command=["qwen-tmqa", "evaluate"],
        dataset_gate={"counts": {"clean": 10}},
        prescription={"tasks": []},
    )
    payload = json.loads((tmp_path / "closed_loop_manifest.json").read_text(encoding="utf-8"))
    assert payload["commands"]["train"][0] == "python"
    assert payload["commands"]["qwen_tmqa"][0] == "qwen-tmqa"
    assert Path(bundle).name == "closed_loop_manifest.json"
