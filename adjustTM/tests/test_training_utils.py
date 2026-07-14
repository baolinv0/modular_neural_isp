from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adjustTM.model import ControlledBrightnessISP, load_baseline_checkpoint
from adjustTM.train import enforce_parameter_budget


def test_default_control_parameter_counts_are_budget_matched() -> None:
    counts = {}
    for method in ("param_residual", "parallel_adapter", "film", "dual_lora"):
        counts[method] = ControlledBrightnessISP(method).parameter_report()["trainable"]
    target = 1040
    assert all(abs(count - target) / target <= 0.10 for count in counts.values())


def test_parameter_budget_rejects_unfair_configuration() -> None:
    with pytest.raises(RuntimeError):
        enforce_parameter_budget({"trainable": 2000}, target=1000, tolerance=0.10, allow_mismatch=False)


def test_baseline_checkpoint_loads_strictly(tmp_path: Path) -> None:
    source = ControlledBrightnessISP("film")
    path = tmp_path / "baseline.pth"
    torch.save(source.baseline.state_dict(), path)
    target = ControlledBrightnessISP("dual_lora")
    load_baseline_checkpoint(target.baseline, path)
    for source_value, target_value in zip(source.baseline.state_dict().values(), target.baseline.state_dict().values()):
        assert torch.equal(source_value, target_value)
