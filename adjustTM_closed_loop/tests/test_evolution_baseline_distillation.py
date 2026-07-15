from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from adjustTM_closed_loop.evolution.baseline_distillation import (
    DistillationLossConfig,
    select_trainable_baseline_modules,
    weighted_distillation_loss,
)


def test_distillation_loss_is_zero_for_matching_images() -> None:
    target = torch.full((2, 3, 8, 8), 0.4)
    loss, parts = weighted_distillation_loss(
        target.clone(), target, torch.ones(2), DistillationLossConfig()
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert all(value == pytest.approx(0.0, abs=1e-8) for value in parts.values())


def test_sample_weight_suppresses_low_confidence_teacher() -> None:
    target = torch.zeros(2, 3, 4, 4)
    prediction = target.clone()
    prediction[0] = 1.0
    high_weight, _ = weighted_distillation_loss(
        prediction, target, torch.tensor([1.0, 1.0]), DistillationLossConfig(lambda_gradient=0, lambda_chroma=0)
    )
    low_weight, _ = weighted_distillation_loss(
        prediction, target, torch.tensor([0.01, 1.0]), DistillationLossConfig(lambda_gradient=0, lambda_chroma=0)
    )
    assert low_weight < high_weight


class DummyBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._gain_net = nn.Linear(2, 2)
        self._gtm_net = nn.Linear(2, 2)
        self._ltm_net = nn.Linear(2, 2)


def test_default_trainable_scope_is_gain_and_gtm_only() -> None:
    model = DummyBaseline()
    report = select_trainable_baseline_modules(model, ("gain", "gtm"))
    assert report["gain"] > 0
    assert report["gtm"] > 0
    assert report["ltm"] == 0
    assert all(parameter.requires_grad for parameter in model._gain_net.parameters())
    assert all(parameter.requires_grad for parameter in model._gtm_net.parameters())
    assert not any(parameter.requires_grad for parameter in model._ltm_net.parameters())


def test_unknown_trainable_module_fails() -> None:
    with pytest.raises(ValueError, match="Unknown baseline module"):
        select_trainable_baseline_modules(DummyBaseline(), ("color",))


def _write_rgb(path, value, dtype):
    import cv2
    import numpy as np
    from pathlib import Path
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    maximum = np.iinfo(dtype).max
    image = np.full((8, 8, 3), round(value * maximum), dtype=dtype)
    assert cv2.imwrite(str(path), image)


def test_fixed_baseline_distillation_writes_reloadable_checkpoint(tmp_path, monkeypatch) -> None:
    import json
    import sys
    import types
    import numpy as np
    from pathlib import Path

    from adjustTM_closed_loop.evolution.baseline_distillation import BaselineDistillationConfig, distill_fixed_baseline
    from adjustTM_closed_loop.evolution.schemas import DistributionStatus, TeacherRecord

    class TinyBaseline(nn.Module):
        def __init__(self, device="cpu"):
            super().__init__()
            self._gain_net = nn.Conv2d(3, 3, 1, bias=False)
            self._gtm_net = nn.Conv2d(3, 3, 1, bias=False)
            self._ltm_net = nn.Conv2d(3, 3, 1, bias=False)
            with torch.no_grad():
                eye = torch.eye(3).view(3, 3, 1, 1)
                self._gain_net.weight.copy_(eye)
                self._gtm_net.weight.copy_(eye)
                self._ltm_net.weight.copy_(eye)
            self.to(device)

        def forward(self, x):
            y = self._ltm_net(self._gtm_net(self._gain_net(x))).clamp(0, 1)
            return {"output": y}

    def load_checkpoint(model, path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        state = payload.get("state_dict", payload)
        model.load_state_dict(state)

    package = types.ModuleType("adjustTM")
    package.__path__ = []
    module = types.ModuleType("adjustTM.model")
    module.LuminanceOnlyBaseline = TinyBaseline
    module.load_baseline_checkpoint = load_checkpoint
    monkeypatch.setitem(sys.modules, "adjustTM", package)
    monkeypatch.setitem(sys.modules, "adjustTM.model", module)

    baseline = TinyBaseline()
    baseline_path = tmp_path / "baseline.pth"
    torch.save({"state_dict": baseline.state_dict()}, baseline_path)
    rows = []
    for index, (split, status, source_value, target_value) in enumerate([
        ("train", "improved", .2, .35),
        ("train", "baseline_anchor", .6, .6),
        ("train", "improved", .3, .42),
        ("validation", "baseline_anchor", .7, .7),
    ]):
        source = tmp_path / "input" / f"s{index}.png"
        target = tmp_path / "target" / f"s{index}.png"
        _write_rgb(source, source_value, np.uint16)
        _write_rgb(target, target_value, np.uint16)
        record = TeacherRecord(
            scene_id=f"s{index}.png", input_path=str(source), baseline_path=str(target), target_path=str(target),
            selected_alpha=.25 if status == "improved" else 0, selected_level="x", baseline_score=.5,
            selected_score=.7 if status == "improved" else .5, score_delta=.2 if status == "improved" else 0,
            confidence=1, sample_weight=1, status=status, split=split,
            distribution_status=DistributionStatus.IN_DOMAIN, reason="test", metadata={},
        )
        rows.append(record.to_dict())
    manifest = tmp_path / "teacher.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = distill_fixed_baseline(BaselineDistillationConfig(
        teacher_manifest=manifest, baseline_checkpoint=baseline_path, output_dir=tmp_path / "out",
        image_size=8, batch_size=2, epochs=2, learning_rate=1e-2, device="cpu",
    ))
    checkpoint = Path(report["checkpoint"])
    assert checkpoint.is_file()
    reloaded = TinyBaseline()
    load_checkpoint(reloaded, checkpoint)
    assert report["train_improved_count"] == 2
    assert report["train_anchor_count"] == 1
