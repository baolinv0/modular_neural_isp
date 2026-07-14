from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from adjustTM.constants import LEVELS
from adjustTM.manifest import create_or_load_split_manifest, write_sample_index
from adjustTM.metrics import luminance_ssim, trajectory_metrics
from adjustTM.model import ControlledBrightnessISP, load_control_checkpoint
from adjustTM.sampler import LevelBalancedBatchSampler
from adjustTM.transfer import linear_to_srgb
from photofinishing.photofinishing_model import PhotofinishingModule


class _TinyBaseNet(nn.Module):
    def __init__(self, output_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, output_dim),
            activation,
        )


class _TinyGainNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._input_size = 8
        self._gain_min = 0.25
        self._gain_max = 4.0
        self._gain_net = _TinyBaseNet(1, nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=True)
        scale = self._gain_net._net(x)
        gain = self._gain_min + (self._gain_max - self._gain_min) * scale
        return gain.view(-1, 1, 1, 1)


class _TinyGTMNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._input_size = 8
        self._gtm_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 3),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=True)
        return self._gtm_net(x)


class _TinyLTMNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, x_gtm: torch.Tensor, training_mode: bool = False) -> torch.Tensor:
        del training_mode
        self.calls += 1
        b, _, h, w = x.shape
        coeffs = x.new_zeros((b, 5, h, w))
        coeffs[:, 0] = -20.0
        coeffs[:, 1:4] = 1.0
        coeffs[:, 4] = self.scale
        return coeffs


class TinyBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._gain_net = _TinyGainNet()
        self._gtm_net = _TinyGTMNet()
        self._ltm_net = _TinyLTMNet()

    def forward(self, x: torch.Tensor, training_mode: bool = False) -> dict[str, torch.Tensor]:
        gain = self._gain_net(x)
        x_gain = PhotofinishingModule._apply_gain(x, gain)
        gtm = self._gtm_net(x_gain)
        x_gtm = PhotofinishingModule._apply_gtm(x_gain, gtm)
        ltm = self._ltm_net(x_gain, x_gtm, training_mode=training_mode)
        linear = PhotofinishingModule._apply_ltm(x_gain, x_gtm, ltm).clamp(0.0, 1.0)
        return {
            "output": linear_to_srgb(linear),
            "linear_output": linear,
            "gain_factor": gain,
            "gtm_params": gtm,
            "ltm_params": ltm,
        }


class _FakePairDataset:
    def __init__(self, scenes: int = 5) -> None:
        self.scene_names = [f"scene_{idx:03d}.png" for idx in range(scenes)]
        self.level_pairs = [(i, j) for i in range(9) for j in range(i + 1, 9)]
        self.samples = [
            (scene, low, high)
            for scene in self.scene_names
            for low, high in self.level_pairs
        ]

    def __len__(self) -> int:
        return len(self.samples)


def _make_model(method: str) -> ControlledBrightnessISP:
    return ControlledBrightnessISP(method, baseline=TinyBaseline(), device="cpu")


def test_level_balanced_sampler_covers_each_sample_once_and_balances_every_block() -> None:
    dataset = _FakePairDataset(scenes=5)
    sampler = LevelBalancedBatchSampler(dataset, batch_size=18, seed=17)
    sampler.set_epoch(3)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(dataset)))
    for batch in batches:
        counts: Counter[int] = Counter()
        for index in batch:
            _, low, high = dataset.samples[index]
            counts[low] += 1
            counts[high] += 1
        assert set(counts.values()) == {4}


def test_split_manifest_and_sample_index_are_deterministic(tmp_path: Path) -> None:
    scenes = [f"scene_{idx:03d}.png" for idx in range(20)]
    manifest_path = tmp_path / "split.json"
    first = create_or_load_split_manifest(scenes, manifest_path, val_fraction=0.2, seed=9)
    second = create_or_load_split_manifest(scenes, manifest_path, val_fraction=0.2, seed=999)
    assert first == second
    assert not set(first["train"]) & set(first["val"])
    assert set(first["train"]) | set(first["val"]) == set(scenes)

    dataset = _FakePairDataset(scenes=2)
    index_path = tmp_path / "index.json"
    write_sample_index(dataset, index_path)
    first_text = index_path.read_text(encoding="utf-8")
    write_sample_index(dataset, index_path)
    assert index_path.read_text(encoding="utf-8") == first_text


@torch.no_grad()
def test_luminance_ssim_and_trajectory_metrics_are_exact_for_identical_curves() -> None:
    image = torch.rand(2, 3, 16, 16)
    assert torch.allclose(luminance_ssim(image, image), torch.ones(2), atol=1e-6)
    curve = torch.linspace(-2.0, 1.0, len(LEVELS))
    result = trajectory_metrics(curve, curve)
    assert result["curve_mae"] == 0.0
    assert result["adjacent_step_mae"] == 0.0
    assert result["endpoint_range_error"] == 0.0
    assert result["spearman"] == 1.0


@torch.no_grad()
def test_alpha_zero_and_frozen_ltm_are_preserved_for_all_methods() -> None:
    x = torch.rand(2, 3, 12, 12)
    for method in ("param_residual", "parallel_adapter", "film", "dual_lora"):
        model = _make_model(method)
        controlled = model(x, torch.zeros(2), training_mode=True)["output"]
        baseline = model.forward_baseline(x, training_mode=True)["output"]
        assert torch.equal(controlled, baseline)
        assert model.baseline._ltm_net.calls >= 2
        assert all(not parameter.requires_grad for parameter in model.baseline._ltm_net.parameters())


def test_every_control_method_supports_optimizer_step_and_strict_checkpoint_round_trip(tmp_path: Path) -> None:
    x = torch.rand(2, 3, 12, 12)
    for method in ("param_residual", "parallel_adapter", "film", "dual_lora"):
        model = _make_model(method)
        baseline_before = {key: value.detach().clone() for key, value in model.baseline.state_dict().items()}
        control_before = {key: value.detach().clone() for key, value in model.control_state_dict().items()}
        optimizer = torch.optim.Adam(model.control_parameters(), lr=1e-2)
        low = model(x, torch.tensor([-0.75, -0.25]))["output"]
        high = model(x, torch.tensor([0.25, 0.75]))["output"]
        loss = low.mean() - high.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert any(
            not torch.equal(control_before[key], value)
            for key, value in model.control_state_dict().items()
        )
        assert all(torch.equal(baseline_before[key], value) for key, value in model.baseline.state_dict().items())

        path = tmp_path / f"{method}.pth"
        torch.save({"control_method": method, "control_state_dict": model.control_state_dict()}, path)
        restored = _make_model(method)
        load_control_checkpoint(restored, path)
        assert all(
            torch.equal(model.control_state_dict()[key], restored.control_state_dict()[key])
            for key in model.control_state_dict()
        )


def test_control_checkpoint_rejects_missing_keys(tmp_path: Path) -> None:
    model = _make_model("film")
    state = model.control_state_dict()
    state.pop(next(iter(state)))
    path = tmp_path / "partial.pth"
    torch.save({"control_method": "film", "control_state_dict": state}, path)
    try:
        load_control_checkpoint(model, path)
    except RuntimeError as error:
        assert "missing" in str(error).lower()
    else:
        raise AssertionError("partial control checkpoint was accepted")
