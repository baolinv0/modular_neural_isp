from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from adjustTM.constants import LEVELS
from adjustTM.dataset import MultiLevelPairDataset, build_level_pairs, read_linear_png16
from adjustTM.losses import BrightnessOnlyLoss, monotonic_hinge_loss
from adjustTM.model import ControlledBrightnessISP
from adjustTM.transfer import linear_to_srgb, srgb_to_linear


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = image[..., ::-1] if image.ndim == 3 else image
    ok = cv2.imwrite(str(path), out)
    assert ok


def _make_data(root: Path, scenes: int = 2, h: int = 16, w: int = 20) -> tuple[Path, Path]:
    input_dir = root / "input_linear"
    gt_root = root / "gt_levels"
    for scene_idx in range(scenes):
        name = f"scene_{scene_idx:04d}.png"
        linear = np.full((h, w, 3), 12000 + scene_idx * 1000, dtype=np.uint16)
        _write_png(input_dir / name, linear)
        for level_idx, (level_name, alpha) in enumerate(LEVELS):
            value = np.clip(80 + level_idx * 15 + scene_idx, 0, 255)
            gt = np.full((h, w, 3), value, dtype=np.uint8)
            _write_png(gt_root / level_name / name, gt)
    return input_dir, gt_root


def test_linear_png16_is_normalized_without_color_processing(tmp_path: Path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint16)
    image[..., 0] = 65535
    image[..., 1] = 32768
    path = tmp_path / "linear.png"
    _write_png(path, image)

    tensor = read_linear_png16(path)

    assert tensor.shape == (3, 4, 5)
    assert tensor.dtype == torch.float32
    assert tensor[0, 0, 0].item() == pytest.approx(1.0)
    assert tensor[1, 0, 0].item() == pytest.approx(32768 / 65535.0)
    assert tensor[2, 0, 0].item() == pytest.approx(0.0)


def test_all_36_pairs_have_equal_level_marginals() -> None:
    pairs = build_level_pairs()
    counts: Counter[int] = Counter()
    assert len(pairs) == 36
    for low_idx, high_idx in pairs:
        assert low_idx < high_idx
        counts[low_idx] += 1
        counts[high_idx] += 1
    assert set(counts.values()) == {8}


def test_dataset_matches_by_filename_and_returns_same_scene_pairs(tmp_path: Path) -> None:
    input_dir, gt_root = _make_data(tmp_path, scenes=2)
    dataset = MultiLevelPairDataset(input_dir=input_dir, gt_root=gt_root, image_size=None)

    assert len(dataset) == 2 * 36
    sample = dataset[0]
    assert sample["in_image"].shape == sample["gt_low"].shape == sample["gt_high"].shape
    assert sample["alpha_low"].item() < sample["alpha_high"].item()
    assert sample["scene_name"].startswith("scene_")


def test_dataset_fails_on_missing_level_file(tmp_path: Path) -> None:
    input_dir, gt_root = _make_data(tmp_path, scenes=1)
    (gt_root / "a_p100" / "scene_0000.png").unlink()
    with pytest.raises(FileNotFoundError):
        MultiLevelPairDataset(input_dir=input_dir, gt_root=gt_root)


def test_srgb_oetf_round_trip() -> None:
    x = torch.tensor([0.0, 0.0031308, 0.18, 1.0])
    encoded = linear_to_srgb(x)
    decoded = srgb_to_linear(encoded)
    assert torch.allclose(decoded, x, atol=2e-6, rtol=1e-5)
    assert encoded[1].item() == pytest.approx(12.92 * 0.0031308, rel=1e-5)


@pytest.mark.parametrize("method", ["param_residual", "parallel_adapter", "film", "dual_lora"])
def test_alpha_zero_is_exact_baseline_for_all_methods(method: str) -> None:
    torch.manual_seed(7)
    model = ControlledBrightnessISP(control_method=method, device=torch.device("cpu"))
    x = torch.rand(2, 3, 32, 32)
    alpha = torch.zeros(2)

    with torch.no_grad():
        controlled = model(x, alpha, training_mode=True)["output"]
        baseline = model.forward_baseline(x, training_mode=True)["output"]

    assert torch.equal(controlled, baseline)


@pytest.mark.parametrize("method", ["param_residual", "parallel_adapter", "film", "dual_lora"])
def test_only_control_parameters_are_trainable(method: str) -> None:
    model = ControlledBrightnessISP(control_method=method, device=torch.device("cpu"))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("control" in name for name in trainable)
    assert all(not parameter.requires_grad for parameter in model.baseline._ltm_net.parameters())


def test_monotonic_hinge_penalizes_wrong_direction() -> None:
    low = torch.full((2, 3, 8, 8), 0.6)
    high = torch.full((2, 3, 8, 8), 0.4)
    alpha_low = torch.tensor([-0.5, 0.0])
    alpha_high = torch.tensor([0.0, 0.5])
    loss = monotonic_hinge_loss(low, high, alpha_low, alpha_high, margin_per_alpha=0.02)
    assert loss.item() > 0


def test_brightness_loss_backpropagates_through_control_parameters() -> None:
    model = ControlledBrightnessISP(control_method="param_residual", device=torch.device("cpu"))
    objective = BrightnessOnlyLoss()
    x = torch.rand(2, 3, 32, 32)
    low = model(x, torch.tensor([-0.5, -0.25]), training_mode=True)
    high = model(x, torch.tensor([0.5, 0.75]), training_mode=True)
    gt_low = low["output"].detach() * 0.9
    gt_high = high["output"].detach().clamp(0, 1)

    result = objective(
        pred_low=low["output"],
        pred_high=high["output"],
        gt_low=gt_low,
        gt_high=gt_high,
        alpha_low=torch.tensor([-0.5, -0.25]),
        alpha_high=torch.tensor([0.5, 0.75]),
        baseline_zero=None,
        pred_zero=None,
    )
    result["total"].backward()
    assert any(p.grad is not None for p in model.control_parameters())
