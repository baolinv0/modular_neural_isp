from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .constants import LEVELS


def build_level_pairs() -> list[tuple[int, int]]:
    return list(itertools.combinations(range(len(LEVELS)), 2))


def discover_scene_names(input_dir: str | Path) -> list[str]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    names = sorted(path.name for path in input_dir.glob("*.png") if path.is_file())
    if not names:
        raise ValueError(f"No PNG files found in {input_dir}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate input filenames detected")
    return names


def _read_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read PNG: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel PNG, got shape {image.shape}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_linear_png16(path: str | Path) -> torch.Tensor:
    path = Path(path)
    image = _read_png(path)
    if image.dtype != np.uint16:
        raise TypeError(f"Input must be uint16 PNG, got {image.dtype}: {path}")
    image = np.ascontiguousarray(image.astype(np.float32) / 65535.0)
    return torch.from_numpy(image).permute(2, 0, 1)


def read_srgb_png(path: str | Path) -> torch.Tensor:
    path = Path(path)
    image = _read_png(path)
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"GT must be uint8 or uint16 PNG, got {image.dtype}: {path}")
    image = np.ascontiguousarray(image.astype(np.float32) / scale)
    return torch.from_numpy(image).permute(2, 0, 1)


def _resize(image: torch.Tensor, image_size: int | None) -> torch.Tensor:
    if image_size is None:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)


def _select_scenes(all_names: Sequence[str], requested: Sequence[str] | None) -> list[str]:
    if requested is None:
        return list(all_names)
    selected = list(requested)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("scene_names must be non-empty and unique")
    missing = sorted(set(selected) - set(all_names))
    if missing:
        raise FileNotFoundError(f"Requested input scenes do not exist: {missing[:10]}")
    return sorted(selected)


class MultiLevelPairDataset(Dataset):
    """Expands each selected scene across all 36 low/high brightness pairs."""

    def __init__(
        self,
        input_dir: str | Path,
        gt_root: str | Path,
        image_size: int | None = 512,
        geometric_aug: bool = False,
        seed: int = 42,
        scene_names: Sequence[str] | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.gt_root = Path(gt_root)
        self.image_size = image_size
        self.geometric_aug = geometric_aug
        self.seed = seed
        self.epoch = 0
        self.level_pairs = build_level_pairs()

        all_input_names = discover_scene_names(self.input_dir)
        selected_names = _select_scenes(all_input_names, scene_names)
        selected_set = set(selected_names)
        self.input_files = {name: self.input_dir / name for name in selected_names}
        self.gt_files: dict[str, dict[str, Path]] = {}
        full_dataset = scene_names is None
        full_input_set = set(all_input_names)

        for level_name, _ in LEVELS:
            level_dir = self.gt_root / level_name
            if not level_dir.is_dir():
                raise NotADirectoryError(level_dir)
            files = sorted(path for path in level_dir.glob("*.png") if path.is_file())
            level_names = [path.name for path in files]
            if len(level_names) != len(set(level_names)):
                raise ValueError(f"Duplicate {level_name} GT filenames detected")
            level_set = set(level_names)
            missing = sorted(selected_set - level_set)
            if missing:
                raise FileNotFoundError(f"Missing {level_name} GT files: {missing[:10]}")
            if full_dataset:
                extra = sorted(level_set - full_input_set)
                if extra:
                    raise ValueError(f"Unexpected {level_name} GT files: {extra[:10]}")
            file_map = {path.name: path for path in files}
            self.gt_files[level_name] = {name: file_map[name] for name in selected_names}

        self.scene_names = selected_names
        self.samples = [
            (scene_name, low_idx, high_idx)
            for scene_name in self.scene_names
            for low_idx, high_idx in self.level_pairs
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, images: Iterable[torch.Tensor], index: int) -> list[torch.Tensor]:
        images = list(images)
        if not self.geometric_aug:
            return images
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        if rng.random() < 0.5:
            images = [torch.flip(image, dims=(-1,)) for image in images]
        if rng.random() < 0.5:
            images = [torch.flip(image, dims=(-2,)) for image in images]
        rotations = rng.randrange(4)
        if rotations:
            images = [torch.rot90(image, rotations, dims=(-2, -1)) for image in images]
        return images

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        scene_name, low_idx, high_idx = self.samples[index]
        low_name, alpha_low = LEVELS[low_idx]
        high_name, alpha_high = LEVELS[high_idx]
        input_image = read_linear_png16(self.input_files[scene_name])
        gt_low = read_srgb_png(self.gt_files[low_name][scene_name])
        gt_high = read_srgb_png(self.gt_files[high_name][scene_name])
        if input_image.shape != gt_low.shape or input_image.shape != gt_high.shape:
            raise ValueError(
                f"Spatial mismatch for {scene_name}: input={tuple(input_image.shape)}, "
                f"low={tuple(gt_low.shape)}, high={tuple(gt_high.shape)}"
            )
        input_image, gt_low, gt_high = self._augment((input_image, gt_low, gt_high), index)
        input_image = _resize(input_image, self.image_size)
        gt_low = _resize(gt_low, self.image_size)
        gt_high = _resize(gt_high, self.image_size)
        return {
            "in_image": input_image,
            "gt_low": gt_low,
            "gt_high": gt_high,
            "alpha_low": torch.tensor(alpha_low, dtype=torch.float32),
            "alpha_high": torch.tensor(alpha_high, dtype=torch.float32),
            "scene_name": scene_name,
            "level_low": low_name,
            "level_high": high_name,
        }


class MultiLevelDataset(Dataset):
    """Each selected scene-level combination appears exactly once."""

    def __init__(
        self,
        input_dir: str | Path,
        gt_root: str | Path,
        image_size: int | None = 512,
        scene_names: Sequence[str] | None = None,
    ) -> None:
        pair_dataset = MultiLevelPairDataset(
            input_dir=input_dir,
            gt_root=gt_root,
            image_size=image_size,
            geometric_aug=False,
            scene_names=scene_names,
        )
        self.input_files = pair_dataset.input_files
        self.gt_files = pair_dataset.gt_files
        self.scene_names = pair_dataset.scene_names
        self.image_size = image_size
        self.samples = [
            (scene_name, level_idx)
            for scene_name in self.scene_names
            for level_idx in range(len(LEVELS))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        scene_name, level_idx = self.samples[index]
        level_name, alpha = LEVELS[level_idx]
        input_image = read_linear_png16(self.input_files[scene_name])
        gt = read_srgb_png(self.gt_files[level_name][scene_name])
        if input_image.shape != gt.shape:
            raise ValueError(
                f"Spatial mismatch for {scene_name}: input={tuple(input_image.shape)}, gt={tuple(gt.shape)}"
            )
        return {
            "in_image": _resize(input_image, self.image_size),
            "gt_image": _resize(gt, self.image_size),
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "scene_name": scene_name,
            "level_name": level_name,
        }
