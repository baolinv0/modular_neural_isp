"""Balanced scene-wise multi-level data loading for brightness control."""
from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

LEVELS: tuple[tuple[str, float], ...] = (
    ("a_m100", -1.00),
    ("a_m075", -0.75),
    ("a_m050", -0.50),
    ("a_m025", -0.25),
    ("a_000", 0.00),
    ("a_p025", 0.25),
    ("a_p050", 0.50),
    ("a_p075", 0.75),
    ("a_p100", 1.00),
)
PNG_SUFFIXES = {".png"}


def _read_png(path: Path, expected_dtype: np.dtype) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read PNG: {path}")
    if image.dtype != expected_dtype:
        raise TypeError(f"Expected {expected_dtype} at {path}, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected three-channel RGB PNG at {path}, got {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_linear_png(path: Path | str) -> torch.Tensor:
    image = _read_png(Path(path), np.uint16).astype(np.float32) / 65535.0
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def read_srgb_png(path: Path | str) -> torch.Tensor:
    image = _read_png(Path(path), np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def build_balanced_pair_schedule(num_scenes: int, seed: int = 0) -> list[tuple[int, int, int]]:
    if num_scenes <= 0:
        raise ValueError("num_scenes must be positive")
    pairs = list(itertools.combinations(range(len(LEVELS)), 2))
    schedule = [(scene_idx, low_idx, high_idx)
                for scene_idx in range(num_scenes)
                for low_idx, high_idx in pairs]
    random.Random(seed).shuffle(schedule)
    return schedule


def _resize_chw(image: torch.Tensor, size: Optional[int]) -> torch.Tensor:
    if size is None:
        return image
    if size <= 0:
        raise ValueError("image_size must be positive or None")
    return F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear",
                         align_corners=False).squeeze(0)


def _augment_triplet(image: torch.Tensor, target_low: torch.Tensor,
                     target_high: torch.Tensor, code: int):
    def apply(x: torch.Tensor) -> torch.Tensor:
        if code & 1:
            x = torch.flip(x, dims=(-1,))
        if code & 2:
            x = torch.flip(x, dims=(-2,))
        rotations = (code >> 2) % 4
        if rotations:
            x = torch.rot90(x, rotations, dims=(-2, -1))
        return x.contiguous()
    return apply(image), apply(target_low), apply(target_high)


class BrightnessPairDataset(Dataset):
    """Each epoch contains every scene and all 36 ordered level pairs exactly once."""

    def __init__(self, input_dir: Path | str, gt_root: Path | str, *,
                 image_size: Optional[int] = None, seed: int = 0,
                 geometric_aug: bool = False) -> None:
        self.input_dir = Path(input_dir)
        self.gt_root = Path(gt_root)
        self.image_size = image_size
        self.seed = seed
        self.geometric_aug = geometric_aug
        self.epoch = 0

        if not self.input_dir.is_dir():
            raise FileNotFoundError(self.input_dir)
        input_names = {p.name for p in self.input_dir.iterdir()
                       if p.suffix.lower() in PNG_SUFFIXES}
        if not input_names:
            raise RuntimeError(f"No PNG files found in {self.input_dir}")

        for level_name, _ in LEVELS:
            level_dir = self.gt_root / level_name
            if not level_dir.is_dir():
                raise FileNotFoundError(level_dir)
            level_names = {p.name for p in level_dir.iterdir()
                           if p.suffix.lower() in PNG_SUFFIXES}
            if level_names != input_names:
                missing = sorted(input_names - level_names)
                extra = sorted(level_names - input_names)
                raise RuntimeError(
                    f"Filename mismatch for {level_name}: missing={missing[:5]}, extra={extra[:5]}"
                )

        self.scene_names = sorted(input_names)
        self._schedule = build_balanced_pair_schedule(len(self.scene_names), seed=self.seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._schedule = build_balanced_pair_schedule(
            len(self.scene_names), seed=self.seed + self.epoch)

    def __len__(self) -> int:
        return len(self._schedule)

    def __getitem__(self, index: int):
        scene_idx, low_idx, high_idx = self._schedule[index]
        scene_name = self.scene_names[scene_idx]
        low_name, low_alpha = LEVELS[low_idx]
        high_name, high_alpha = LEVELS[high_idx]

        image = _resize_chw(read_linear_png(self.input_dir / scene_name), self.image_size)
        target_low = _resize_chw(
            read_srgb_png(self.gt_root / low_name / scene_name), self.image_size)
        target_high = _resize_chw(
            read_srgb_png(self.gt_root / high_name / scene_name), self.image_size)
        target_anchor = _resize_chw(
            read_srgb_png(self.gt_root / "a_000" / scene_name), self.image_size)

        if self.geometric_aug:
            code = random.Random((self.seed + self.epoch) * 1_000_003 + index).randrange(16)
            image, target_low, target_high = _augment_triplet(
                image, target_low, target_high, code)
            target_anchor = _augment_triplet(
                target_anchor, target_anchor, target_anchor, code)[0]

        return {
            "input": image,
            "target_low": target_low,
            "target_high": target_high,
            "target_anchor": target_anchor,
            "alpha_low": torch.tensor([low_alpha], dtype=torch.float32),
            "alpha_high": torch.tensor([high_alpha], dtype=torch.float32),
            "scene_name": scene_name,
            "level_low": low_name,
            "level_high": high_name,
        }


class BrightnessSceneDataset(Dataset):
    """Returns one input and all nine level targets for common evaluation."""

    def __init__(self, input_dir: Path | str, gt_root: Path | str, *,
                 image_size: Optional[int] = None) -> None:
        self.input_dir = Path(input_dir)
        self.gt_root = Path(gt_root)
        self.image_size = image_size
        pair_dataset = BrightnessPairDataset(
            self.input_dir, self.gt_root, image_size=image_size,
            seed=0, geometric_aug=False)
        self.scene_names = pair_dataset.scene_names

    def __len__(self) -> int:
        return len(self.scene_names)

    def __getitem__(self, index: int):
        scene_name = self.scene_names[index]
        image = _resize_chw(read_linear_png(self.input_dir / scene_name), self.image_size)
        targets = torch.stack([
            _resize_chw(
                read_srgb_png(self.gt_root / level_name / scene_name), self.image_size)
            for level_name, _ in LEVELS
        ])
        alphas = torch.tensor([alpha for _, alpha in LEVELS], dtype=torch.float32)
        return {"input": image, "targets": targets,
                "alphas": alphas, "scene_name": scene_name}
