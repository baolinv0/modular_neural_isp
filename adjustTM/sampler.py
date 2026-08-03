from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterator

from torch.utils.data import Sampler


class LevelBalancedBatchSampler(Sampler[list[int]]):
    """Yield deterministic batches with exactly balanced level marginals.

    The 36 edges of K9 are partitioned into two 18-edge, 4-regular
    circulant graphs. Each 18-sample block therefore contains every level
    exactly four times. Scene assignment rotates across blocks so every
    scene-pair sample appears exactly once per epoch.
    """

    _LEVEL_COUNT = 9
    _BLOCK_SIZE = 18

    def __init__(self, dataset, batch_size: int = 18, seed: int = 42) -> None:
        if batch_size <= 0 or batch_size % self._BLOCK_SIZE != 0:
            raise ValueError(f"batch_size must be a positive multiple of {self._BLOCK_SIZE}")
        if not getattr(dataset, "scene_names", None):
            raise ValueError("dataset must expose non-empty scene_names")
        if len(getattr(dataset, "level_pairs", ())) != 36:
            raise ValueError("dataset must expose all 36 level_pairs")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.blocks_per_batch = self.batch_size // self._BLOCK_SIZE
        self._pair_groups = self._build_pair_groups(dataset.level_pairs)
        self._index_by_triplet = {
            (scene_name, low, high): index
            for index, (scene_name, low, high) in enumerate(dataset.samples)
        }
        expected = len(dataset.scene_names) * 36
        if len(self._index_by_triplet) != expected or len(dataset) != expected:
            raise ValueError("dataset samples must contain every scene-pair combination exactly once")

    @classmethod
    def _build_pair_groups(cls, level_pairs: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        groups: list[list[tuple[int, int]]] = [[], []]
        valid_pairs = set(level_pairs)
        for pair in level_pairs:
            low, high = pair
            distance = min(high - low, cls._LEVEL_COUNT - (high - low))
            group_index = 0 if distance in (1, 2) else 1
            groups[group_index].append(pair)
        if any(len(group) != cls._BLOCK_SIZE for group in groups):
            raise ValueError("failed to partition 36 level pairs into two balanced blocks")
        for group in groups:
            counts: Counter[int] = Counter()
            for low, high in group:
                if (low, high) not in valid_pairs:
                    raise ValueError("pair partition contains an unknown pair")
                counts[low] += 1
                counts[high] += 1
            if set(counts) != set(range(cls._LEVEL_COUNT)) or set(counts.values()) != {4}:
                raise ValueError("pair block is not level-balanced")
        return groups[0], groups[1]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        scenes = list(self.dataset.scene_names)
        blocks: list[list[int]] = []
        for group in self._pair_groups:
            pair_order = list(group)
            scene_order = list(scenes)
            rng.shuffle(pair_order)
            rng.shuffle(scene_order)
            scene_count = len(scene_order)
            for round_index in range(scene_count):
                block = []
                for pair_position, (low, high) in enumerate(pair_order):
                    scene_name = scene_order[(round_index + pair_position) % scene_count]
                    block.append(self._index_by_triplet[(scene_name, low, high)])
                rng.shuffle(block)
                blocks.append(block)
        rng.shuffle(blocks)
        for start in range(0, len(blocks), self.blocks_per_batch):
            batch = [index for block in blocks[start : start + self.blocks_per_batch] for index in block]
            yield batch

    def __len__(self) -> int:
        block_count = 2 * len(self.dataset.scene_names)
        return math.ceil(block_count / self.blocks_per_batch)
