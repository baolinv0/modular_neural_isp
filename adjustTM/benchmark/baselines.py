from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from adjustTM.constants import LEVELS
from adjustTM.transfer import linear_luminance, srgb_to_linear
from .transforms import exposure_transform, luminance_gamma_transform


@dataclass(frozen=True)
class SearchResult:
    parameter: float
    objective: float


def log_luma_objective(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    pred_y = linear_luminance(srgb_to_linear(pred)).clamp_min(eps).log()
    target_y = linear_luminance(srgb_to_linear(target)).clamp_min(eps).log()
    return (pred_y - target_y).abs().mean()


def _transform(kind: str) -> Callable[[torch.Tensor, float], torch.Tensor]:
    if kind == "exposure":
        return exposure_transform
    if kind == "gamma":
        return luminance_gamma_transform
    raise ValueError(f"Unknown simple baseline kind: {kind}")


def search_best_parameter(
    baseline: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str,
    minimum: float,
    maximum: float,
    steps: int = 161,
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = log_luma_objective,
    refine_rounds: int = 3,
) -> SearchResult:
    if baseline.shape != target.shape:
        raise ValueError(f"Shape mismatch: baseline={baseline.shape}, target={target.shape}")
    if steps < 3 or minimum >= maximum:
        raise ValueError("Search requires steps >= 3 and minimum < maximum")
    transform = _transform(kind)
    left, right = float(minimum), float(maximum)
    best_parameter = left
    best_objective = float("inf")
    for _ in range(max(refine_rounds, 1)):
        candidates = np.linspace(left, right, steps)
        scores: list[float] = []
        with torch.no_grad():
            for candidate in candidates:
                score = float(objective(transform(baseline, float(candidate)), target))
                scores.append(score)
        best_index = min(range(len(scores)), key=lambda index: (scores[index], abs(float(candidates[index]) - (0.0 if kind == "exposure" else 1.0))))
        best_parameter = float(candidates[best_index])
        best_objective = float(scores[best_index])
        spacing = float(candidates[1] - candidates[0])
        left = max(minimum, best_parameter - spacing)
        right = min(maximum, best_parameter + spacing)
    return SearchResult(best_parameter, best_objective)


def _pava(values: Sequence[float], increasing: bool) -> list[float]:
    data = [float(value) for value in values]
    if not increasing:
        data = [-value for value in data]
    blocks: list[tuple[float, int]] = []
    for value in data:
        blocks.append((value, 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_mean, right_count = blocks.pop()
            left_mean, left_count = blocks.pop()
            count = left_count + right_count
            blocks.append(((left_mean * left_count + right_mean * right_count) / count, count))
    output: list[float] = []
    for mean, count in blocks:
        output.extend([mean] * count)
    if not increasing:
        output = [-value for value in output]
    return output


def project_level_parameters(parameters: Mapping[str, float], *, kind: str) -> dict[str, float]:
    level_names = [name for name, _ in LEVELS]
    missing = sorted(set(level_names) - set(parameters))
    if missing:
        raise KeyError(f"Missing level parameters: {missing}")
    anchor = 0.0 if kind == "exposure" else 1.0
    values = [float(parameters[name]) for name in level_names]
    values[4] = anchor
    if kind == "exposure":
        left = _pava(values[:5], increasing=True)
        right = _pava(values[4:], increasing=True)
        left = [min(v, anchor) for v in left]
        right = [max(v, anchor) for v in right]
    elif kind == "gamma":
        left = _pava(values[:5], increasing=False)
        right = _pava(values[4:], increasing=False)
        left = [max(v, anchor) for v in left]
        right = [min(v, anchor) for v in right]
    else:
        raise ValueError(f"Unknown kind: {kind}")
    combined = left[:-1] + [anchor] + right[1:]
    return dict(zip(level_names, combined))


def calibrate_global_parameters(
    baseline_by_scene: Mapping[str, torch.Tensor],
    target_by_scene_level: Mapping[tuple[str, str], torch.Tensor],
    *,
    kind: str,
    minimum: float,
    maximum: float,
    steps: int,
) -> dict[str, float]:
    scene_ids = sorted(baseline_by_scene)
    if not scene_ids:
        raise ValueError("No calibration scenes")
    transform = _transform(kind)
    parameters: dict[str, float] = {}
    for level_name, _ in LEVELS:
        if level_name == "a_000":
            parameters[level_name] = 0.0 if kind == "exposure" else 1.0
            continue
        targets = [target_by_scene_level[(scene_id, level_name)] for scene_id in scene_ids]
        baselines = [baseline_by_scene[scene_id] for scene_id in scene_ids]
        candidates = np.linspace(minimum, maximum, steps)
        scores = []
        with torch.no_grad():
            for candidate in candidates:
                per_scene = [float(log_luma_objective(transform(base, float(candidate)), target)) for base, target in zip(baselines, targets)]
                scores.append(float(np.mean(per_scene)))
        best_index = int(np.argmin(np.asarray(scores)))
        parameters[level_name] = float(candidates[best_index])
    return project_level_parameters(parameters, kind=kind)
