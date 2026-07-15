from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schemas import DatasetClass, DatasetGateReport, DatasetGateSummary, SceneGateResult


LEVELS: tuple[str, ...] = (
    "a_m100", "a_m075", "a_m050", "a_m025", "a_000",
    "a_p025", "a_p050", "a_p075", "a_p100",
)


@dataclass(frozen=True)
class DatasetGateConfig:
    monotonic_tolerance: float = 1e-4
    min_endpoint_range_ev: float = 0.40
    max_endpoint_clip_ratio: float = 0.20
    max_endpoint_shadow_ratio: float = 0.45
    max_chroma_drift: float = 0.08
    clip_threshold: float = 0.99
    shadow_threshold: float = 0.01


def _read_rgb(path: Path, *, require_uint16: bool = False) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read PNG: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel PNG, got {image.shape}: {path}")
    if require_uint16 and image.dtype != np.uint16:
        raise TypeError(f"Input must be uint16 PNG, got {image.dtype}: {path}")
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"PNG must be uint8 or uint16, got {image.dtype}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / scale


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _mean_log_luminance(rgb: np.ndarray) -> float:
    return float(np.mean(np.log2(_luminance(rgb) + 1e-6)))


def _chromaticity(rgb: np.ndarray) -> np.ndarray:
    denominator = np.sum(rgb, axis=-1, keepdims=True) + 1e-6
    return rgb[..., :2] / denominator


class DatasetGate:
    def __init__(self, config: DatasetGateConfig) -> None:
        self.config = config

    def _discover(self, input_dir: Path, gt_root: Path) -> tuple[list[str], dict[str, dict[str, Path]]]:
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)
        names = sorted(path.name for path in input_dir.glob("*.png") if path.is_file())
        if not names:
            raise ValueError(f"No PNG files found in {input_dir}")
        level_files: dict[str, dict[str, Path]] = {}
        for level in LEVELS:
            level_dir = gt_root / level
            if not level_dir.is_dir():
                raise NotADirectoryError(level_dir)
            mapping = {path.name: path for path in level_dir.glob("*.png") if path.is_file()}
            missing = sorted(set(names) - set(mapping))
            if missing:
                raise FileNotFoundError(f"Missing {level} GT files: {missing[:10]}")
            extra = sorted(set(mapping) - set(names))
            if extra:
                raise ValueError(f"Unexpected {level} GT files: {extra[:10]}")
            level_files[level] = mapping
        return names, level_files

    def _evaluate_scene(self, scene_name: str, input_path: Path, level_files: dict[str, dict[str, Path]]) -> SceneGateResult:
        source = _read_rgb(input_path, require_uint16=True)
        images = [_read_rgb(level_files[level][scene_name]) for level in LEVELS]
        for level, image in zip(LEVELS, images, strict=True):
            if image.shape != source.shape:
                raise ValueError(f"Spatial mismatch for {scene_name}/{level}: {image.shape} != {source.shape}")

        log_luma = np.asarray([_mean_log_luminance(image) for image in images], dtype=np.float64)
        differences = np.diff(log_luma)
        violation_rate = float(np.mean(differences < -self.config.monotonic_tolerance))
        endpoint_range = float(log_luma[-1] - log_luma[0])
        bright_luma = _luminance(images[-1])
        dark_luma = _luminance(images[0])
        clip_ratio = float(np.mean(bright_luma >= self.config.clip_threshold))
        shadow_ratio = float(np.mean(dark_luma <= self.config.shadow_threshold))
        neutral_chroma = _chromaticity(images[4])
        chroma_drift = max(float(np.mean(np.abs(_chromaticity(image) - neutral_chroma))) for image in images)

        reasons: list[str] = []
        classification = DatasetClass.CLEAN
        if violation_rate > 0.0:
            classification = DatasetClass.INVALID
            reasons.append("non_monotonic")
        if endpoint_range < self.config.min_endpoint_range_ev:
            classification = DatasetClass.INVALID
            reasons.append("insufficient_endpoint_range")
        boundary_checks = (
            (clip_ratio > self.config.max_endpoint_clip_ratio, "endpoint_clipping"),
            (shadow_ratio > self.config.max_endpoint_shadow_ratio, "endpoint_shadow_crush"),
            (chroma_drift > self.config.max_chroma_drift, "chroma_drift"),
        )
        for failed, reason in boundary_checks:
            if failed:
                reasons.append(reason)
                if classification is DatasetClass.CLEAN:
                    classification = DatasetClass.BOUNDARY

        return SceneGateResult(
            scene_name=scene_name,
            classification=classification,
            reasons=tuple(reasons),
            level_luminance=tuple(float(value) for value in log_luma),
            monotonic_violation_rate=violation_rate,
            endpoint_range_ev=endpoint_range,
            endpoint_clip_ratio=clip_ratio,
            endpoint_shadow_ratio=shadow_ratio,
            max_chroma_drift=chroma_drift,
        )

    def run(self, input_dir: str | Path, gt_root: str | Path) -> DatasetGateReport:
        input_dir = Path(input_dir)
        gt_root = Path(gt_root)
        names, level_files = self._discover(input_dir, gt_root)
        scenes = tuple(
            self._evaluate_scene(name, input_dir / name, level_files)
            for name in names
        )
        counts = {classification: 0 for classification in DatasetClass}
        for scene in scenes:
            counts[scene.classification] += 1
        return DatasetGateReport(
            summary=DatasetGateSummary(total_scenes=len(scenes), counts=counts),
            scenes=scenes,
        )
