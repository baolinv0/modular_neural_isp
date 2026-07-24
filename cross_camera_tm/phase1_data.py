from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .adapters import LUMA_WEIGHTS
from .contracts import AlignmentQuality, CanonicalizationResult, LinearMetadata, canonical_tensor_sha256


PHASE1_FEATURE_NAMES = (
    "log_luma_p10",
    "log_luma_p25",
    "log_luma_p50",
    "log_luma_p75",
    "log_luma_p90",
    "highlight_headroom",
    "clipping_ratio",
    "luma_mad",
    "reliable_coverage",
    "exposure_scale",
    "awb_confidence",
    "color_confidence",
    "exposure_confidence",
    "overall_confidence",
)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_mapping(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ",".join(unknown))
    if missing:
        raise ValueError(f"missing {name} fields: " + ",".join(missing))


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_device(metadata: LinearMetadata, role: str) -> None:
    if role.lower() not in metadata.device.lower():
        raise ValueError(f"{role} metadata device role is invalid")


def _require_metadata_file_binding(metadata: LinearMetadata, tensor_path: Path, name: str) -> None:
    if metadata.sample_id != tensor_path.stem:
        raise ValueError(f"{name} metadata sample_id must match tensor filename stem")


@dataclass(frozen=True)
class AlignmentEvidence:
    quality: AlignmentQuality
    overlap: float
    forward_backward_consistency: float
    valid_roi_fraction: float
    residual_displacement_px: float

    _FIELDS = {
        "quality",
        "overlap",
        "forward_backward_consistency",
        "valid_roi_fraction",
        "residual_displacement_px",
    }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AlignmentEvidence":
        if not isinstance(payload, Mapping):
            raise TypeError("alignment evidence must be a mapping")
        _strict_mapping(payload, cls._FIELDS, "alignment evidence")
        return cls(
            quality=AlignmentQuality(str(payload["quality"])),
            overlap=_unit_interval(payload["overlap"], "overlap"),
            forward_backward_consistency=_unit_interval(
                payload["forward_backward_consistency"], "forward_backward_consistency"
            ),
            valid_roi_fraction=_unit_interval(payload["valid_roi_fraction"], "valid_roi_fraction"),
            residual_displacement_px=_nonnegative(
                payload["residual_displacement_px"], "residual_displacement_px"
            ),
        )

    @property
    def enabled_losses(self) -> tuple[str, ...]:
        if self.quality is AlignmentQuality.SCENE_ONLY:
            return ("tone",)
        if self.quality is AlignmentQuality.ROI:
            return ("tone", "roi")
        return ("tone", "roi", "lowfreq")


@dataclass(frozen=True)
class Phase1SourceExample:
    sample_id: str
    scene_group: str
    samsung_image: torch.Tensor
    samsung_gt: torch.Tensor
    metadata: LinearMetadata

    def __post_init__(self) -> None:
        _nonempty(self.sample_id, "source sample_id")
        _nonempty(self.scene_group, "source scene_group")
        if self.metadata.sample_id != self.sample_id:
            raise ValueError("source metadata sample_id must match source sample_id")
        _require_device(self.metadata, "Samsung")
        images = (self.samsung_image, self.samsung_gt)
        if any(image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3 for image in images):
            raise ValueError("source images must have shape [1,3,H,W]")
        if any(not torch.isfinite(image).all() or image.min().item() < 0 for image in images):
            raise ValueError("source images must contain finite non-negative linear RGB")
        if self.samsung_image.shape != self.samsung_gt.shape:
            raise ValueError("source Samsung input and GT shapes must match")


@dataclass(frozen=True)
class Phase1CalibrationExample:
    pair_id: str
    scene_group: str
    split: str
    iphone_image: torch.Tensor
    samsung_image: torch.Tensor
    samsung_gt: torch.Tensor
    iphone_metadata: LinearMetadata
    samsung_metadata: LinearMetadata
    alignment: AlignmentEvidence
    roi_mask: torch.Tensor | None = None
    alignment_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _nonempty(self.pair_id, "calibration pair_id")
        _nonempty(self.scene_group, "calibration scene_group")
        if self.split not in {"development", "locked"}:
            raise ValueError("split must be development or locked")
        _require_device(self.iphone_metadata, "iPhone")
        _require_device(self.samsung_metadata, "Samsung")
        images = (self.iphone_image, self.samsung_image, self.samsung_gt)
        if any(image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3 for image in images):
            raise ValueError("calibration images must have shape [1,3,H,W]")
        if any(not torch.isfinite(image).all() or image.min().item() < 0 for image in images):
            raise ValueError("calibration images must contain finite non-negative linear RGB")
        if self.samsung_image.shape != self.samsung_gt.shape:
            raise ValueError("Samsung input and GT must share shape")
        if self.iphone_image.shape != self.samsung_image.shape:
            raise ValueError("Phase 1 tensors must be prewarped/resized to a shared canvas")
        expected_mask_shape = (1, 1, self.iphone_image.shape[2], self.iphone_image.shape[3])
        if self.alignment.quality in {AlignmentQuality.ROI, AlignmentQuality.LOW_FREQUENCY}:
            if self.roi_mask is None or self.roi_mask.shape != expected_mask_shape:
                raise ValueError("ROI/low-frequency supervision requires a shared ROI mask")
        if self.alignment.quality is AlignmentQuality.LOW_FREQUENCY:
            if self.alignment_mask is None or self.alignment_mask.shape != expected_mask_shape:
                raise ValueError("low-frequency supervision requires an alignment mask")


@dataclass(frozen=True)
class GroupFold:
    fold_index: int
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]


def build_group_folds(
    examples: Sequence[Phase1CalibrationExample], *, folds: int = 5
) -> tuple[GroupFold, ...]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    development = [example for example in examples if example.split == "development"]
    if not development:
        raise ValueError("development examples are required")
    counts: dict[str, int] = {}
    for example in development:
        counts[example.scene_group] = counts.get(example.scene_group, 0) + 1
    if len(counts) < folds:
        raise ValueError("scene-group cross-validation requires at least one group per fold")
    buckets: list[list[str]] = [[] for _ in range(folds)]
    bucket_sizes = [0] * folds
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        target = min(range(folds), key=lambda index: (bucket_sizes[index], index))
        buckets[target].append(group)
        bucket_sizes[target] += count
    all_groups = set(counts)
    result = []
    for index, validation in enumerate(buckets):
        validation_set = set(validation)
        result.append(
            GroupFold(
                fold_index=index,
                train_groups=tuple(sorted(all_groups - validation_set)),
                validation_groups=tuple(sorted(validation_set)),
            )
        )
    return tuple(result)


def _resolve(base: Path, value: Any, name: str) -> Path:
    path = Path(str(value))
    resolved = path if path.is_absolute() else base / path
    if not resolved.is_file():
        raise ValueError(f"{name} does not exist: {resolved}")
    return resolved


def _load_tensor(path: Path, name: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and set(payload) == {"tensor"}:
        payload = payload["tensor"]
    if not torch.is_tensor(payload):
        raise ValueError(f"{name} must contain a tensor")
    tensor = payload.detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] not in {1, 3}:
        raise ValueError(f"{name} tensor shape is invalid")
    if not torch.isfinite(tensor).all() or tensor.min().item() < 0:
        raise ValueError(f"{name} tensor must contain finite non-negative values")
    return tensor


def _load_metadata(path: Path) -> LinearMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("metadata JSON must contain an object")
    for field in (
        "is_normalized",
        "black_level_corrected",
        "white_balanced",
        "awb_gains_comparable",
        "metadata_complete",
    ):
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"metadata field {field} must be a JSON boolean")
    return LinearMetadata.from_mapping(payload)


def _verify_tensor_hash(tensor: torch.Tensor, claimed: Any, name: str) -> str:
    actual = canonical_tensor_sha256(tensor)
    if str(claimed) != actual:
        raise ValueError(f"{name} hash does not match canonical tensor bytes")
    return actual


def load_source_manifest(path: Path | str) -> tuple[Phase1SourceExample, ...]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source manifest must be a JSON object")
    _strict_mapping(payload, {"schema_version", "samples"}, "source manifest")
    if int(payload["schema_version"]) != 1 or not isinstance(payload["samples"], list):
        raise ValueError("source manifest schema is invalid")
    base = manifest_path.parent
    result = []
    sample_ids: set[str] = set()
    metadata_ids: set[str] = set()
    image_hashes: set[str] = set()
    fields = {
        "sample_id",
        "scene_group",
        "samsung_tensor",
        "samsung_gt_tensor",
        "metadata",
        "samsung_sha256",
        "gt_sha256",
    }
    for row in payload["samples"]:
        if not isinstance(row, Mapping):
            raise ValueError("source sample must be an object")
        _strict_mapping(row, fields, "source sample")
        sample_id = _nonempty(row["sample_id"], "source sample_id")
        scene_group = _nonempty(row["scene_group"], "source scene_group")
        samsung_path = _resolve(base, row["samsung_tensor"], "samsung_tensor")
        image = _load_tensor(samsung_path, "samsung_tensor")
        gt = _load_tensor(_resolve(base, row["samsung_gt_tensor"], "samsung_gt_tensor"), "samsung_gt_tensor")
        metadata = _load_metadata(_resolve(base, row["metadata"], "metadata"))
        _require_device(metadata, "Samsung")
        if metadata.sample_id != sample_id:
            raise ValueError("source metadata sample_id must match source sample_id")
        _require_metadata_file_binding(metadata, samsung_path, "source")
        if image.shape != gt.shape:
            raise ValueError("source Samsung input and GT shapes must match")
        image_hash = _verify_tensor_hash(image, row["samsung_sha256"], "samsung")
        _verify_tensor_hash(gt, row["gt_sha256"], "GT")
        if sample_id in sample_ids or metadata.sample_id in metadata_ids:
            raise ValueError("source sample identifiers must be unique")
        if image_hash in image_hashes:
            raise ValueError("source manifest contains duplicate Samsung content")
        sample_ids.add(sample_id)
        metadata_ids.add(metadata.sample_id)
        image_hashes.add(image_hash)
        result.append(
            Phase1SourceExample(
                sample_id=sample_id,
                scene_group=scene_group,
                samsung_image=image,
                samsung_gt=gt,
                metadata=metadata,
            )
        )
    if len(result) < 10:
        raise ValueError("teacher qualification requires at least ten independent Samsung source samples")
    if len({item.scene_group for item in result}) < 5:
        raise ValueError("teacher qualification requires at least five independent source scene groups")
    return tuple(result)


def load_calibration_manifest(path: Path | str) -> tuple[Phase1CalibrationExample, ...]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration manifest must be a JSON object")
    _strict_mapping(payload, {"schema_version", "pairs"}, "calibration manifest")
    if int(payload["schema_version"]) != 1 or not isinstance(payload["pairs"], list):
        raise ValueError("calibration manifest schema is invalid")
    base = manifest_path.parent
    result = []
    role_hashes: dict[str, dict[str, set[str]]] = {
        "development": {"iphone": set(), "samsung": set(), "gt": set()},
        "locked": {"iphone": set(), "samsung": set(), "gt": set()},
    }
    pair_signatures: set[tuple[str, str, str]] = set()
    iphone_metadata_ids: set[str] = set()
    samsung_metadata_ids: set[str] = set()
    fields = {
        "pair_id",
        "scene_group",
        "split",
        "iphone_tensor",
        "samsung_tensor",
        "samsung_gt_tensor",
        "iphone_metadata",
        "samsung_metadata",
        "alignment",
        "roi_mask",
        "alignment_mask",
        "iphone_sha256",
        "samsung_sha256",
        "gt_sha256",
    }
    for row in payload["pairs"]:
        if not isinstance(row, Mapping):
            raise ValueError("calibration pair must be an object")
        _strict_mapping(row, fields, "calibration pair")
        pair_id = _nonempty(row["pair_id"], "calibration pair_id")
        scene_group = _nonempty(row["scene_group"], "calibration scene_group")
        split = str(row["split"])
        if split not in {"development", "locked"}:
            raise ValueError("calibration split must be development or locked")
        iphone_path = _resolve(base, row["iphone_tensor"], "iphone_tensor")
        samsung_path = _resolve(base, row["samsung_tensor"], "samsung_tensor")
        iphone = _load_tensor(iphone_path, "iphone_tensor")
        samsung = _load_tensor(samsung_path, "samsung_tensor")
        gt = _load_tensor(_resolve(base, row["samsung_gt_tensor"], "samsung_gt_tensor"), "samsung_gt_tensor")
        iphone_metadata = _load_metadata(_resolve(base, row["iphone_metadata"], "iphone_metadata"))
        samsung_metadata = _load_metadata(_resolve(base, row["samsung_metadata"], "samsung_metadata"))
        _require_device(iphone_metadata, "iPhone")
        _require_device(samsung_metadata, "Samsung")
        _require_metadata_file_binding(iphone_metadata, iphone_path, "iPhone")
        _require_metadata_file_binding(samsung_metadata, samsung_path, "Samsung")
        alignment = AlignmentEvidence.from_mapping(row["alignment"])
        roi_mask = None if row["roi_mask"] is None else _load_tensor(_resolve(base, row["roi_mask"], "roi_mask"), "roi_mask")
        alignment_mask = (
            None
            if row["alignment_mask"] is None
            else _load_tensor(_resolve(base, row["alignment_mask"], "alignment_mask"), "alignment_mask")
        )
        if roi_mask is not None:
            roi_mask = roi_mask[:, :1].bool()
        if alignment_mask is not None:
            alignment_mask = alignment_mask[:, :1].bool()
        iphone_hash = _verify_tensor_hash(iphone, row["iphone_sha256"], "iPhone")
        samsung_hash = _verify_tensor_hash(samsung, row["samsung_sha256"], "Samsung")
        gt_hash = _verify_tensor_hash(gt, row["gt_sha256"], "GT")
        signature = (iphone_hash, samsung_hash, gt_hash)
        if signature in pair_signatures:
            raise ValueError("calibration manifest contains a duplicate pair content signature")
        if iphone_metadata.sample_id in iphone_metadata_ids or samsung_metadata.sample_id in samsung_metadata_ids:
            raise ValueError("calibration metadata sample identifiers must be unique")
        pair_signatures.add(signature)
        iphone_metadata_ids.add(iphone_metadata.sample_id)
        samsung_metadata_ids.add(samsung_metadata.sample_id)
        role_hashes[split]["iphone"].add(iphone_hash)
        role_hashes[split]["samsung"].add(samsung_hash)
        role_hashes[split]["gt"].add(gt_hash)
        result.append(
            Phase1CalibrationExample(
                pair_id=pair_id,
                scene_group=scene_group,
                split=split,
                iphone_image=iphone,
                samsung_image=samsung,
                samsung_gt=gt,
                iphone_metadata=iphone_metadata,
                samsung_metadata=samsung_metadata,
                alignment=alignment,
                roi_mask=roi_mask,
                alignment_mask=alignment_mask,
            )
        )
    if len(result) != 50:
        raise ValueError("Phase 1 requires exactly 50 calibration pairs")
    if sum(item.split == "development" for item in result) != 40:
        raise ValueError("Phase 1 requires exactly 40 development pairs")
    if sum(item.split == "locked" for item in result) != 10:
        raise ValueError("Phase 1 requires exactly 10 locked pairs")
    if len({item.pair_id for item in result}) != 50:
        raise ValueError("calibration pair identifiers must be unique")
    development_groups = {item.scene_group for item in result if item.split == "development"}
    locked_groups = {item.scene_group for item in result if item.split == "locked"}
    if development_groups & locked_groups:
        raise ValueError("locked calibration scene groups must be unseen during development")
    for role in ("iphone", "samsung", "gt"):
        if role_hashes["development"][role] & role_hashes["locked"][role]:
            raise ValueError(f"development and locked calibration {role} content must be disjoint")
    build_group_folds(result, folds=5)
    return tuple(result)


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def extract_phase1_features(
    canonical: CanonicalizationResult, metadata: LinearMetadata
) -> torch.Tensor:
    luma = _luma(canonical.image).clamp(0.0, 1.0)
    log_luma = torch.log(luma.clamp_min(1e-6)).flatten()
    quantiles = torch.quantile(log_luma, luma.new_tensor((0.1, 0.25, 0.5, 0.75, 0.9)))
    p99 = torch.quantile(luma.flatten(), 0.99)
    median = torch.median(luma)
    mad = torch.median((luma - median).abs())
    values = torch.cat(
        (
            quantiles,
            luma.new_tensor(
                [
                    float((1.0 - p99).clamp_min(0.0).item()),
                    float((luma >= 0.995).float().mean().item()),
                    float(mad.item()),
                    canonical.reliable_coverage,
                    canonical.exposure_scale,
                    canonical.confidence.awb,
                    canonical.confidence.color,
                    canonical.confidence.exposure,
                    canonical.confidence.overall,
                ]
            ),
        )
    )
    if values.numel() != len(PHASE1_FEATURE_NAMES):
        raise RuntimeError("Phase 1 feature schema mismatch")
    return values.unsqueeze(0)


def teacher_error_metrics(output: torch.Tensor, gt: torch.Tensor) -> tuple[dict[str, float], bool]:
    if output.shape != gt.shape or output.ndim != 4 or output.shape[1] != 3:
        raise ValueError("teacher output and GT must share shape [B,3,H,W]")
    if not torch.isfinite(output).all() or not torch.isfinite(gt).all():
        return {
            "log_luma_mae": float("inf"),
            "highlight_error": float("inf"),
            "clipping_delta": float("inf"),
            "contrast_error": float("inf"),
        }, True
    output_luma = _luma(output).clamp(0.0, 1.0)
    gt_luma = _luma(gt).clamp(0.0, 1.0)
    output_log = torch.log(output_luma.clamp_min(1e-6))
    gt_log = torch.log(gt_luma.clamp_min(1e-6))
    output_headroom = 1.0 - torch.quantile(output_luma.flatten(), 0.99)
    gt_headroom = 1.0 - torch.quantile(gt_luma.flatten(), 0.99)
    output_clip = (output_luma >= 0.995).float().mean()
    gt_clip = (gt_luma >= 0.995).float().mean()
    output_contrast = torch.quantile(output_luma.flatten(), 0.75) - torch.quantile(output_luma.flatten(), 0.25)
    gt_contrast = torch.quantile(gt_luma.flatten(), 0.75) - torch.quantile(gt_luma.flatten(), 0.25)
    metrics = {
        "log_luma_mae": float((output_log - gt_log).abs().mean().item()),
        "highlight_error": float((output_headroom - gt_headroom).abs().item()),
        "clipping_delta": float((output_clip - gt_clip).abs().item()),
        "contrast_error": float((output_contrast - gt_contrast).abs().item()),
    }
    hard_defect = bool(output_clip.item() > gt_clip.item() + 0.05)
    return metrics, hard_defect


def manifest_sha256(path: Path | str) -> str:
    return file_sha256(path)
