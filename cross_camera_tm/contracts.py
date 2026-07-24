from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import torch


class FailureType(str, Enum):
    GLOBAL_UNDEREXPOSURE = "global_underexposure"
    FACE_UNDEREXPOSURE = "face_underexposure"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


class AlignmentQuality(str, Enum):
    SCENE_ONLY = "scene_only"
    ROI = "roi"
    LOW_FREQUENCY = "low_frequency"


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} values must be positive")
    return result


def _optional_positive(value: Any, name: str) -> float | None:
    return None if value is None else _finite_float(value, name, positive=True)


def _vector3(value: Any, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(_finite_float(item, name, positive=True) for item in value)
    return result  # type: ignore[return-value]


def _matrix3(value: Any, name: str) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise ValueError(f"{name} must be a 3x3 matrix")
        rows.append(tuple(_finite_float(item, name) for item in row))
    return tuple(rows)


@dataclass(frozen=True)
class LinearMetadata:
    sample_id: str
    device: str
    white_level: float
    is_normalized: bool
    black_level_corrected: bool
    white_balanced: bool
    awb_gains_applied: tuple[float, float, float] | None
    reference_awb_gains: tuple[float, float, float] | None
    awb_gains_comparable: bool
    ccm_to_common: tuple[tuple[float, float, float], ...] | None
    exposure_time_s: float | None
    iso: float | None
    aperture: float | None
    reference_exposure_product: float | None
    hdr_confidence: float
    metadata_complete: bool

    _FIELDS = frozenset(
        {
            "sample_id",
            "device",
            "white_level",
            "is_normalized",
            "black_level_corrected",
            "white_balanced",
            "awb_gains_applied",
            "reference_awb_gains",
            "awb_gains_comparable",
            "ccm_to_common",
            "exposure_time_s",
            "iso",
            "aperture",
            "reference_exposure_product",
            "hdr_confidence",
            "metadata_complete",
        }
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LinearMetadata":
        if not isinstance(payload, Mapping):
            raise TypeError("linear metadata must be a mapping")
        unknown = sorted(set(payload) - cls._FIELDS)
        if unknown:
            raise ValueError("unknown fields: " + ", ".join(unknown))
        missing = sorted(cls._FIELDS - set(payload))
        if missing:
            raise ValueError("missing fields: " + ", ".join(missing))
        sample_id = str(payload["sample_id"]).strip()
        device = str(payload["device"]).strip()
        if not sample_id or not device:
            raise ValueError("sample_id and device must be non-empty")
        black_level_corrected = bool(payload["black_level_corrected"])
        white_balanced = bool(payload["white_balanced"])
        if not black_level_corrected:
            raise ValueError("input must already be black-level-corrected")
        if not white_balanced:
            raise ValueError("input must already be white-balanced")
        hdr_confidence = _finite_float(payload["hdr_confidence"], "hdr_confidence")
        if not 0.0 <= hdr_confidence <= 1.0:
            raise ValueError("hdr_confidence must lie in [0, 1]")
        return cls(
            sample_id=sample_id,
            device=device,
            white_level=_finite_float(payload["white_level"], "white_level", positive=True),
            is_normalized=bool(payload["is_normalized"]),
            black_level_corrected=black_level_corrected,
            white_balanced=white_balanced,
            awb_gains_applied=_vector3(payload["awb_gains_applied"], "awb_gains_applied"),
            reference_awb_gains=_vector3(payload["reference_awb_gains"], "reference_awb_gains"),
            awb_gains_comparable=bool(payload["awb_gains_comparable"]),
            ccm_to_common=_matrix3(payload["ccm_to_common"], "ccm_to_common"),
            exposure_time_s=_optional_positive(payload["exposure_time_s"], "exposure_time_s"),
            iso=_optional_positive(payload["iso"], "iso"),
            aperture=_optional_positive(payload["aperture"], "aperture"),
            reference_exposure_product=_optional_positive(
                payload["reference_exposure_product"], "reference_exposure_product"
            ),
            hdr_confidence=hdr_confidence,
            metadata_complete=bool(payload["metadata_complete"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceSummary:
    white_level: float
    awb: float
    color: float
    exposure: float
    hdr: float
    completeness: float
    overall: float


@dataclass(frozen=True)
class CanonicalizationResult:
    sample_id: str
    image: torch.Tensor
    reliable_mask: torch.Tensor
    highlight_valid_mask: torch.Tensor
    confidence: ConfidenceSummary
    exposure_scale: float
    reliable_coverage: float
    operations: tuple[str, ...]
    input_sha256: str
    output_sha256: str


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    if not torch.is_tensor(tensor):
        raise TypeError("canonical tensor hashing requires a torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("canonical tensor hashing rejects non-finite values")
    canonical = tensor.detach().cpu().contiguous()
    header = f"shape={tuple(canonical.shape)};dtype={canonical.dtype};".encode("ascii")
    return hashlib.sha256(header + canonical.numpy().tobytes(order="C")).hexdigest()


def _strict_payload(payload: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    unknown = sorted(set(payload) - fields)
    missing = sorted(fields - set(payload))
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ",".join(unknown))
    if missing:
        raise ValueError(f"missing {name} fields: " + ",".join(missing))


def _validate_linear_image(image: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"{name} must be a single finite [1,3,H,W] tensor")
    if not torch.isfinite(image).all() or image.min().item() < 0:
        raise ValueError(f"{name} must contain finite non-negative linear RGB")
    return image


def _require_hash(image: torch.Tensor, claimed: str, name: str) -> str:
    actual = canonical_tensor_sha256(image)
    if claimed != actual:
        raise ValueError(f"{name} does not match canonical tensor bytes")
    return claimed


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    scene_group: str
    linear_rgb: torch.Tensor
    samsung_gt: torch.Tensor
    metadata: LinearMetadata
    linear_sha256: str
    gt_sha256: str

    _FIELDS = frozenset(
        {"sample_id", "scene_group", "linear_rgb", "samsung_gt", "metadata", "linear_sha256", "gt_sha256"}
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceSample":
        _strict_payload(payload, cls._FIELDS, "source sample")
        linear = _validate_linear_image(payload["linear_rgb"], "linear_rgb")
        gt = _validate_linear_image(payload["samsung_gt"], "samsung_gt")
        if gt.shape != linear.shape:
            raise ValueError("source linear and GT shapes must match")
        metadata = payload["metadata"]
        if not isinstance(metadata, LinearMetadata) or metadata.device.lower().find("samsung") < 0:
            raise ValueError("source sample requires Samsung LinearMetadata")
        sample_id, scene_group = str(payload["sample_id"]).strip(), str(payload["scene_group"]).strip()
        if not sample_id or not scene_group or metadata.sample_id != sample_id:
            raise ValueError("source identifiers must be non-empty and metadata-bound")
        return cls(
            sample_id,
            scene_group,
            linear,
            gt,
            metadata,
            _require_hash(linear, str(payload["linear_sha256"]), "linear_sha256"),
            _require_hash(gt, str(payload["gt_sha256"]), "gt_sha256"),
        )


@dataclass(frozen=True)
class CalibrationPair:
    pair_id: str
    scene_group: str
    split: str
    alignment_quality: AlignmentQuality
    iphone_linear: torch.Tensor
    samsung_linear: torch.Tensor
    samsung_gt: torch.Tensor
    iphone_metadata: LinearMetadata
    samsung_metadata: LinearMetadata
    iphone_sha256: str
    samsung_sha256: str
    gt_sha256: str

    _FIELDS = frozenset(
        {
            "pair_id",
            "scene_group",
            "split",
            "alignment_quality",
            "iphone_linear",
            "samsung_linear",
            "samsung_gt",
            "iphone_metadata",
            "samsung_metadata",
            "iphone_sha256",
            "samsung_sha256",
            "gt_sha256",
        }
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CalibrationPair":
        _strict_payload(payload, cls._FIELDS, "calibration pair")
        images = tuple(
            _validate_linear_image(payload[name], name)
            for name in ("iphone_linear", "samsung_linear", "samsung_gt")
        )
        if images[1].shape != images[2].shape:
            raise ValueError("Samsung calibration input and GT shapes must match")
        split = str(payload["split"])
        if split not in {"development", "locked"}:
            raise ValueError("calibration split must be development or locked")
        iphone_metadata, samsung_metadata = payload["iphone_metadata"], payload["samsung_metadata"]
        if not isinstance(iphone_metadata, LinearMetadata) or not isinstance(samsung_metadata, LinearMetadata):
            raise ValueError("calibration metadata must be parsed LinearMetadata")
        if "iphone" not in iphone_metadata.device.lower() or "samsung" not in samsung_metadata.device.lower():
            raise ValueError("calibration pair device roles are invalid")
        pair_id, group = str(payload["pair_id"]).strip(), str(payload["scene_group"]).strip()
        if not pair_id or not group:
            raise ValueError("calibration identifiers must be non-empty")
        return cls(
            pair_id=pair_id,
            scene_group=group,
            split=split,
            alignment_quality=AlignmentQuality(payload["alignment_quality"]),
            iphone_linear=images[0],
            samsung_linear=images[1],
            samsung_gt=images[2],
            iphone_metadata=iphone_metadata,
            samsung_metadata=samsung_metadata,
            iphone_sha256=_require_hash(images[0], str(payload["iphone_sha256"]), "iphone_sha256"),
            samsung_sha256=_require_hash(images[1], str(payload["samsung_sha256"]), "samsung_sha256"),
            gt_sha256=_require_hash(images[2], str(payload["gt_sha256"]), "gt_sha256"),
        )


@dataclass(frozen=True)
class CalibrationDataset:
    pairs: tuple[CalibrationPair, ...]

    def __post_init__(self) -> None:
        if len(self.pairs) != 50:
            raise ValueError("the frozen calibration contract requires exactly 50 pairs")
        if sum(pair.split == "development" for pair in self.pairs) != 40:
            raise ValueError("calibration requires exactly 40 development pairs")
        if sum(pair.split == "locked" for pair in self.pairs) != 10:
            raise ValueError("calibration requires exactly 10 locked pairs")
        if len({pair.pair_id for pair in self.pairs}) != 50:
            raise ValueError("calibration pair ids must be unique")
        if len({pair.scene_group for pair in self.pairs}) < 5:
            raise ValueError("calibration requires at least five scene groups")


@dataclass(frozen=True)
class TargetSample:
    sample_id: str
    scene_group: str
    split: str
    iphone_linear: torch.Tensor
    metadata: LinearMetadata
    input_sha256: str

    _FIELDS = frozenset(
        {"sample_id", "scene_group", "split", "iphone_linear", "metadata", "input_sha256"}
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TargetSample":
        _strict_payload(payload, cls._FIELDS, "target sample")
        image = _validate_linear_image(payload["iphone_linear"], "iphone_linear")
        metadata = payload["metadata"]
        if not isinstance(metadata, LinearMetadata) or "iphone" not in metadata.device.lower():
            raise ValueError("target sample requires iPhone LinearMetadata")
        split = str(payload["split"])
        if split not in {"train", "validation", "locked_holdout"}:
            raise ValueError("invalid target split")
        sample_id, group = str(payload["sample_id"]).strip(), str(payload["scene_group"]).strip()
        if not sample_id or not group or metadata.sample_id != sample_id:
            raise ValueError("target identifiers must be non-empty and metadata-bound")
        return cls(
            sample_id,
            group,
            split,
            image,
            metadata,
            _require_hash(image, str(payload["input_sha256"]), "input_sha256"),
        )
