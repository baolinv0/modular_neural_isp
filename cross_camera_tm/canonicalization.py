from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import CanonicalizationResult, ConfidenceSummary, LinearMetadata, canonical_tensor_sha256


@dataclass(frozen=True)
class CanonicalizationConfig:
    exposure_scale_min: float = 0.5
    exposure_scale_max: float = 2.0
    reliable_dark_threshold: float = 0.01
    highlight_threshold: float = 0.98

    def __post_init__(self) -> None:
        values = (
            self.exposure_scale_min,
            self.exposure_scale_max,
            self.reliable_dark_threshold,
            self.highlight_threshold,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("canonicalization thresholds must be finite")
        if not 0 < self.exposure_scale_min <= self.exposure_scale_max:
            raise ValueError("exposure scale bounds are invalid")
        if not 0 <= self.reliable_dark_threshold < self.highlight_threshold <= 1:
            raise ValueError("reliable/highlight thresholds are invalid")


class DeviceCanonicalizer:
    def __init__(self, config: CanonicalizationConfig | None = None):
        self.config = config or CanonicalizationConfig()

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if not torch.is_tensor(image):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("linear RGB image must have shape [B,3,H,W]")
        if image.shape[0] < 1 or image.shape[2] < 1 or image.shape[3] < 1:
            raise ValueError("linear RGB image dimensions must be non-empty")
        if not torch.isfinite(image).all():
            raise ValueError("linear RGB image must contain finite values")

    def canonicalize(self, image: torch.Tensor, metadata: LinearMetadata) -> CanonicalizationResult:
        self._validate_image(image)
        if not isinstance(metadata, LinearMetadata):
            raise TypeError("metadata must be LinearMetadata")
        input_sha = canonical_tensor_sha256(image)
        output = image.to(dtype=torch.float32)
        operations: list[str] = []

        if metadata.is_normalized:
            white_confidence = 1.0
        else:
            output = output / metadata.white_level
            operations.append("white_level")
            white_confidence = 1.0

        if (
            metadata.awb_gains_comparable
            and metadata.awb_gains_applied is not None
            and metadata.reference_awb_gains is not None
        ):
            applied = output.new_tensor(metadata.awb_gains_applied).view(1, 3, 1, 1)
            reference = output.new_tensor(metadata.reference_awb_gains).view(1, 3, 1, 1)
            output = output * (reference / applied)
            operations.append("awb_gain_alignment")
            awb_confidence = 1.0
        else:
            awb_confidence = 0.0

        if metadata.ccm_to_common is not None:
            matrix = output.new_tensor(metadata.ccm_to_common).unsqueeze(0).expand(output.shape[0], 3, 3)
            flat = output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, 3)
            output = torch.bmm(flat, matrix.transpose(1, 2)).reshape(
                output.shape[0], output.shape[2], output.shape[3], 3
            ).permute(0, 3, 1, 2)
            operations.append("common_ccm")
            color_confidence = 1.0
        else:
            color_confidence = 0.0

        exposure_values = (
            metadata.exposure_time_s,
            metadata.iso,
            metadata.aperture,
            metadata.reference_exposure_product,
        )
        if all(value is not None for value in exposure_values):
            current = metadata.exposure_time_s * metadata.iso / (metadata.aperture**2)  # type: ignore[operator]
            raw_scale = metadata.reference_exposure_product / current  # type: ignore[operator]
            exposure_scale = float(
                min(self.config.exposure_scale_max, max(self.config.exposure_scale_min, raw_scale))
            )
            output = output * exposure_scale
            operations.append("exposure_prior")
            exposure_confidence = metadata.hdr_confidence
        else:
            exposure_scale = 1.0
            exposure_confidence = 0.0

        output = output.clamp(0.0, 1.0)
        luma = 0.2989 * output[:, 0:1] + 0.5870 * output[:, 1:2] + 0.1140 * output[:, 2:3]
        highlight_valid = luma < self.config.highlight_threshold
        reliable = (luma > self.config.reliable_dark_threshold) & highlight_valid
        completeness = 1.0 if metadata.metadata_complete else 0.5
        components = (
            white_confidence,
            awb_confidence,
            color_confidence,
            exposure_confidence,
            metadata.hdr_confidence,
            completeness,
        )
        overall = float(sum(components) / len(components))
        confidence = ConfidenceSummary(
            white_level=white_confidence,
            awb=awb_confidence,
            color=color_confidence,
            exposure=exposure_confidence,
            hdr=metadata.hdr_confidence,
            completeness=completeness,
            overall=overall,
        )
        return CanonicalizationResult(
            sample_id=metadata.sample_id,
            image=output,
            reliable_mask=reliable,
            highlight_valid_mask=highlight_valid,
            confidence=confidence,
            exposure_scale=exposure_scale,
            reliable_coverage=float(reliable.float().mean().item()),
            operations=tuple(operations),
            input_sha256=input_sha,
            output_sha256=canonical_tensor_sha256(output),
        )
