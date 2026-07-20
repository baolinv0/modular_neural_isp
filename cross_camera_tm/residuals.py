from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional

from .adapters import LUMA_WEIGHTS
from .contracts import ConfidenceSummary, FailureType, GateStatus, canonical_tensor_sha256


@dataclass(frozen=True)
class PsiFeatures:
    names: tuple[str, ...]
    values: torch.Tensor

    def as_mapping(self) -> dict[str, float]:
        if self.values.shape[0] != 1:
            raise ValueError("as_mapping is defined only for a single sample")
        return {name: float(self.values[0, index].item()) for index, name in enumerate(self.names)}


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _masked_values(value: torch.Tensor, mask: torch.Tensor, batch: int) -> torch.Tensor:
    selected = value[batch][mask[batch].bool()]
    return selected if selected.numel() else value.new_zeros(1)


def _quantiles(value: torch.Tensor, levels: Sequence[float]) -> list[torch.Tensor]:
    quantile_levels = value.new_tensor(tuple(levels))
    return list(torch.quantile(value, quantile_levels).unbind())


class PsiFeatureExtractor:
    FEATURE_NAMES = (
        "global_log_p10",
        "global_log_p25",
        "global_log_p50",
        "global_log_p75",
        "global_log_p90",
        "dynamic_log_p10",
        "dynamic_log_p25",
        "dynamic_log_p50",
        "dynamic_log_p75",
        "dynamic_log_p90",
        "luma_mad",
        "local_contrast",
        "highlight_headroom",
        "clipping_ratio",
        "shadow_coverage",
        "midtone_coverage",
        "highlight_coverage",
        "face_log_p25",
        "face_log_p50",
        "face_log_p75",
        "face_background_ratio",
        "face_area",
        "face_center_y",
        "face_center_x",
        "face_present",
        "reliable_dark_coverage",
        "highlight_valid_coverage",
        "effective_dynamic_range",
        "saturation_ratio",
        "issue_unreliable_coverage",
        "metadata_completeness",
        "exposure_confidence",
        "white_level_confidence",
        "awb_confidence",
        "hdr_confidence",
        "phase1_bound_margin",
        "canonicalization_confidence",
        "calibration_support_distance",
        "failure_code",
        "scene_code",
    )

    def extract(
        self,
        baseline: torch.Tensor,
        adapted_linear: torch.Tensor,
        reliable_mask: torch.Tensor,
        highlight_valid_mask: torch.Tensor,
        confidence: ConfidenceSummary,
        *,
        phase1_bound_margin: float,
        calibration_support_distance: float,
        failure_type: FailureType,
        scene_code: int,
        face_mask: torch.Tensor | None = None,
    ) -> PsiFeatures:
        if baseline.ndim != 4 or baseline.shape[1] != 3 or adapted_linear.shape != baseline.shape:
            raise ValueError("baseline and adapted_linear must share shape [B,3,H,W]")
        expected_mask = (baseline.shape[0], 1, baseline.shape[2], baseline.shape[3])
        if reliable_mask.shape != expected_mask or highlight_valid_mask.shape != expected_mask:
            raise ValueError("reliability masks must have shape [B,1,H,W]")
        if face_mask is not None and face_mask.shape != expected_mask:
            raise ValueError("face_mask must have shape [B,1,H,W]")
        if not torch.isfinite(baseline).all() or not torch.isfinite(adapted_linear).all():
            raise ValueError("feature inputs must be finite")

        luma = _luma(baseline).clamp(0.0, 1.0)
        linear_luma = _luma(adapted_linear).clamp(0.0, 1.0)
        dynamic_mask = reliable_mask.bool() & highlight_valid_mask.bool()
        rows: list[torch.Tensor] = []
        for batch in range(baseline.shape[0]):
            global_values = torch.log1p(luma[batch].flatten())
            dynamic_values = torch.log1p(_masked_values(luma, dynamic_mask, batch))
            global_q = _quantiles(global_values, (0.1, 0.25, 0.5, 0.75, 0.9))
            dynamic_q = _quantiles(dynamic_values, (0.1, 0.25, 0.5, 0.75, 0.9))
            median = torch.quantile(luma[batch].flatten(), 0.5)
            mad = torch.quantile((luma[batch].flatten() - median).abs(), 0.5)
            local = (luma[batch : batch + 1] - functional.avg_pool2d(
                luma[batch : batch + 1], kernel_size=3, stride=1, padding=1
            )).abs().mean()
            p10, p90 = torch.quantile(luma[batch].flatten(), luma.new_tensor((0.1, 0.9)))
            headroom = 1.0 - torch.quantile(luma[batch].flatten(), 0.99)
            clipping = (luma[batch] >= 0.995).float().mean()
            shadow = (luma[batch] < 0.12).float().mean()
            midtone = ((luma[batch] >= 0.12) & (luma[batch] <= 0.72)).float().mean()
            highlight = (luma[batch] > 0.72).float().mean()

            face_present = face_mask is not None and bool(face_mask[batch].any().item())
            if face_present:
                current_face = face_mask[batch].bool()
                face_values = torch.log1p(luma[batch][current_face])
                face_q = _quantiles(face_values, (0.25, 0.5, 0.75))
                background = luma[batch][~current_face]
                background_median = torch.quantile(background, 0.5) if background.numel() else luma.new_tensor(0.0)
                face_ratio = torch.expm1(face_q[1]) / background_median.clamp_min(1e-6)
                coords = current_face.squeeze(0).nonzero().to(luma.dtype)
                face_area = current_face.float().mean()
                center_y = coords[:, 0].mean() / max(1, baseline.shape[2] - 1)
                center_x = coords[:, 1].mean() / max(1, baseline.shape[3] - 1)
            else:
                face_q = [luma.new_tensor(0.0)] * 3
                face_ratio = luma.new_tensor(0.0)
                face_area = luma.new_tensor(0.0)
                center_y = luma.new_tensor(0.0)
                center_x = luma.new_tensor(0.0)

            saturation = ((adapted_linear[batch].amax(dim=0) >= 0.995)).float().mean()
            scalars = [
                *global_q,
                *dynamic_q,
                mad,
                local,
                headroom,
                clipping,
                shadow,
                midtone,
                highlight,
                *face_q,
                face_ratio,
                face_area,
                center_y,
                center_x,
                luma.new_tensor(float(face_present)),
                reliable_mask[batch].float().mean(),
                highlight_valid_mask[batch].float().mean(),
                p90 - p10,
                saturation,
                1.0 - dynamic_mask[batch].float().mean(),
                luma.new_tensor(confidence.completeness),
                luma.new_tensor(confidence.exposure),
                luma.new_tensor(confidence.white_level),
                luma.new_tensor(confidence.awb),
                luma.new_tensor(confidence.hdr),
                luma.new_tensor(float(phase1_bound_margin)),
                luma.new_tensor(confidence.overall),
                luma.new_tensor(float(calibration_support_distance)),
                luma.new_tensor(1.0 if failure_type is FailureType.FACE_UNDEREXPOSURE else 0.0),
                luma.new_tensor(float(scene_code)),
            ]
            rows.append(torch.stack(scalars))
        values = torch.stack(rows)
        if values.shape[1] != len(self.FEATURE_NAMES):
            raise RuntimeError("internal feature schema mismatch")
        return PsiFeatures(names=self.FEATURE_NAMES, values=values)


@dataclass(frozen=True)
class ResidualComponent:
    center: float
    lower: float
    upper: float
    sigma: float


@dataclass(frozen=True)
class ResidualEstimate:
    available: bool
    reason: str
    global_residual: ResidualComponent
    face_residual: ResidualComponent
    luma_low: float
    luma_high: float
    support_distance: float
    neighbor_count: int


@dataclass(frozen=True)
class SourceResidualProfile:
    feature_names: tuple[str, ...]
    features: torch.Tensor
    residuals: torch.Tensor
    luma_intervals: torch.Tensor
    location: torch.Tensor
    scale: torch.Tensor
    support_radius: float
    neighbors: int
    dataset_sha256: str
    model_sha256: str
    feature_schema_sha256: str
    profile_sha256: str
    synthetic: bool

    @classmethod
    def calibrate(
        cls,
        features: torch.Tensor,
        residuals: torch.Tensor,
        luma_intervals: torch.Tensor,
        *,
        feature_names: Sequence[str],
        neighbors: int = 5,
        dataset_sha256: str | None = None,
        model_sha256: str | None = None,
        synthetic: bool = True,
    ) -> "SourceResidualProfile":
        if features.ndim != 2 or features.shape[0] < 5:
            raise ValueError("source profile requires at least five feature rows")
        if residuals.shape != (features.shape[0], 2):
            raise ValueError("residuals must have shape [N,2]")
        if luma_intervals.shape != (features.shape[0], 2):
            raise ValueError("luma_intervals must have shape [N,2]")
        names = tuple(feature_names)
        if len(names) != features.shape[1] or len(set(names)) != len(names):
            raise ValueError("feature names must exactly describe unique columns")
        if not all(torch.isfinite(value).all() for value in (features, residuals, luma_intervals)):
            raise ValueError("source profile rejects non-finite values")
        location = torch.median(features, dim=0).values
        mad = torch.median((features - location).abs(), dim=0).values * 1.4826
        standard = features.std(dim=0, unbiased=False)
        scale = torch.where(mad > 1e-6, mad, torch.where(standard > 1e-6, standard, torch.ones_like(mad)))
        normalized = (features - location) / scale
        distances = torch.cdist(normalized, normalized)
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        radius = float((torch.quantile(nearest, 0.95) * 1.5 + 1e-6).item())
        if not synthetic and (dataset_sha256 is None or model_sha256 is None):
            raise ValueError("real source profiles require dataset and model SHA-256 provenance")
        derived_dataset_hash = hashlib.sha256(
            (
                canonical_tensor_sha256(features)
                + canonical_tensor_sha256(residuals)
                + canonical_tensor_sha256(luma_intervals)
            ).encode("ascii")
        ).hexdigest()
        dataset_hash = dataset_sha256 or derived_dataset_hash
        model_hash = model_sha256 or hashlib.sha256(b"synthetic-model-interface-double").hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", dataset_hash) or not re.fullmatch(
            r"[0-9a-f]{64}", model_hash
        ):
            raise ValueError("source profile provenance must use SHA-256 identifiers")
        schema_hash = hashlib.sha256(
            json.dumps(names, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        profile_payload = {
            "dataset_sha256": dataset_hash,
            "model_sha256": model_hash,
            "feature_schema_sha256": schema_hash,
            "support_radius": radius,
            "neighbors": min(max(1, neighbors), features.shape[0]),
            "synthetic": synthetic,
            "features_sha256": canonical_tensor_sha256(features),
            "residuals_sha256": canonical_tensor_sha256(residuals),
            "luma_intervals_sha256": canonical_tensor_sha256(luma_intervals),
        }
        profile_hash = hashlib.sha256(
            json.dumps(profile_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            feature_names=names,
            features=features.detach().clone(),
            residuals=residuals.detach().clone(),
            luma_intervals=luma_intervals.detach().clone(),
            location=location,
            scale=scale,
            support_radius=radius,
            neighbors=min(max(1, neighbors), features.shape[0]),
            dataset_sha256=dataset_hash,
            model_sha256=model_hash,
            feature_schema_sha256=schema_hash,
            profile_sha256=profile_hash,
            synthetic=synthetic,
        )


class SourceResidualEstimator:
    def __init__(self, profile: SourceResidualProfile):
        self.profile = profile

    @staticmethod
    def _component(values: torch.Tensor) -> ResidualComponent:
        center = torch.median(values)
        lower, upper = torch.quantile(values, values.new_tensor((0.1, 0.9)))
        sigma = torch.median((values - center).abs()) * 1.4826
        return ResidualComponent(
            center=float(center.item()),
            lower=float(min(lower.item(), center.item())),
            upper=float(max(upper.item(), center.item())),
            sigma=float(max(sigma.item(), 1e-3)),
        )

    def estimate(self, features: torch.Tensor, names: Sequence[str]) -> ResidualEstimate:
        if tuple(names) != self.profile.feature_names:
            raise ValueError("feature schema/order does not match source profile")
        if features.shape != (1, self.profile.features.shape[1]) or not torch.isfinite(features).all():
            raise ValueError("estimator expects one finite feature row")
        normalized_source = (self.profile.features - self.profile.location) / self.profile.scale
        normalized_query = (features - self.profile.location) / self.profile.scale
        distances = torch.cdist(normalized_query, normalized_source)[0]
        support_distance = float(distances.min().item())
        unavailable = ResidualEstimate(
            available=False,
            reason="outside_source_support",
            global_residual=ResidualComponent(0.0, 0.0, 0.0, 1.0),
            face_residual=ResidualComponent(0.0, 0.0, 0.0, 1.0),
            luma_low=0.0,
            luma_high=0.0,
            support_distance=support_distance,
            neighbor_count=0,
        )
        if support_distance > self.profile.support_radius:
            return unavailable
        indices = torch.topk(distances, self.profile.neighbors, largest=False).indices
        local_residuals = self.profile.residuals[indices]
        local_luma = self.profile.luma_intervals[indices]
        return ResidualEstimate(
            available=True,
            reason="ok",
            global_residual=self._component(local_residuals[:, 0]),
            face_residual=self._component(local_residuals[:, 1]),
            luma_low=float(torch.median(local_luma[:, 0]).item()),
            luma_high=float(torch.median(local_luma[:, 1]).item()),
            support_distance=support_distance,
            neighbor_count=len(indices),
        )


@dataclass(frozen=True)
class DynamicROIResult:
    status: GateStatus
    mask: torch.Tensor
    coverage: float
    reason: str


class DynamicROIBuilder:
    def __init__(self, *, min_coverage: float = 0.02, face_sigma_fraction: float = 0.08):
        self.min_coverage = float(min_coverage)
        self.face_sigma_fraction = float(face_sigma_fraction)

    @staticmethod
    def _feather(mask: torch.Tensor, sigma: float) -> torch.Tensor:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
        kernel_1d = torch.exp(-0.5 * (coordinates / max(sigma, 1e-3)).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)
        return functional.conv2d(mask, kernel, padding=radius).clamp(0.0, 1.0)

    def build(
        self,
        baseline: torch.Tensor,
        estimate: ResidualEstimate,
        reliable_mask: torch.Tensor,
        highlight_valid_mask: torch.Tensor,
        failure_type: FailureType,
        *,
        face_mask: torch.Tensor | None = None,
    ) -> DynamicROIResult:
        shape = (baseline.shape[0], 1, baseline.shape[2], baseline.shape[3])
        empty = baseline.new_zeros(shape)
        if not estimate.available:
            return DynamicROIResult(GateStatus.UNAVAILABLE, empty, 0.0, estimate.reason)
        if reliable_mask.shape != shape or highlight_valid_mask.shape != shape:
            raise ValueError("ROI masks must have shape [B,1,H,W]")
        luma = _luma(baseline)
        valid = reliable_mask.bool() & highlight_valid_mask.bool()
        in_range = (luma >= estimate.luma_low) & (luma <= estimate.luma_high)
        if failure_type is FailureType.FACE_UNDEREXPOSURE:
            if face_mask is None or face_mask.shape != shape or not face_mask.any():
                return DynamicROIResult(GateStatus.UNAVAILABLE, empty, 0.0, "face_mask_unavailable")
            face_width = float(face_mask.float().sum().sqrt().item())
            soft_face = self._feather(face_mask.to(baseline.dtype), max(0.6, face_width * self.face_sigma_fraction))
            mask = soft_face * valid.to(baseline.dtype) * in_range.to(baseline.dtype)
        else:
            mask = (valid & in_range).to(baseline.dtype)
        coverage = float((mask > 1e-4).float().mean().item())
        status = GateStatus.PASS if coverage >= self.min_coverage else GateStatus.UNAVAILABLE
        reason = "ok" if status is GateStatus.PASS else "insufficient_reliable_issue_area"
        return DynamicROIResult(status, mask, coverage, reason)


@dataclass(frozen=True)
class DirectionAlignmentResult:
    status: GateStatus
    candidate_distance: float
    baseline_distance: float
    reasons: tuple[str, ...]


class DirectionAlignmentGate:
    def __init__(
        self,
        *,
        epsilon_direction: float = 0.05,
        ratio_bounds: tuple[float, float] = (0.5, 2.0),
        non_target_max: float = 0.03,
        sigma_min: float = 0.01,
    ):
        self.epsilon_direction = float(epsilon_direction)
        self.ratio_bounds = ratio_bounds
        self.non_target_max = float(non_target_max)
        self.sigma_min = float(sigma_min)

    def evaluate(
        self,
        estimate: ResidualEstimate,
        corrections: Mapping[str, float],
        *,
        non_target_correction: float,
    ) -> DirectionAlignmentResult:
        if not estimate.available:
            return DirectionAlignmentResult(GateStatus.UNAVAILABLE, math.inf, math.inf, (estimate.reason,))
        components = {
            "global": estimate.global_residual,
            "face": estimate.face_residual,
        }
        reasons: list[str] = []
        candidate_distance = 0.0
        baseline_distance = 0.0
        for name, correction in corrections.items():
            if name not in components or not math.isfinite(float(correction)):
                reasons.append("component")
                continue
            component = components[name]
            value = float(correction)
            if not component.lower <= value <= component.upper:
                reasons.append("interval")
            if component.center != 0.0 and value * component.center <= 0.0:
                reasons.append("sign")
            denominator = max(component.sigma, self.sigma_min)
            candidate_distance += abs(value - component.center) / denominator
            baseline_distance += abs(component.center) / denominator
        if not corrections:
            reasons.append("component")
        if candidate_distance > baseline_distance * (1.0 - self.epsilon_direction):
            reasons.append("distance")
        if "global" in corrections and "face" in corrections:
            denominator = abs(float(corrections["global"]))
            ratio = math.inf if denominator < 1e-8 else abs(float(corrections["face"])) / denominator
            if not self.ratio_bounds[0] <= ratio <= self.ratio_bounds[1]:
                reasons.append("ratio")
        if abs(float(non_target_correction)) > self.non_target_max:
            reasons.append("non_target")
        unique = tuple(dict.fromkeys(reasons))
        return DirectionAlignmentResult(
            GateStatus.FAIL if unique else GateStatus.PASS,
            candidate_distance,
            baseline_distance,
            unique,
        )


@dataclass(frozen=True)
class ActivationSample:
    eligible: bool
    severity: float
    source_p75: float
    scene_group: str
    source_supported: bool
    phase1_bound_margin: float
    source_replay_regressed: bool
    failure_type: FailureType


@dataclass(frozen=True)
class Phase2ActivationResult:
    activated: bool
    eligible_count: int
    scene_group_count: int
    above_p75_fraction: float
    bootstrap_lower_bound: float
    reasons: tuple[str, ...]


def assess_phase2_activation(
    samples: Sequence[ActivationSample],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> Phase2ActivationResult:
    if bootstrap_samples < 10:
        raise ValueError("bootstrap_samples must be at least ten")
    reasons: list[str] = []
    if any(sample.source_replay_regressed for sample in samples):
        reasons.append("source_replay")
    eligible = [
        sample
        for sample in samples
        if sample.eligible
        and sample.source_supported
        and sample.phase1_bound_margin > 0.0
        and not sample.source_replay_regressed
    ]
    if len(eligible) < 50:
        reasons.append("eligible_count")
    groups = Counter(sample.scene_group for sample in eligible)
    if len(groups) < 5:
        reasons.append("scene_groups")
    if eligible and max(groups.values(), default=0) / len(eligible) > 0.4:
        reasons.append("group_dominance")
    above_fraction = (
        sum(sample.severity > sample.source_p75 for sample in eligible) / len(eligible) if eligible else 0.0
    )
    if above_fraction < 0.6:
        reasons.append("prevalence")
    by_group: dict[str, list[float]] = defaultdict(list)
    for sample in eligible:
        by_group[sample.scene_group].append(sample.severity - sample.source_p75)
    positive_groups = sum(torch.median(torch.tensor(values)).item() > 0.0 for values in by_group.values())
    if by_group and positive_groups < math.ceil(0.8 * len(by_group)):
        reasons.append("direction_stability")

    deltas = torch.tensor([sample.severity - sample.source_p75 for sample in eligible], dtype=torch.float64)
    if deltas.numel():
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randint(
            0, deltas.numel(), (bootstrap_samples, deltas.numel()), generator=generator
        )
        bootstrap_medians = torch.median(deltas[indices], dim=1).values
        lower_bound = float(torch.quantile(bootstrap_medians, 0.05).item())
    else:
        lower_bound = float("-inf")
    if lower_bound <= 0.0:
        reasons.append("bootstrap_direction")
    unique = tuple(dict.fromkeys(reasons))
    return Phase2ActivationResult(
        activated=not unique,
        eligible_count=len(eligible),
        scene_group_count=len(groups),
        above_p75_fraction=above_fraction,
        bootstrap_lower_bound=lower_bound,
        reasons=unique,
    )
