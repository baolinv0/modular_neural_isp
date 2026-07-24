from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import phase1_training as core
from .adapters import TargetCameraAdapter
from .canonicalization import CanonicalizationConfig, DeviceCanonicalizer
from .contracts import AlignmentQuality
from .phase1 import FrozenSamsungTM, TeacherQualifier
from .phase1_data import (
    PHASE1_FEATURE_NAMES,
    AlignmentEvidence,
    Phase1CalibrationExample,
    extract_phase1_features,
    load_calibration_manifest,
)


@dataclass(frozen=True)
class AlignmentPolicy:
    """Frozen numeric evidence thresholds used to downgrade alignment claims."""

    roi_overlap_min: float = 0.50
    roi_valid_fraction_min: float = 0.30
    lowfreq_overlap_min: float = 0.70
    lowfreq_valid_fraction_min: float = 0.50
    lowfreq_forward_backward_min: float = 0.80
    lowfreq_displacement_max_px: float = 2.0

    def __post_init__(self) -> None:
        unit_values = (
            self.roi_overlap_min,
            self.roi_valid_fraction_min,
            self.lowfreq_overlap_min,
            self.lowfreq_valid_fraction_min,
            self.lowfreq_forward_backward_min,
        )
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in unit_values):
            raise ValueError("alignment policy fractions must be finite and lie in [0,1]")
        displacement = float(self.lowfreq_displacement_max_px)
        if not math.isfinite(displacement) or displacement < 0:
            raise ValueError("alignment displacement threshold must be finite and non-negative")
        if self.lowfreq_overlap_min < self.roi_overlap_min:
            raise ValueError("low-frequency overlap threshold cannot be weaker than ROI")
        if self.lowfreq_valid_fraction_min < self.roi_valid_fraction_min:
            raise ValueError("low-frequency ROI threshold cannot be weaker than ROI")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def effective_quality(self, evidence: AlignmentEvidence) -> AlignmentQuality:
        roi_supported = (
            evidence.overlap >= self.roi_overlap_min
            and evidence.valid_roi_fraction >= self.roi_valid_fraction_min
        )
        lowfreq_supported = (
            evidence.overlap >= self.lowfreq_overlap_min
            and evidence.valid_roi_fraction >= self.lowfreq_valid_fraction_min
            and evidence.forward_backward_consistency >= self.lowfreq_forward_backward_min
            and evidence.residual_displacement_px <= self.lowfreq_displacement_max_px
        )
        if evidence.quality is AlignmentQuality.LOW_FREQUENCY and lowfreq_supported:
            return AlignmentQuality.LOW_FREQUENCY
        if evidence.quality in {AlignmentQuality.ROI, AlignmentQuality.LOW_FREQUENCY} and roi_supported:
            return AlignmentQuality.ROI
        return AlignmentQuality.SCENE_ONLY


DEFAULT_ALIGNMENT_POLICY = AlignmentPolicy()


def load_calibration_manifest_strict(
    path: Path | str,
    *,
    policy: AlignmentPolicy = DEFAULT_ALIGNMENT_POLICY,
) -> tuple[Phase1CalibrationExample, ...]:
    """Load the manifest and derive the maximum legal loss level from evidence.

    The external quality label is treated only as an upper bound. Numeric
    evidence can preserve it or downgrade it, but can never upgrade it.
    """

    examples = load_calibration_manifest(path)
    result: list[Phase1CalibrationExample] = []
    for example in examples:
        effective = policy.effective_quality(example.alignment)
        alignment = replace(example.alignment, quality=effective)
        result.append(replace(example, alignment=alignment))
    return tuple(result)


@dataclass
class HardenedPhase1Artifact:
    base: core.Phase1Artifact
    canonicalization_config: CanonicalizationConfig
    canonicalization_sha256: str
    alignment_policy: AlignmentPolicy
    alignment_policy_sha256: str
    max_support_distance: float
    minimum_parameter_bound_margin: float
    real_phase1_calibration_accepted: bool
    real_source_replay_verified: bool
    real_target_effectiveness_verified: bool

    @property
    def adapter(self):
        return self.base.adapter

    @property
    def feature_mean(self):
        return self.base.feature_mean

    @property
    def feature_std(self):
        return self.base.feature_std

    @property
    def support_min(self):
        return self.base.support_min

    @property
    def support_max(self):
        return self.base.support_max

    @property
    def samsung_model_sha256(self) -> str:
        return self.base.samsung_model_sha256

    @property
    def source_manifest_sha256(self) -> str:
        return self.base.source_manifest_sha256

    @property
    def calibration_manifest_sha256(self) -> str:
        return self.base.calibration_manifest_sha256

    @property
    def phase1_passed(self) -> bool:
        return self.base.phase1_passed

    @property
    def validation_report(self):
        return self.base.validation_report

    @property
    def teacher_profile(self):
        return self.base.teacher_profile

    @property
    def data_mode(self) -> str:
        return self.base.data_mode


def _support_distance_to_bounds(
    normalized: torch.Tensor,
    support_min: torch.Tensor,
    support_max: torch.Tensor,
) -> torch.Tensor:
    below = (support_min - normalized).clamp_min(0.0)
    above = (normalized - support_max).clamp_min(0.0)
    return torch.sqrt((below.square() + above.square()).mean(dim=1))


def _calibrated_support_threshold(
    examples: Sequence[Phase1CalibrationExample],
    canonicalizer: DeviceCanonicalizer,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> float:
    development = [item for item in examples if item.split == "development"]
    if len(development) != 40:
        raise ValueError("support calibration requires the frozen 40-pair development set")
    features = []
    groups = []
    for item in development:
        canonical = canonicalizer.canonicalize(item.iphone_image, item.iphone_metadata)
        features.append(extract_phase1_features(canonical, item.iphone_metadata))
        groups.append(item.scene_group)
    normalized = (torch.cat(features, dim=0) - feature_mean) / feature_std
    distances: list[torch.Tensor] = []
    for group in sorted(set(groups)):
        train_indices = [index for index, value in enumerate(groups) if value != group]
        validation_indices = [index for index, value in enumerate(groups) if value == group]
        if not train_indices or not validation_indices:
            continue
        train = normalized[train_indices]
        validation = normalized[validation_indices]
        distances.append(
            _support_distance_to_bounds(
                validation,
                train.min(dim=0).values,
                train.max(dim=0).values,
            )
        )
    if not distances:
        raise ValueError("support calibration requires multiple development scene groups")
    values = torch.cat(distances)
    threshold = float(max(0.05, torch.quantile(values, 0.95).item() + 0.05))
    if not math.isfinite(threshold):
        raise ValueError("support calibration produced a non-finite threshold")
    return threshold


def _calibrated_margin_threshold(
    examples: Sequence[Phase1CalibrationExample],
    canonicalizer: DeviceCanonicalizer,
    frozen_tm: FrozenSamsungTM,
    artifact: core.Phase1Artifact,
) -> float:
    development = [item for item in examples if item.split == "development"]
    if len(development) != 40:
        raise ValueError("parameter-margin calibration requires the frozen 40-pair development set")
    qualifier = TeacherQualifier(artifact.teacher_profile)
    prepared = core._prepare_pairs(development, canonicalizer, frozen_tm, qualifier)
    eligible = [item for item in prepared if item.teacher_weight > 0.0]
    _, margins = core._evaluate_items(
        eligible,
        artifact.adapter,
        artifact.feature_mean,
        artifact.feature_std,
        frozen_tm,
    )
    positive = torch.tensor(
        [value for value in margins if math.isfinite(float(value)) and value > 0.0],
        dtype=torch.float64,
    )
    if positive.numel() < 8:
        raise ValueError("parameter-margin calibration has insufficient qualified samples")
    threshold = float(max(0.0, torch.quantile(positive, 0.05).item() - 0.05))
    if not math.isfinite(threshold):
        raise ValueError("parameter-margin calibration produced a non-finite threshold")
    return threshold


def seal_phase1_artifact(
    path: Path | str,
    *,
    calibration_examples: Sequence[Phase1CalibrationExample],
    canonicalizer: DeviceCanonicalizer,
    alignment_policy: AlignmentPolicy,
    frozen_tm: FrozenSamsungTM,
    expected_model_sha256: str,
) -> HardenedPhase1Artifact:
    artifact_path = Path(path)
    base = core.load_phase1_artifact(artifact_path, expected_model_sha256=expected_model_sha256)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("only an unsealed Phase 1 artifact can be sealed")
    support_threshold = _calibrated_support_threshold(
        calibration_examples,
        canonicalizer,
        base.feature_mean,
        base.feature_std,
    )
    margin_threshold = _calibrated_margin_threshold(
        calibration_examples,
        canonicalizer,
        frozen_tm,
        base,
    )
    sealed = dict(payload)
    sealed.update(
        {
            "schema_version": 2,
            "canonicalization_config": canonicalizer.config.to_dict(),
            "canonicalization_sha256": canonicalizer.config.sha256,
            "alignment_policy": alignment_policy.to_dict(),
            "alignment_policy_sha256": alignment_policy.sha256,
            "max_support_distance": support_threshold,
            "minimum_parameter_bound_margin": margin_threshold,
            "real_phase1_calibration_accepted": bool(
                base.data_mode == "real" and base.phase1_passed
            ),
            "real_source_replay_verified": False,
            "real_target_effectiveness_verified": False,
        }
    )
    torch.save(sealed, artifact_path)
    return load_hardened_phase1_artifact(
        artifact_path,
        expected_model_sha256=expected_model_sha256,
        expected_canonicalization_sha256=canonicalizer.config.sha256,
        expected_alignment_policy_sha256=alignment_policy.sha256,
    )


def _base_artifact_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_model_sha256: str | None,
) -> core.Phase1Artifact:
    if tuple(payload["feature_names"]) != PHASE1_FEATURE_NAMES:
        raise ValueError("Phase 1 artifact feature schema mismatch")
    model_sha = str(payload["samsung_model_sha256"])
    if expected_model_sha256 is not None and model_sha != expected_model_sha256:
        raise ValueError("Phase 1 artifact is bound to a different Samsung model")
    adapter_payload = payload["adapter"]
    adapter = TargetCameraAdapter(
        int(adapter_payload["feature_dim"]),
        int(adapter_payload["hidden_dim"]),
        curve_points=int(adapter_payload["curve_points"]),
        max_log_gain=float(adapter_payload["max_log_gain"]),
        max_matrix_delta=float(adapter_payload["max_matrix_delta"]),
    )
    adapter.load_state_dict(adapter_payload["state_dict"], strict=True)
    adapter.eval()
    return core.Phase1Artifact(
        adapter=adapter,
        feature_mean=payload["feature_mean"],
        feature_std=payload["feature_std"],
        support_min=payload["support_min"],
        support_max=payload["support_max"],
        samsung_model_sha256=model_sha,
        source_manifest_sha256=str(payload["source_manifest_sha256"]),
        calibration_manifest_sha256=str(payload["calibration_manifest_sha256"]),
        phase1_passed=bool(payload["phase1_passed"]),
        validation_report=payload["validation_report"],
        teacher_profile=core._profile_from_payload(payload["teacher_profile"]),
        data_mode=str(payload["data_mode"]),
    )


def load_hardened_phase1_artifact(
    path: Path | str,
    *,
    expected_model_sha256: str | None = None,
    expected_canonicalization_sha256: str | None = None,
    expected_alignment_policy_sha256: str | None = None,
) -> HardenedPhase1Artifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "feature_names",
        "feature_mean",
        "feature_std",
        "support_min",
        "support_max",
        "adapter",
        "samsung_model_sha256",
        "source_manifest_sha256",
        "calibration_manifest_sha256",
        "phase1_passed",
        "training_config",
        "validation_report",
        "teacher_profile",
        "data_mode",
        "canonicalization_config",
        "canonicalization_sha256",
        "alignment_policy",
        "alignment_policy_sha256",
        "max_support_distance",
        "minimum_parameter_bound_margin",
        "real_phase1_calibration_accepted",
        "real_source_replay_verified",
        "real_target_effectiveness_verified",
    }
    if not isinstance(payload, Mapping) or set(payload) != required or int(payload["schema_version"]) != 2:
        raise ValueError("hardened Phase 1 artifact schema is invalid")
    canonicalization = CanonicalizationConfig(**payload["canonicalization_config"])
    if canonicalization.sha256 != str(payload["canonicalization_sha256"]):
        raise ValueError("canonicalization configuration hash is invalid")
    policy = AlignmentPolicy(**payload["alignment_policy"])
    if policy.sha256 != str(payload["alignment_policy_sha256"]):
        raise ValueError("alignment policy hash is invalid")
    if expected_canonicalization_sha256 is not None and canonicalization.sha256 != expected_canonicalization_sha256:
        raise ValueError("canonicalization configuration does not match the Phase 1 artifact")
    if expected_alignment_policy_sha256 is not None and policy.sha256 != expected_alignment_policy_sha256:
        raise ValueError("alignment policy does not match the Phase 1 artifact")
    max_support_distance = float(payload["max_support_distance"])
    minimum_margin = float(payload["minimum_parameter_bound_margin"])
    if (
        not math.isfinite(max_support_distance)
        or not math.isfinite(minimum_margin)
        or max_support_distance < 0
        or minimum_margin < 0
    ):
        raise ValueError("hardened Phase 1 thresholds are invalid")
    return HardenedPhase1Artifact(
        base=_base_artifact_from_payload(payload, expected_model_sha256=expected_model_sha256),
        canonicalization_config=canonicalization,
        canonicalization_sha256=canonicalization.sha256,
        alignment_policy=policy,
        alignment_policy_sha256=policy.sha256,
        max_support_distance=max_support_distance,
        minimum_parameter_bound_margin=minimum_margin,
        real_phase1_calibration_accepted=bool(payload["real_phase1_calibration_accepted"]),
        real_source_replay_verified=bool(payload["real_source_replay_verified"]),
        real_target_effectiveness_verified=bool(payload["real_target_effectiveness_verified"]),
    )


def run_hardened_phase1_inference(
    *,
    image: torch.Tensor,
    metadata,
    frozen_tm: FrozenSamsungTM,
    artifact: HardenedPhase1Artifact,
    canonicalizer: DeviceCanonicalizer,
    require_real_artifact: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if require_real_artifact and artifact.data_mode != "real":
        raise ValueError("real-run requires a real Phase 1 artifact")
    if canonicalizer.config.sha256 != artifact.canonicalization_sha256:
        raise ValueError("canonicalization configuration does not match the Phase 1 artifact")
    output, manifest = core.run_phase1_inference(
        image=image,
        metadata=metadata,
        frozen_tm=frozen_tm,
        artifact=artifact.base,
        canonicalizer=canonicalizer,
    )
    support_distance = float(manifest["calibration_support_distance"])
    margin = float(manifest["adapter_parameter_bound_margin"])
    if not math.isfinite(support_distance) or support_distance > artifact.max_support_distance:
        raise ValueError("BLOCKED_OUTSIDE_CALIBRATION_SUPPORT")
    if not math.isfinite(margin) or margin <= 0.0 or margin < artifact.minimum_parameter_bound_margin:
        raise ValueError("BLOCKED_OUTSIDE_ADAPTER_SUPPORT")
    manifest.pop("real_data_effectiveness_verified", None)
    manifest.update(
        {
            "schema_version": 2,
            "canonicalization_sha256": artifact.canonicalization_sha256,
            "alignment_policy_sha256": artifact.alignment_policy_sha256,
            "max_support_distance": artifact.max_support_distance,
            "minimum_parameter_bound_margin": artifact.minimum_parameter_bound_margin,
            "real_phase1_calibration_accepted": artifact.real_phase1_calibration_accepted,
            "real_source_replay_verified": artifact.real_source_replay_verified,
            "real_target_effectiveness_verified": artifact.real_target_effectiveness_verified,
        }
    )
    return output, manifest


def evaluate_hardened_phase1_artifact(
    *,
    calibration_examples: Sequence[Phase1CalibrationExample],
    frozen_tm: FrozenSamsungTM,
    artifact: HardenedPhase1Artifact,
    canonicalizer: DeviceCanonicalizer,
) -> dict[str, Any]:
    if artifact.data_mode != "real":
        raise ValueError("evaluate-phase1 requires a real Phase 1 artifact")
    if canonicalizer.config.sha256 != artifact.canonicalization_sha256:
        raise ValueError("canonicalization configuration does not match the Phase 1 artifact")
    report = core.evaluate_phase1_artifact(
        calibration_examples=calibration_examples,
        frozen_tm=frozen_tm,
        artifact=artifact.base,
        canonicalizer=canonicalizer,
    )
    return {
        **report,
        "canonicalization_sha256": artifact.canonicalization_sha256,
        "alignment_policy_sha256": artifact.alignment_policy_sha256,
        "max_support_distance": artifact.max_support_distance,
        "minimum_parameter_bound_margin": artifact.minimum_parameter_bound_margin,
        "real_phase1_calibration_accepted": artifact.real_phase1_calibration_accepted,
        "real_source_replay_verified": artifact.real_source_replay_verified,
        "real_target_effectiveness_verified": artifact.real_target_effectiveness_verified,
    }
