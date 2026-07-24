from __future__ import annotations

import math
import re
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .adapters import LUMA_WEIGHTS
from .contracts import GateStatus, canonical_tensor_sha256
from .residuals import DirectionAlignmentResult


CRITICAL_GATE_NAMES = (
    "Phase1Valid",
    "SourceSupport",
    "Eligibility",
    "InputSupport",
    "DirectionAlignment",
    "IssueImprovement",
    "HighlightSafety",
    "Geometry",
    "HighFrequency",
    "ColorPreservation",
    "NonTargetRegression",
    "TMFeasibility",
    "BoundaryArtifact",
)


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


@dataclass(frozen=True)
class CertificationProfile:
    profile_id: str
    profile_sha256: str
    dataset_sha256: str
    synthetic: bool
    sample_count: int
    issue_lift_min: float
    issue_lift_max: float
    clipping_growth_max: float
    geometry_correlation_min: float
    high_frequency_correlation_min: float
    high_frequency_energy_min: float
    high_frequency_energy_max: float
    chromaticity_mae_max: float
    non_target_correction_max: float
    boundary_artifact_max: float

    _FIELDS = frozenset(
        {
            "profile_id",
            "profile_sha256",
            "dataset_sha256",
            "synthetic",
            "sample_count",
            "issue_lift_min",
            "issue_lift_max",
            "clipping_growth_max",
            "geometry_correlation_min",
            "high_frequency_correlation_min",
            "high_frequency_energy_min",
            "high_frequency_energy_max",
            "chromaticity_mae_max",
            "non_target_correction_max",
            "boundary_artifact_max",
        }
    )

    def __post_init__(self) -> None:
        if not self.profile_id or not _is_sha256(self.profile_sha256) or not _is_sha256(self.dataset_sha256):
            raise ValueError("certification profile requires an id and exact SHA-256 provenance")
        if self.sample_count < 1:
            raise ValueError("certification profile sample_count must be positive")
        numeric = (
            self.issue_lift_min,
            self.issue_lift_max,
            self.clipping_growth_max,
            self.geometry_correlation_min,
            self.high_frequency_correlation_min,
            self.high_frequency_energy_min,
            self.high_frequency_energy_max,
            self.chromaticity_mae_max,
            self.non_target_correction_max,
            self.boundary_artifact_max,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("certification thresholds must be finite")
        if not 0.0 <= self.issue_lift_min <= self.issue_lift_max:
            raise ValueError("issue lift interval is invalid")

    @property
    def real_calibrated(self) -> bool:
        return not self.synthetic and self.sample_count >= 30

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CertificationProfile":
        unknown = sorted(set(payload) - cls._FIELDS)
        missing = sorted(cls._FIELDS - set(payload))
        if unknown or missing:
            raise ValueError(
                "invalid certification profile fields; unknown="
                + ",".join(unknown)
                + "; missing="
                + ",".join(missing)
            )
        return cls(**{key: payload[key] for key in cls._FIELDS})

    @classmethod
    def from_calibration(
        cls,
        rows: Sequence[Mapping[str, float]],
        *,
        profile_id: str,
        dataset_sha256: str,
        synthetic: bool,
    ) -> "CertificationProfile":
        if len(rows) < 30:
            raise ValueError("gate calibration requires at least 30 samples")
        metric_names = (
            "issue_lift",
            "clipping_growth",
            "geometry_correlation",
            "high_frequency_correlation",
            "high_frequency_energy_ratio",
            "chromaticity_mae",
            "non_target_correction",
            "boundary_artifact",
        )
        if any(set(row) != set(metric_names) for row in rows):
            raise ValueError("calibration metric schema mismatch")
        columns = {
            name: torch.tensor([float(row[name]) for row in rows], dtype=torch.float64)
            for name in metric_names
        }
        if not all(torch.isfinite(values).all() for values in columns.values()):
            raise ValueError("calibration metrics must be finite")
        q = lambda name, level: float(torch.quantile(columns[name], level).item())
        thresholds = {
            "issue_lift_min": q("issue_lift", 0.10),
            "issue_lift_max": q("issue_lift", 0.90),
            "clipping_growth_max": q("clipping_growth", 0.95),
            "geometry_correlation_min": q("geometry_correlation", 0.05),
            "high_frequency_correlation_min": q("high_frequency_correlation", 0.05),
            "high_frequency_energy_min": q("high_frequency_energy_ratio", 0.05),
            "high_frequency_energy_max": q("high_frequency_energy_ratio", 0.95),
            "chromaticity_mae_max": q("chromaticity_mae", 0.95),
            "non_target_correction_max": q("non_target_correction", 0.95),
            "boundary_artifact_max": q("boundary_artifact", 0.95),
        }
        provenance = {
            "profile_id": profile_id,
            "dataset_sha256": dataset_sha256,
            "synthetic": synthetic,
            "sample_count": len(rows),
            **thresholds,
        }
        profile_sha256 = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            profile_id=profile_id,
            profile_sha256=profile_sha256,
            dataset_sha256=dataset_sha256,
            synthetic=synthetic,
            sample_count=len(rows),
            **thresholds,
        )


@dataclass(frozen=True)
class CertificationInputs:
    phase1_valid: bool | None
    source_supported: bool | None
    eligible: bool | None
    input_supported: bool | None
    direction_alignment: DirectionAlignmentResult | None
    clipping_growth: float | None
    geometry_correlation: float | None
    high_frequency_correlation: float | None
    high_frequency_energy_ratio: float | None
    chromaticity_mae: float | None
    non_target_correction: float | None
    tm_feasible: bool | None
    boundary_artifact: float | None
    candidate_kind: str


@dataclass(frozen=True)
class GateEvidence:
    name: str
    status: GateStatus
    value: Any
    threshold: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CertificationResult:
    accepted: bool
    full_certification: bool
    candidate_sha256: str
    profile_sha256: str
    gates: tuple[GateEvidence, ...]

    def gate(self, name: str) -> GateEvidence:
        for gate in self.gates:
            if gate.name == name:
                return gate
        raise KeyError(name)


def _bool_gate(name: str, value: bool | None, *, false_reason: str) -> GateEvidence:
    if value is None:
        return GateEvidence(name, GateStatus.UNAVAILABLE, None, "required", "signal_unavailable")
    return GateEvidence(
        name,
        GateStatus.PASS if value else GateStatus.FAIL,
        value,
        "true",
        "ok" if value else false_reason,
    )


def _upper_gate(name: str, value: float | None, maximum: float) -> GateEvidence:
    if value is None or not math.isfinite(float(value)):
        return GateEvidence(name, GateStatus.UNAVAILABLE, value, f"<= {maximum}", "signal_unavailable")
    passed = float(value) <= maximum
    return GateEvidence(name, GateStatus.PASS if passed else GateStatus.FAIL, float(value), f"<= {maximum}", "ok" if passed else "threshold")


class Certifier:
    def certify(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        roi_mask: torch.Tensor | None,
        inputs: CertificationInputs,
        profile: CertificationProfile,
    ) -> CertificationResult:
        if reference.shape != candidate.shape or reference.ndim != 4 or reference.shape[1] != 3:
            raise ValueError("certification images must share shape [B,3,H,W]")
        if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
            raise ValueError("certification rejects non-finite images")
        gates: list[GateEvidence] = []
        gates.append(_bool_gate("Phase1Valid", inputs.phase1_valid, false_reason="phase1_invalid"))
        gates.append(_bool_gate("SourceSupport", inputs.source_supported, false_reason="outside_source_support"))
        eligible = inputs.eligible
        eligibility_reason = "ineligible"
        if inputs.candidate_kind == "raw_generated":
            eligible = False
            eligibility_reason = "raw_generated_requires_projection"
        gates.append(_bool_gate("Eligibility", eligible, false_reason=eligibility_reason))
        gates.append(_bool_gate("InputSupport", inputs.input_supported, false_reason="input_unsupported"))

        if inputs.direction_alignment is None:
            gates.append(GateEvidence("DirectionAlignment", GateStatus.UNAVAILABLE, None, "PASS", "signal_unavailable"))
        else:
            gates.append(
                GateEvidence(
                    "DirectionAlignment",
                    inputs.direction_alignment.status,
                    inputs.direction_alignment.candidate_distance,
                    f"distance < {inputs.direction_alignment.baseline_distance}",
                    "ok" if inputs.direction_alignment.status is GateStatus.PASS else ",".join(inputs.direction_alignment.reasons),
                )
            )

        weights = reference.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        reference_luma = (reference * weights).sum(dim=1, keepdim=True)
        candidate_luma = (candidate * weights).sum(dim=1, keepdim=True)
        if roi_mask is None or roi_mask.shape != reference_luma.shape or not (roi_mask > 0).any():
            issue_gate = GateEvidence(
                "IssueImprovement",
                GateStatus.UNAVAILABLE,
                None,
                f"[{profile.issue_lift_min}, {profile.issue_lift_max}]",
                "dynamic_roi_unavailable",
            )
        else:
            selected = roi_mask > 0
            log_lift = torch.log(candidate_luma[selected].clamp_min(1e-6)) - torch.log(
                reference_luma[selected].clamp_min(1e-6)
            )
            value = float(torch.median(log_lift).item())
            passed = profile.issue_lift_min <= value <= profile.issue_lift_max
            issue_gate = GateEvidence(
                "IssueImprovement",
                GateStatus.PASS if passed else GateStatus.FAIL,
                value,
                f"[{profile.issue_lift_min}, {profile.issue_lift_max}]",
                "ok" if passed else "outside_calibrated_interval",
            )
        gates.append(issue_gate)
        gates.append(_upper_gate("HighlightSafety", inputs.clipping_growth, profile.clipping_growth_max))

        if inputs.geometry_correlation is None or not math.isfinite(float(inputs.geometry_correlation)):
            gates.append(GateEvidence("Geometry", GateStatus.UNAVAILABLE, None, f">= {profile.geometry_correlation_min}", "signal_unavailable"))
        else:
            passed = inputs.geometry_correlation >= profile.geometry_correlation_min
            gates.append(GateEvidence("Geometry", GateStatus.PASS if passed else GateStatus.FAIL, inputs.geometry_correlation, f">= {profile.geometry_correlation_min}", "ok" if passed else "threshold"))

        hf_values = (inputs.high_frequency_correlation, inputs.high_frequency_energy_ratio)
        if any(value is None or not math.isfinite(float(value)) for value in hf_values):
            gates.append(GateEvidence("HighFrequency", GateStatus.UNAVAILABLE, None, "correlation and energy interval", "signal_unavailable"))
        else:
            correlation = float(inputs.high_frequency_correlation)  # type: ignore[arg-type]
            energy = float(inputs.high_frequency_energy_ratio)  # type: ignore[arg-type]
            passed = (
                correlation >= profile.high_frequency_correlation_min
                and profile.high_frequency_energy_min <= energy <= profile.high_frequency_energy_max
            )
            gates.append(GateEvidence("HighFrequency", GateStatus.PASS if passed else GateStatus.FAIL, {"correlation": correlation, "energy_ratio": energy}, f"corr >= {profile.high_frequency_correlation_min}; energy in [{profile.high_frequency_energy_min}, {profile.high_frequency_energy_max}]", "ok" if passed else "threshold"))
        gates.append(_upper_gate("ColorPreservation", inputs.chromaticity_mae, profile.chromaticity_mae_max))
        gates.append(_upper_gate("NonTargetRegression", inputs.non_target_correction, profile.non_target_correction_max))
        gates.append(_bool_gate("TMFeasibility", inputs.tm_feasible, false_reason="outside_tm_space"))
        gates.append(_upper_gate("BoundaryArtifact", inputs.boundary_artifact, profile.boundary_artifact_max))
        if tuple(gate.name for gate in gates) != CRITICAL_GATE_NAMES:
            raise RuntimeError("certification gate set/order is incomplete")
        accepted = all(gate.status is GateStatus.PASS for gate in gates)
        return CertificationResult(
            accepted=accepted,
            full_certification=len(gates) == len(CRITICAL_GATE_NAMES),
            candidate_sha256=canonical_tensor_sha256(candidate),
            profile_sha256=profile.profile_sha256,
            gates=tuple(gates),
        )
