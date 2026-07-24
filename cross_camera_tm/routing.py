from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .certification import CertificationProfile, CertificationResult


class Route(str, Enum):
    DIAGNOSTIC = "diagnostic"
    REJECT = "reject"
    PARAMETER = "parameter"
    RANGE = "range"
    PREFERENCE = "preference"
    PIXEL = "pixel"


@dataclass(frozen=True)
class StructuredUncertainty:
    source_supported: bool
    residual_interval_width: float
    diagnosis_reliable: bool
    teacher_agreement: bool
    projection_retention: float
    arbiter_available: bool
    arbiter_order_stable: bool
    arbiter_preference: str
    metadata_complete: bool
    ovis_available: bool
    ovis_hard_defect: bool


@dataclass(frozen=True)
class SupervisionCandidate:
    teacher_level: str
    raw_generated: bool
    projected: bool
    full_recertified: bool
    certification: CertificationResult
    uncertainty: StructuredUncertainty
    parameter_stable: bool
    range_bounded: bool
    real_data: bool
    calibration_profile: CertificationProfile


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    reasons: tuple[str, ...]


class SupervisionRouter:
    def __init__(self, *, pixel_route_enabled: bool = False):
        self.pixel_route_enabled = pixel_route_enabled

    def route(self, candidate: SupervisionCandidate) -> RoutingDecision:
        uncertainty = candidate.uncertainty
        if not uncertainty.source_supported or not uncertainty.metadata_complete:
            return RoutingDecision(Route.DIAGNOSTIC, ("source_or_metadata_support",))
        if uncertainty.ovis_available and uncertainty.ovis_hard_defect:
            return RoutingDecision(Route.REJECT, ("ovis_hard_defect",))
        if candidate.raw_generated:
            return RoutingDecision(Route.REJECT, ("raw_generated_pixel_forbidden",))
        if not candidate.certification.accepted:
            failed = tuple(
                gate.name for gate in candidate.certification.gates if gate.status.value != "pass"
            )
            return RoutingDecision(Route.REJECT, failed or ("certification",))
        if candidate.teacher_level == "L1" and candidate.parameter_stable:
            return RoutingDecision(Route.PARAMETER, ("stable_lowest_sufficient_teacher",))
        if candidate.teacher_level == "L2" and candidate.range_bounded:
            return RoutingDecision(Route.RANGE, ("bounded_non_unique_spatial_solution",))
        if not candidate.projected or not candidate.full_recertified:
            return RoutingDecision(Route.REJECT, ("projection_and_recertification_required",))
        if candidate.real_data and not candidate.calibration_profile.real_calibrated:
            return RoutingDecision(Route.PREFERENCE, ("real_calibration_profile",))
        if not self.pixel_route_enabled:
            return RoutingDecision(Route.PREFERENCE, ("pixel_route_disabled",))
        if (
            not uncertainty.teacher_agreement
            or uncertainty.projection_retention <= 0.0
            or not uncertainty.arbiter_available
            or not uncertainty.arbiter_order_stable
            or uncertainty.arbiter_preference != "PROJECTED"
        ):
            return RoutingDecision(Route.PREFERENCE, ("perceptual_or_teacher_ambiguity",))
        return RoutingDecision(Route.PIXEL, ("all_pixel_necessary_conditions",))
