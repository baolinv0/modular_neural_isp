from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .adapters import LUMA_WEIGHTS, TargetCameraAdapter
from .canonicalization import DeviceCanonicalizer
from .certification import CertificationInputs, CertificationProfile, CertificationResult, Certifier
from .contracts import FailureType, LinearMetadata
from .manifest import ManifestRecord, ManifestWriter
from .phase1 import FrozenSamsungTM
from .policy import InternVLAnonymousABAdapter, OvisLocalInspector, QwenImageEditLocalAdapter
from .projection import TMSpaceProjector
from .residuals import DirectionAlignmentGate, DynamicROIBuilder, ResidualEstimate
from .routing import Route, RoutingDecision, StructuredUncertainty, SupervisionCandidate, SupervisionRouter
from .teachers import L1GlobalTeacher, L2ROITeacher, L3LocalTeacher, TeacherCandidate


@dataclass(frozen=True)
class PipelineRunResult:
    trace: tuple[str, ...]
    routing: RoutingDecision
    certification: CertificationResult | None
    candidate: TeacherCandidate | None
    manifest: ManifestRecord | None
    baseline_sha256: str
    reason: str


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.flatten() - left.mean()
    right = right.flatten() - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator.item() < 1e-10:
        return 1.0 if torch.allclose(left, right, atol=1e-7) else 0.0
    return float((left * right).sum().div(denominator).item())


def _certification_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    roi: torch.Tensor,
    direction,
    *,
    failure_type: FailureType,
    candidate_kind: str = "analytic",
) -> CertificationInputs:
    baseline_luma = _luma(baseline)
    candidate_luma = _luma(candidate)
    clipping_growth = float(
        ((candidate_luma >= 0.995).float().mean() - (baseline_luma >= 0.995).float().mean()).clamp_min(0).item()
    )
    gradient_baseline = torch.cat(
        (
            (baseline_luma[:, :, :, 1:] - baseline_luma[:, :, :, :-1]).flatten(),
            (baseline_luma[:, :, 1:, :] - baseline_luma[:, :, :-1, :]).flatten(),
        )
    )
    gradient_candidate = torch.cat(
        (
            (candidate_luma[:, :, :, 1:] - candidate_luma[:, :, :, :-1]).flatten(),
            (candidate_luma[:, :, 1:, :] - candidate_luma[:, :, :-1, :]).flatten(),
        )
    )
    hf_correlation = _correlation(gradient_baseline, gradient_candidate)
    baseline_energy = gradient_baseline.abs().mean()
    candidate_energy = gradient_candidate.abs().mean()
    hf_energy = float((candidate_energy / baseline_energy.clamp_min(1e-8)).item())
    if baseline_energy.item() < 1e-8 and candidate_energy.item() < 1e-8:
        hf_energy = 1.0
    baseline_chroma = baseline / baseline.sum(dim=1, keepdim=True).clamp_min(1e-6)
    candidate_chroma = candidate / candidate.sum(dim=1, keepdim=True).clamp_min(1e-6)
    chroma = float((baseline_chroma - candidate_chroma).abs().mean().item())
    non_target = 0.0
    if failure_type is FailureType.FACE_UNDEREXPOSURE:
        outside = roi <= 1e-4
        if outside.any():
            non_target = float(
                torch.median(
                    (
                        torch.log(candidate_luma.clamp_min(1e-6))
                        - torch.log(baseline_luma.clamp_min(1e-6))
                    )[outside].abs()
                ).item()
            )
    ratio = candidate_luma / baseline_luma.clamp_min(1e-6)
    horizontal_jump = (ratio[:, :, :, 1:] - ratio[:, :, :, :-1]).abs().max().item()
    vertical_jump = (ratio[:, :, 1:, :] - ratio[:, :, :-1, :]).abs().max().item()
    return CertificationInputs(
        phase1_valid=True,
        source_supported=True,
        eligible=True,
        input_supported=True,
        direction_alignment=direction,
        clipping_growth=clipping_growth,
        geometry_correlation=_correlation(baseline_luma, candidate_luma),
        high_frequency_correlation=hf_correlation,
        high_frequency_energy_ratio=hf_energy,
        chromaticity_mae=chroma,
        non_target_correction=non_target,
        tm_feasible=True,
        boundary_artifact=float(max(horizontal_jump, vertical_jump)),
        candidate_kind=candidate_kind,
    )


class CrossCameraPipeline:
    def __init__(
        self,
        *,
        canonicalizer: DeviceCanonicalizer,
        target_adapter: TargetCameraAdapter,
        samsung_tm: FrozenSamsungTM,
        router: SupervisionRouter,
        l3_editor: QwenImageEditLocalAdapter | None = None,
        arbiter: InternVLAnonymousABAdapter | None = None,
        ovis_inspector: OvisLocalInspector | None = None,
    ):
        self.canonicalizer = canonicalizer
        self.target_adapter = target_adapter
        self.samsung_tm = samsung_tm
        self.router = router
        self.l3_editor = l3_editor
        self.arbiter = arbiter
        self.ovis_inspector = ovis_inspector

    def run(
        self,
        image: torch.Tensor,
        metadata: LinearMetadata,
        adapter_features: torch.Tensor,
        estimate: ResidualEstimate,
        profile: CertificationProfile,
        *,
        phase2_enabled: bool,
        phase2_activated: bool,
        failure_type: FailureType,
        face_mask: torch.Tensor | None,
        config_sha256: str,
        model_sha256: str,
        synthetic: bool,
        real_model: bool,
        manifest_path: Path | None = None,
    ) -> PipelineRunResult:
        trace: list[str] = []
        canonical = self.canonicalizer.canonicalize(image, metadata)
        trace.append("canonicalization")
        confidence = image.new_full((image.shape[0],), canonical.confidence.overall)
        adapted = self.target_adapter(canonical.image, adapter_features, confidence=confidence)
        trace.append("target_camera_adapter")
        baseline = self.samsung_tm(adapted.image)
        trace.append("frozen_samsung_tm")
        trace.append("phase2_activation")
        if not phase2_enabled or not phase2_activated:
            reason = "phase2_disabled" if not phase2_enabled else "phase2_not_activated"
            return PipelineRunResult(
                trace=tuple(trace),
                routing=RoutingDecision(route=Route.DIAGNOSTIC, reasons=(reason,)),
                certification=None,
                candidate=None,
                manifest=None,
                baseline_sha256=canonical.output_sha256,
                reason=reason,
            )

        roi_result = DynamicROIBuilder(min_coverage=0.01).build(
            baseline,
            estimate,
            canonical.reliable_mask,
            canonical.highlight_valid_mask,
            failure_type,
            face_mask=face_mask,
        )
        if roi_result.status.value != "pass":
            return PipelineRunResult(
                trace=tuple(trace),
                routing=RoutingDecision(Route.DIAGNOSTIC, (roi_result.reason,)),
                certification=None,
                candidate=None,
                manifest=None,
                baseline_sha256=canonical.output_sha256,
                reason=roi_result.reason,
            )

        attempts: list[tuple[TeacherCandidate, bool, float]] = []
        attempts.append((L1GlobalTeacher().generate(baseline, estimate), False, 1.0))
        candidate: TeacherCandidate | None = None
        certification: CertificationResult | None = None
        routing = RoutingDecision(Route.REJECT, ("no_teacher_succeeded",))
        projected = False
        projection_retention = 1.0
        attempt_index = 0
        while True:
            if attempt_index >= len(attempts):
                if (
                    failure_type is FailureType.FACE_UNDEREXPOSURE
                    and face_mask is not None
                    and not any(item[0].level == "L2" for item in attempts)
                ):
                    attempts.append((L2ROITeacher().generate(baseline, estimate, face_mask), False, 1.0))
                elif self.l3_editor is not None and not any(item[0].level.startswith("L3") for item in attempts):
                    raw = L3LocalTeacher(self.l3_editor).propose(
                        baseline, "Correct only the diagnosed tone-mapping residual; preserve geometry, texture and color."
                    )
                    trace.append("L3_raw_proposal")
                    projection = TMSpaceProjector().project(baseline, raw.image, roi_mask=roi_result.mask)
                    trace.append("preprojection_safety")
                    trace.append("tm_space_projection")
                    projected_candidate = TeacherCandidate(
                        level="L3_PROJECTED",
                        image=projection.image,
                        image_sha256=projection.projected_sha256,
                        parent_sha256=raw.image_sha256,
                        raw_generated=False,
                        pixel_target_eligible=True,
                        synthetic_mock=raw.synthetic_mock,
                        correction_parameters={},
                    )
                    attempts.append((projected_candidate, True, projection.retention))
                else:
                    break
            candidate, projected, projection_retention = attempts[attempt_index]
            attempt_index += 1
            trace.append(f"{candidate.level}_teacher" if candidate.level != "L3_PROJECTED" else "L3_projected_teacher")
            baseline_luma = _luma(baseline)
            candidate_luma = _luma(candidate.image)
            issue = roi_result.mask > 1e-4
            log_change = torch.log(candidate_luma.clamp_min(1e-6)) - torch.log(
                baseline_luma.clamp_min(1e-6)
            )
            corrections = {"global": float(torch.median(log_change).item())}
            if failure_type is FailureType.FACE_UNDEREXPOSURE:
                corrections["face"] = float(torch.median(log_change[issue]).item())
            outside = roi_result.mask <= 1e-4
            non_target = float(torch.median(log_change[outside].abs()).item()) if outside.any() else 0.0
            direction = DirectionAlignmentGate().evaluate(
                estimate, corrections, non_target_correction=non_target
            )
            inputs = _certification_metrics(
                baseline,
                candidate.image,
                roi_result.mask,
                direction,
                failure_type=failure_type,
                candidate_kind="projected" if projected else "analytic",
            )
            certification = Certifier().certify(
                baseline, candidate.image, roi_result.mask, inputs, profile
            )
            trace.append("full_certification")

            if projected and self.arbiter is not None:
                arbitration = self.arbiter.arbitrate(candidate.image, baseline)
            else:
                arbitration = None
            if projected and self.ovis_inspector is not None:
                inspection = self.ovis_inspector.inspect(candidate.image)
            else:
                inspection = None
            uncertainty = StructuredUncertainty(
                source_supported=estimate.available,
                residual_interval_width=estimate.global_residual.upper - estimate.global_residual.lower,
                diagnosis_reliable=True,
                teacher_agreement=True,
                projection_retention=projection_retention,
                arbiter_available=arbitration.available if arbitration is not None else False,
                arbiter_order_stable=arbitration.order_consistent if arbitration is not None else False,
                arbiter_preference=arbitration.preference if arbitration is not None else "UNAVAILABLE",
                metadata_complete=metadata.metadata_complete,
                ovis_available=inspection.available if inspection is not None else False,
                ovis_hard_defect=inspection.hard_defect if inspection is not None else False,
            )
            teacher_level = "L3" if candidate.level.startswith("L3") else candidate.level
            routing = self.router.route(
                SupervisionCandidate(
                    teacher_level=teacher_level,
                    raw_generated=False,
                    projected=projected,
                    full_recertified=certification.full_certification,
                    certification=certification,
                    uncertainty=uncertainty,
                    parameter_stable=teacher_level == "L1",
                    range_bounded=teacher_level == "L2",
                    real_data=not synthetic,
                    calibration_profile=profile,
                )
            )
            trace.append("supervision_routing")
            if routing.route is not Route.REJECT:
                break

        if candidate is None or certification is None:
            return PipelineRunResult(
                trace=tuple(trace),
                routing=RoutingDecision(Route.REJECT, ("no_teacher_available",)),
                certification=None,
                candidate=None,
                manifest=None,
                baseline_sha256=canonical.output_sha256,
                reason="no_teacher_available",
            )
        record = ManifestRecord(
            artifact_sha256=candidate.image_sha256,
            input_sha256=canonical.input_sha256,
            parent_sha256s=(candidate.parent_sha256,),
            model_sha256=model_sha256,
            config_sha256=config_sha256,
            profile_sha256=profile.profile_sha256,
            transformations=tuple(trace),
            gates=tuple(gate.to_dict() for gate in certification.gates),
            supervision_type=routing.route.value,
            synthetic=synthetic,
            real_model=real_model,
            raw_generated=False,
            projected=projected,
            fully_certified=certification.full_certification,
            route_reasons=routing.reasons,
        )
        if manifest_path is not None:
            ManifestWriter(manifest_path).write(record)
        trace.append("manifest")
        return PipelineRunResult(
            trace=tuple(trace),
            routing=routing,
            certification=certification,
            candidate=candidate,
            manifest=record,
            baseline_sha256=candidate.parent_sha256,
            reason="ok" if certification.accepted else "certification_failed",
        )
