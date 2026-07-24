from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .adapters import LUMA_WEIGHTS, _piecewise_curve
from .contracts import GateStatus, canonical_tensor_sha256


@dataclass(frozen=True)
class PreProjectionSafetyResult:
    status: GateStatus
    reasons: tuple[str, ...]


class PreProjectionSafetyChecker:
    def __init__(self, *, maximum_mean_change: float = 0.35):
        self.maximum_mean_change = float(maximum_mean_change)

    def check(self, reference: torch.Tensor, proposal: torch.Tensor) -> PreProjectionSafetyResult:
        reasons: list[str] = []
        if reference.ndim != 4 or reference.shape[1] != 3 or proposal.shape != reference.shape:
            reasons.append("shape_or_geometry")
        elif not torch.isfinite(reference).all() or not torch.isfinite(proposal).all():
            reasons.append("non_finite")
        else:
            if proposal.min().item() < 0.0 or proposal.max().item() > 1.0:
                reasons.append("range")
            if (proposal - reference).abs().mean().item() > self.maximum_mean_change:
                reasons.append("gross_change")
        return PreProjectionSafetyResult(GateStatus.FAIL if reasons else GateStatus.PASS, tuple(reasons))


@dataclass(frozen=True)
class ProjectionResult:
    image: torch.Tensor
    curve_y: torch.Tensor
    log_gain_grid: torch.Tensor
    raw_proposal_sha256: str
    projected_sha256: str
    raw_generated_eligible: bool
    requires_full_recertification: bool
    retention: float


def _smooth_grid(grid: torch.Tensor) -> torch.Tensor:
    kernel = grid.new_tensor((1.0, 2.0, 1.0))
    kernel = torch.outer(kernel, kernel)
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
    padded = functional.pad(grid, (1, 1, 1, 1), mode="replicate")
    return functional.conv2d(padded, kernel)


class TMSpaceProjector:
    def __init__(self, *, curve_points: int = 6, grid_size: int = 8, max_log_gain: float = 0.18):
        self.curve_points = curve_points
        self.grid_size = grid_size
        self.max_log_gain = float(max_log_gain)
        self.safety = PreProjectionSafetyChecker()

    def project(
        self,
        reference: torch.Tensor,
        proposal: torch.Tensor,
        *,
        roi_mask: torch.Tensor | None = None,
    ) -> ProjectionResult:
        safety = self.safety.check(reference, proposal)
        if safety.status is not GateStatus.PASS:
            raise ValueError("pre-projection safety failed: " + ",".join(safety.reasons))
        weights = reference.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        reference_luma = (reference * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        proposal_luma = (proposal * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        x_points = torch.linspace(0.0, 1.0, self.curve_points, device=reference.device, dtype=reference.dtype)
        curves: list[torch.Tensor] = []
        for batch in range(reference.shape[0]):
            source = reference_luma[batch].flatten()
            target = proposal_luma[batch].flatten()
            bandwidth = 1.0 / max(2, self.curve_points - 1)
            kernel = torch.exp(-0.5 * ((x_points[:, None] - source[None, :]) / bandwidth).square())
            residual = (kernel * (target - source)[None, :]).sum(dim=1) / kernel.sum(dim=1).clamp_min(1e-8)
            curve = (x_points + residual).clamp(0.0, 1.0)
            curve[0], curve[-1] = 0.0, 1.0
            curve = torch.cummax(curve, dim=0).values
            curve[-1] = 1.0
            curves.append(curve)
        curve_y = torch.stack(curves)
        curve_luma = _piecewise_curve(reference_luma, curve_y)
        residual_log_gain = torch.log(
            proposal_luma.clamp_min(1e-5) / curve_luma.clamp_min(1e-5)
        ).clamp(-self.max_log_gain, self.max_log_gain)
        log_gain_grid = functional.adaptive_avg_pool2d(residual_log_gain, (self.grid_size, self.grid_size))
        log_gain_grid = _smooth_grid(log_gain_grid)
        log_gain = functional.interpolate(
            log_gain_grid,
            size=reference.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).clamp(-self.max_log_gain, self.max_log_gain)
        log_gain = _smooth_grid(log_gain)
        if roi_mask is not None:
            if roi_mask.shape != reference_luma.shape:
                raise ValueError("roi_mask must have shape [B,1,H,W]")
            log_gain = log_gain * roi_mask.to(reference.dtype).clamp(0.0, 1.0)
        projected_luma = (curve_luma * torch.exp(log_gain)).clamp(0.0, 1.0)
        projected = (reference * (projected_luma / reference_luma.clamp_min(1e-6))).clamp(0.0, 1.0)
        desired_change = (proposal_luma - reference_luma).abs().mean()
        retained_change = (projected_luma - reference_luma).abs().mean()
        retention = float((retained_change / desired_change.clamp_min(1e-6)).clamp(0.0, 2.0).item())
        return ProjectionResult(
            image=projected,
            curve_y=curve_y,
            log_gain_grid=log_gain_grid,
            raw_proposal_sha256=canonical_tensor_sha256(proposal),
            projected_sha256=canonical_tensor_sha256(projected),
            raw_generated_eligible=False,
            requires_full_recertification=True,
            retention=retention,
        )
