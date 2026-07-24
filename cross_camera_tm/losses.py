from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as functional

from .adapters import LUMA_WEIGHTS, PairTransformParameters
from .contracts import AlignmentQuality


@dataclass(frozen=True)
class DistillationLossResult:
    total: torch.Tensor
    tone: torch.Tensor
    roi: torch.Tensor
    lowfreq: torch.Tensor
    adapter: torch.Tensor
    enabled: tuple[str, ...]


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


class DistillationLoss:
    def __init__(
        self,
        *,
        tone_weight: float = 1.0,
        roi_weight: float = 0.5,
        lowfreq_weight: float = 0.5,
        adapter_weight: float = 0.01,
    ):
        self.tone_weight = float(tone_weight)
        self.roi_weight = float(roi_weight)
        self.lowfreq_weight = float(lowfreq_weight)
        self.adapter_weight = float(adapter_weight)

    def __call__(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        *,
        alignment: AlignmentQuality,
        roi_mask: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        parameters: PairTransformParameters | None = None,
    ) -> DistillationLossResult:
        if student.shape != teacher.shape or student.ndim != 4 or student.shape[1] != 3:
            raise ValueError("student and teacher must share shape [B,3,H,W]")
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise ValueError("distillation inputs must be finite")
        student_luma = _luma(student)
        teacher_luma = _luma(teacher)
        quantiles = student.new_tensor((0.1, 0.25, 0.5, 0.75, 0.9))
        student_q = torch.quantile(torch.log1p(student_luma.flatten(1)), quantiles, dim=1)
        teacher_q = torch.quantile(torch.log1p(teacher_luma.flatten(1)), quantiles, dim=1)
        tone = (student_q - teacher_q).abs().mean()
        zero = student.sum() * 0.0
        roi = zero
        lowfreq = zero
        enabled: list[str] = ["tone"]

        if alignment in (AlignmentQuality.ROI, AlignmentQuality.LOW_FREQUENCY):
            if roi_mask is None or roi_mask.shape != student_luma.shape:
                raise ValueError("ROI alignment requires roi_mask with shape [B,1,H,W]")
            roi_float = roi_mask.to(dtype=student.dtype)
            roi = ((student_luma - teacher_luma).abs() * roi_float).sum() / roi_float.sum().clamp_min(1.0)
            enabled.append("roi")

        if alignment is AlignmentQuality.LOW_FREQUENCY:
            if alignment_mask is None or alignment_mask.shape != student_luma.shape:
                raise ValueError("low-frequency alignment requires alignment_mask")
            pooled_student = functional.avg_pool2d(student_luma, kernel_size=4, stride=4)
            pooled_teacher = functional.avg_pool2d(teacher_luma, kernel_size=4, stride=4)
            pooled_mask = functional.avg_pool2d(alignment_mask.to(student.dtype), kernel_size=4, stride=4)
            lowfreq = ((pooled_student - pooled_teacher).abs() * pooled_mask).sum() / pooled_mask.sum().clamp_min(1.0)
            enabled.append("lowfreq")

        adapter = zero
        if parameters is not None:
            identity_matrix = torch.eye(3, device=student.device, dtype=student.dtype).unsqueeze(0)
            identity_curve = torch.linspace(
                0.0, 1.0, parameters.curve_y.shape[1], device=student.device, dtype=student.dtype
            ).unsqueeze(0)
            adapter = (
                torch.log(parameters.gains.clamp_min(1e-6)).square().mean()
                + (parameters.matrix - identity_matrix).square().mean()
                + (parameters.curve_y - identity_curve).square().mean()
            )
        enabled.append("adapter")
        total = (
            self.tone_weight * tone
            + self.roi_weight * roi
            + self.lowfreq_weight * lowfreq
            + self.adapter_weight * adapter
        )
        return DistillationLossResult(
            total=total,
            tone=tone,
            roi=roi,
            lowfreq=lowfreq,
            adapter=adapter,
            enabled=tuple(enabled),
        )
