from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .contracts import canonical_tensor_sha256
from .policy import QwenImageEditLocalAdapter
from .residuals import ResidualEstimate


@dataclass(frozen=True)
class TeacherCandidate:
    level: str
    image: torch.Tensor
    image_sha256: str
    parent_sha256: str
    raw_generated: bool
    pixel_target_eligible: bool
    synthetic_mock: bool
    correction_parameters: dict[str, float]


class L1GlobalTeacher:
    def generate(self, baseline: torch.Tensor, estimate: ResidualEstimate) -> TeacherCandidate:
        if not estimate.available:
            raise RuntimeError("source residual estimate unavailable")
        gain = math.exp(estimate.global_residual.center)
        image = (baseline * gain).clamp(0.0, 1.0)
        return TeacherCandidate(
            level="L1",
            image=image,
            image_sha256=canonical_tensor_sha256(image),
            parent_sha256=canonical_tensor_sha256(baseline),
            raw_generated=False,
            pixel_target_eligible=False,
            synthetic_mock=False,
            correction_parameters={"global_log_gain": estimate.global_residual.center},
        )


def _gaussian_feather(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel_1d = torch.exp(-0.5 * (coordinates / max(sigma, 1e-3)).square())
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    return functional.conv2d(mask, kernel, padding=radius).clamp(0.0, 1.0)


class L2ROITeacher:
    def generate(
        self, baseline: torch.Tensor, estimate: ResidualEstimate, face_mask: torch.Tensor
    ) -> TeacherCandidate:
        if not estimate.available:
            raise RuntimeError("source residual estimate unavailable")
        expected = (baseline.shape[0], 1, baseline.shape[2], baseline.shape[3])
        if face_mask.shape != expected or not face_mask.any():
            raise ValueError("L2 requires a non-empty face mask")
        approximate_width = float(face_mask.float().sum().sqrt().item())
        soft_face = _gaussian_feather(face_mask.to(baseline.dtype), max(0.7, approximate_width * 0.08))
        global_log_gain = estimate.global_residual.center
        face_log_gain = estimate.face_residual.center
        log_gain = global_log_gain + soft_face * (face_log_gain - global_log_gain)
        image = (baseline * torch.exp(log_gain)).clamp(0.0, 1.0)
        return TeacherCandidate(
            level="L2",
            image=image,
            image_sha256=canonical_tensor_sha256(image),
            parent_sha256=canonical_tensor_sha256(baseline),
            raw_generated=False,
            pixel_target_eligible=False,
            synthetic_mock=False,
            correction_parameters={
                "global_log_gain": global_log_gain,
                "face_log_gain": face_log_gain,
            },
        )


class L3LocalTeacher:
    def __init__(self, editor: QwenImageEditLocalAdapter):
        self.editor = editor

    def propose(self, baseline: torch.Tensor, prompt: str) -> TeacherCandidate:
        proposal = self.editor.edit(baseline, prompt)
        return TeacherCandidate(
            level="L3",
            image=proposal.image,
            image_sha256=canonical_tensor_sha256(proposal.image),
            parent_sha256=canonical_tensor_sha256(baseline),
            raw_generated=True,
            pixel_target_eligible=False,
            synthetic_mock=proposal.synthetic_mock,
            correction_parameters={},
        )
