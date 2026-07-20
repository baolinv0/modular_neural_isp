from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch
from torch import nn

from .adapters import LUMA_WEIGHTS, PairTransformParameters, TargetCameraAdapter
from .losses import AlignmentQuality, DistillationLoss


class FrozenSamsungTM(nn.Module):
    """A gradient-transparent wrapper that permanently freezes Samsung TM weights."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenSamsungTM":
        super().train(False)
        self.module.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        result = self.module(image)
        if torch.is_tensor(result):
            return result
        if isinstance(result, Mapping):
            for key in ("output", "image", "tone_mapped", "prediction"):
                value = result.get(key)
                if torch.is_tensor(value):
                    return value
        if hasattr(result, "output") and torch.is_tensor(result.output):
            return result.output
        raise TypeError("Samsung TM must return a tensor or a mapping/object containing one")


class TeacherQualificationStatus(str, Enum):
    QUALIFIED = "qualified"
    DOWNWEIGHTED = "downweighted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TeacherMetricThreshold:
    p75: float
    p90: float


@dataclass(frozen=True)
class TeacherMetricProfile:
    thresholds: Mapping[str, TeacherMetricThreshold]
    source_count: int

    @classmethod
    def from_source_errors(cls, errors: Sequence[Mapping[str, float]]) -> "TeacherMetricProfile":
        if len(errors) < 4:
            raise ValueError("teacher profile requires at least four source samples")
        keys = tuple(sorted(errors[0]))
        if not keys or any(tuple(sorted(row)) != keys for row in errors):
            raise ValueError("source teacher metrics must have identical non-empty keys")
        thresholds: dict[str, TeacherMetricThreshold] = {}
        for key in keys:
            values = torch.tensor([float(row[key]) for row in errors], dtype=torch.float64)
            if not torch.isfinite(values).all():
                raise ValueError("teacher profile metrics must be finite")
            thresholds[key] = TeacherMetricThreshold(
                p75=float(torch.quantile(values, 0.75).item()),
                p90=float(torch.quantile(values, 0.90).item()),
            )
        return cls(thresholds=thresholds, source_count=len(errors))


@dataclass(frozen=True)
class TeacherQualification:
    status: TeacherQualificationStatus
    weight: float
    failed_metrics: tuple[str, ...]


class TeacherQualifier:
    def __init__(self, profile: TeacherMetricProfile):
        self.profile = profile

    def qualify(self, metrics: Mapping[str, float], hard_defect: bool) -> TeacherQualification:
        if tuple(sorted(metrics)) != tuple(sorted(self.profile.thresholds)):
            raise ValueError("qualification metrics do not match the source profile")
        if hard_defect:
            return TeacherQualification(TeacherQualificationStatus.REJECTED, 0.0, ("hard_defect",))
        failed_p90 = tuple(
            key for key, value in metrics.items() if float(value) > self.profile.thresholds[key].p90
        )
        if failed_p90:
            return TeacherQualification(TeacherQualificationStatus.REJECTED, 0.0, failed_p90)
        above_p75 = tuple(
            key for key, value in metrics.items() if float(value) > self.profile.thresholds[key].p75
        )
        if above_p75:
            return TeacherQualification(TeacherQualificationStatus.DOWNWEIGHTED, 0.5, above_p75)
        return TeacherQualification(TeacherQualificationStatus.QUALIFIED, 1.0, ())


@dataclass(frozen=True)
class PairRefinementResult:
    parameters: PairTransformParameters
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]


class PairParameterSolver:
    def __init__(self, *, curve_points: int = 6):
        if curve_points < 3:
            raise ValueError("curve_points must be at least three")
        self.curve_points = curve_points

    def initialize(
        self,
        iphone: torch.Tensor,
        samsung: torch.Tensor,
        reliable_mask: torch.Tensor,
    ) -> PairTransformParameters:
        if iphone.shape != samsung.shape or iphone.ndim != 4 or iphone.shape[1] != 3:
            raise ValueError("paired images must share shape [B,3,H,W]")
        if reliable_mask.shape != (iphone.shape[0], 1, iphone.shape[2], iphone.shape[3]):
            raise ValueError("reliable_mask must have shape [B,1,H,W]")
        weights = reliable_mask.to(dtype=iphone.dtype).expand_as(iphone)
        numerator = (weights * iphone * samsung).sum(dim=(2, 3))
        denominator = (weights * iphone.square()).sum(dim=(2, 3)).clamp_min(1e-8)
        gains = (numerator / denominator).clamp(0.25, 4.0)
        corrected = iphone * gains[:, :, None, None]
        matrix = torch.eye(3, device=iphone.device, dtype=iphone.dtype).unsqueeze(0).expand(
            iphone.shape[0], -1, -1
        ).clone()

        luma_weights = iphone.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        source_luma = (corrected * luma_weights).sum(dim=1, keepdim=True)
        target_luma = (samsung * luma_weights).sum(dim=1, keepdim=True)
        x_points = torch.linspace(0.0, 1.0, self.curve_points, device=iphone.device, dtype=iphone.dtype)
        curves: list[torch.Tensor] = []
        for batch_index in range(iphone.shape[0]):
            valid = reliable_mask[batch_index].bool()
            x = source_luma[batch_index][valid]
            residual = target_luma[batch_index][valid] - x
            if x.numel() == 0:
                curve = x_points.clone()
            else:
                bandwidth = 1.0 / max(2, self.curve_points - 1)
                kernel = torch.exp(-0.5 * ((x_points[:, None] - x[None, :]) / bandwidth).square())
                residual_points = (kernel * residual[None, :]).sum(dim=1) / kernel.sum(dim=1).clamp_min(1e-8)
                curve = (x_points + residual_points).clamp(0.0, 1.0)
                curve[0], curve[-1] = 0.0, 1.0
                curve = torch.cummax(curve, dim=0).values
                curve[-1] = 1.0
            curves.append(curve)
        return PairTransformParameters(gains=gains, matrix=matrix, curve_y=torch.stack(curves))

    def refine(
        self,
        initial: PairTransformParameters,
        iphone: torch.Tensor,
        teacher: torch.Tensor,
        frozen_tm: FrozenSamsungTM,
        loss_fn: DistillationLoss,
        *,
        steps: int,
        learning_rate: float,
    ) -> PairRefinementResult:
        if steps < 1 or learning_rate <= 0:
            raise ValueError("refinement requires positive steps and learning rate")
        log_gains = nn.Parameter(torch.log(initial.gains.clamp_min(1e-5)).detach().clone())
        identity = torch.eye(3, device=iphone.device, dtype=iphone.dtype).unsqueeze(0)
        matrix_delta = nn.Parameter((initial.matrix - identity).detach().clone())
        curve_inner = nn.Parameter(initial.curve_y[:, 1:-1].detach().clone())
        optimizer = torch.optim.Adam((log_gains, matrix_delta, curve_inner), lr=learning_rate)
        history: list[float] = []
        final_parameters = initial

        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            inner = torch.cummax(curve_inner.clamp(0.0, 1.0), dim=1).values
            curve = torch.cat(
                (
                    torch.zeros_like(inner[:, :1]),
                    inner,
                    torch.ones_like(inner[:, :1]),
                ),
                dim=1,
            )
            parameters = PairTransformParameters(
                gains=torch.exp(log_gains),
                matrix=identity + matrix_delta,
                curve_y=curve,
            )
            adapted = TargetCameraAdapter.apply_explicit(
                iphone, parameters, confidence=torch.ones(iphone.shape[0], device=iphone.device)
            )
            student = frozen_tm(adapted)
            result = loss_fn(
                student,
                teacher,
                alignment=AlignmentQuality.SCENE_ONLY,
                parameters=parameters,
            )
            history.append(float(result.total.detach().item()))
            final_parameters = PairTransformParameters(
                gains=parameters.gains.detach().clone(),
                matrix=parameters.matrix.detach().clone(),
                curve_y=parameters.curve_y.detach().clone(),
            )
            if step == steps:
                break
            result.total.backward()
            optimizer.step()

        return PairRefinementResult(
            parameters=final_parameters,
            initial_loss=history[0],
            final_loss=history[-1],
            loss_history=tuple(history),
        )


@dataclass(frozen=True)
class GradientCanaryReport:
    output_finite: bool
    input_gradient_finite: bool
    input_gradient_nonzero: bool
    trainable_backbone_parameters: int


def gradient_canary(frozen_tm: FrozenSamsungTM, image: torch.Tensor) -> GradientCanaryReport:
    probe = image.detach().clone().requires_grad_(True)
    output = frozen_tm(probe)
    output.mean().backward()
    gradient = probe.grad
    return GradientCanaryReport(
        output_finite=bool(torch.isfinite(output).all().item()),
        input_gradient_finite=bool(gradient is not None and torch.isfinite(gradient).all().item()),
        input_gradient_nonzero=bool(gradient is not None and gradient.abs().sum().item() > 0.0),
        trainable_backbone_parameters=sum(
            parameter.numel() for parameter in frozen_tm.module.parameters() if parameter.requires_grad
        ),
    )


@dataclass(frozen=True)
class PairPredictorFitResult:
    initial_loss: float
    final_loss: float
    steps: int


def fit_pair_parameter_predictor(
    adapter: TargetCameraAdapter,
    features: torch.Tensor,
    targets: PairTransformParameters,
    *,
    confidence: torch.Tensor,
    steps: int = 200,
    learning_rate: float = 0.02,
) -> PairPredictorFitResult:
    if features.shape[0] != targets.gains.shape[0] or confidence.shape[0] != features.shape[0]:
        raise ValueError("pair predictor inputs must share a batch dimension")
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    history: list[float] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted = adapter.predict_parameters(features, confidence=confidence)
        loss = (
            (predicted.gains - targets.gains).square().mean()
            + (predicted.matrix - targets.matrix).square().mean()
            + (predicted.curve_y - targets.curve_y).square().mean()
        )
        history.append(float(loss.detach().item()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return PairPredictorFitResult(history[0], history[-1], steps)


@dataclass(frozen=True)
class Phase1ValidationSample:
    pair_id: str
    scene_group: str
    split: str
    baseline_error: float
    adapted_error: float
    parameter_bound_margin: float


@dataclass(frozen=True)
class Phase1ValidationResult:
    passed: bool
    positive_folds: int
    improved_development_pairs: int
    development_bootstrap_lower: float
    locked_median_improvement: float
    reasons: tuple[str, ...]


def assess_phase1_validation(
    samples: Sequence[Phase1ValidationSample],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> Phase1ValidationResult:
    development = [sample for sample in samples if sample.split == "development"]
    locked = [sample for sample in samples if sample.split == "locked"]
    reasons: list[str] = []
    if len(development) != 40 or len(locked) != 10:
        reasons.append("split_counts")
    if any(sample.parameter_bound_margin <= 0.0 for sample in samples):
        reasons.append("parameter_boundary")
    groups = sorted({sample.scene_group for sample in development})
    fold_values: list[list[float]] = [[] for _ in range(5)]
    for index, group in enumerate(groups):
        for sample in development:
            if sample.scene_group == group:
                fold_values[index % 5].append(sample.baseline_error - sample.adapted_error)
    positive_folds = sum(
        bool(values) and torch.median(torch.tensor(values)).item() > 0.0 for values in fold_values
    )
    if positive_folds < 4:
        reasons.append("fold_consistency")
    dev_improvements = torch.tensor(
        [sample.baseline_error - sample.adapted_error for sample in development], dtype=torch.float64
    )
    improved_pairs = int((dev_improvements > 0.0).sum().item()) if dev_improvements.numel() else 0
    if improved_pairs < 30:
        reasons.append("pair_prevalence")
    if dev_improvements.numel():
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randint(
            0,
            dev_improvements.numel(),
            (bootstrap_samples, dev_improvements.numel()),
            generator=generator,
        )
        medians = torch.median(dev_improvements[indices], dim=1).values
        lower = float(torch.quantile(medians, 0.05).item())
    else:
        lower = float("-inf")
    if lower <= 0.0:
        reasons.append("development_bootstrap")
    locked_values = torch.tensor(
        [sample.baseline_error - sample.adapted_error for sample in locked], dtype=torch.float64
    )
    locked_median = float(torch.median(locked_values).item()) if locked_values.numel() else float("-inf")
    if locked_median <= 0.0:
        reasons.append("locked_direction")
    unique = tuple(dict.fromkeys(reasons))
    return Phase1ValidationResult(
        passed=not unique,
        positive_folds=positive_folds,
        improved_development_pairs=improved_pairs,
        development_bootstrap_lower=lower,
        locked_median_improvement=locked_median,
        reasons=unique,
    )
