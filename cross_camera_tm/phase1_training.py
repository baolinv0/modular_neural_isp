from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import nn

from .adapters import LUMA_WEIGHTS, PairTransformParameters, TargetCameraAdapter
from .canonicalization import DeviceCanonicalizer
from .contracts import AlignmentQuality
from .phase1 import (
    FrozenSamsungTM,
    TeacherMetricProfile,
    TeacherMetricThreshold,
    TeacherQualificationStatus,
    TeacherQualifier,
    fit_pair_parameter_predictor,
)
from .phase1_data import (
    PHASE1_FEATURE_NAMES,
    GroupFold,
    Phase1CalibrationExample,
    Phase1SourceExample,
    build_group_folds,
    extract_phase1_features,
    teacher_error_metrics,
)


@dataclass(frozen=True)
class Phase1TrainingConfig:
    solver_steps: int = 24
    solver_learning_rate: float = 0.03
    predictor_steps: int = 160
    predictor_learning_rate: float = 0.02
    hidden_dim: int = 4
    bootstrap_samples: int = 1000
    seed: int = 17
    ridge: float = 1e-3
    data_mode: str = "real"

    def __post_init__(self) -> None:
        if self.solver_steps < 1 or self.predictor_steps < 1 or self.bootstrap_samples < 50:
            raise ValueError("training step and bootstrap counts are too small")
        if self.solver_learning_rate <= 0 or self.predictor_learning_rate <= 0 or self.ridge <= 0:
            raise ValueError("learning rates and ridge must be positive")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.data_mode not in {"real", "synthetic"}:
            raise ValueError("data_mode must be real or synthetic")


@dataclass(frozen=True)
class FoldReport:
    fold_index: int
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    validation_pairs: int
    improved_pairs: int
    median_improvement: float


@dataclass(frozen=True)
class Phase1TrainingReport:
    passed: bool
    positive_folds: int
    improved_development_pairs: int
    development_bootstrap_lower: float
    locked_median_improvement: float
    qualified_development_pairs: int
    qualified_locked_pairs: int
    fold_reports: tuple[FoldReport, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Phase1TrainingResult:
    report: Phase1TrainingReport
    artifact_path: Path


@dataclass
class Phase1Artifact:
    adapter: TargetCameraAdapter
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    support_min: torch.Tensor
    support_max: torch.Tensor
    samsung_model_sha256: str
    source_manifest_sha256: str
    calibration_manifest_sha256: str
    phase1_passed: bool
    validation_report: Mapping[str, Any]
    teacher_profile: TeacherMetricProfile
    data_mode: str


@dataclass(frozen=True)
class _PreparedPair:
    example: Phase1CalibrationExample
    iphone: torch.Tensor
    samsung: torch.Tensor
    teacher: torch.Tensor
    features: torch.Tensor
    confidence: torch.Tensor
    teacher_weight: float
    teacher_status: TeacherQualificationStatus
    baseline_error: float


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _tone_error(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_luma = _luma(student).clamp(0.0, 1.0)
    teacher_luma = _luma(teacher).clamp(0.0, 1.0)
    quantiles = student.new_tensor((0.1, 0.25, 0.5, 0.75, 0.9))
    student_q = torch.quantile(torch.log(student_luma.clamp_min(1e-6)).flatten(1), quantiles, dim=1)
    teacher_q = torch.quantile(torch.log(teacher_luma.clamp_min(1e-6)).flatten(1), quantiles, dim=1)
    quantile_error = (student_q - teacher_q).abs().mean()
    student_headroom = 1.0 - torch.quantile(student_luma.flatten(1), 0.99, dim=1)
    teacher_headroom = 1.0 - torch.quantile(teacher_luma.flatten(1), 0.99, dim=1)
    headroom_error = (student_headroom - teacher_headroom).abs().mean()
    student_clip = torch.sigmoid((student_luma - 0.995) * 200.0).mean()
    teacher_clip = torch.sigmoid((teacher_luma - 0.995) * 200.0).mean()
    clipping_error = (student_clip - teacher_clip).abs()
    student_contrast = torch.quantile(student_luma.flatten(1), 0.75, dim=1) - torch.quantile(
        student_luma.flatten(1), 0.25, dim=1
    )
    teacher_contrast = torch.quantile(teacher_luma.flatten(1), 0.75, dim=1) - torch.quantile(
        teacher_luma.flatten(1), 0.25, dim=1
    )
    contrast_error = (student_contrast - teacher_contrast).abs().mean()
    student_local = (student_luma - functional.avg_pool2d(student_luma, 3, 1, 1)).abs().mean()
    teacher_local = (teacher_luma - functional.avg_pool2d(teacher_luma, 3, 1, 1)).abs().mean()
    local_error = (student_local - teacher_local).abs()
    return quantile_error + 0.5 * headroom_error + 0.5 * clipping_error + 0.25 * contrast_error + 0.1 * local_error


def observable_pair_error(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    alignment: AlignmentQuality,
    roi_mask: torch.Tensor | None,
    alignment_mask: torch.Tensor | None,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError("observable teacher/student outputs must share shape")
    total = _tone_error(student, teacher)
    student_luma, teacher_luma = _luma(student), _luma(teacher)
    if alignment in {AlignmentQuality.ROI, AlignmentQuality.LOW_FREQUENCY}:
        if roi_mask is None or roi_mask.shape != student_luma.shape:
            raise ValueError("ROI supervision requires a valid shared mask")
        mask = roi_mask.to(dtype=student.dtype)
        student_mean = (student_luma * mask).sum() / mask.sum().clamp_min(1.0)
        teacher_mean = (teacher_luma * mask).sum() / mask.sum().clamp_min(1.0)
        total = total + 0.5 * (student_mean - teacher_mean).abs()
    if alignment is AlignmentQuality.LOW_FREQUENCY:
        if alignment_mask is None or alignment_mask.shape != student_luma.shape:
            raise ValueError("low-frequency supervision requires a valid shared mask")
        pooled_student = functional.avg_pool2d(student_luma, 4, 4)
        pooled_teacher = functional.avg_pool2d(teacher_luma, 4, 4)
        pooled_mask = functional.avg_pool2d(alignment_mask.to(student.dtype), 4, 4)
        lowfreq = ((pooled_student - pooled_teacher).abs() * pooled_mask).sum() / pooled_mask.sum().clamp_min(1.0)
        total = total + 0.5 * lowfreq
    return total


def _parameter_regularization(parameters: PairTransformParameters) -> torch.Tensor:
    identity_matrix = torch.eye(3, device=parameters.matrix.device, dtype=parameters.matrix.dtype).unsqueeze(0)
    identity_curve = torch.linspace(
        0.0, 1.0, parameters.curve_y.shape[1], device=parameters.curve_y.device, dtype=parameters.curve_y.dtype
    ).unsqueeze(0)
    return (
        torch.log(parameters.gains.clamp_min(1e-6)).square().mean()
        + (parameters.matrix - identity_matrix).square().mean()
        + (parameters.curve_y - identity_curve).square().mean()
    )


class ObservablePairSolver:
    """Input-side initialization followed by observable frozen-TM output optimization."""

    def __init__(self, *, curve_points: int = 6, ridge: float = 1e-3):
        self.curve_points = int(curve_points)
        self.ridge = float(ridge)

    def initialize(
        self, iphone: torch.Tensor, samsung: torch.Tensor, reliable_mask: torch.Tensor
    ) -> PairTransformParameters:
        if iphone.shape != samsung.shape or reliable_mask.shape != iphone[:, :1].shape:
            raise ValueError("initializer requires shared image/mask shapes")
        batch = iphone.shape[0]
        gains = []
        matrices = []
        curves = []
        x_points = torch.linspace(0.0, 1.0, self.curve_points, device=iphone.device, dtype=iphone.dtype)
        luma_weights = iphone.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        for index in range(batch):
            mask = reliable_mask[index, 0].reshape(-1).to(dtype=iphone.dtype)
            x = iphone[index].permute(1, 2, 0).reshape(-1, 3)
            y = samsung[index].permute(1, 2, 0).reshape(-1, 3)
            weighted_x = x * mask[:, None]
            numerator = (weighted_x * y).sum(dim=0)
            denominator = (weighted_x * x).sum(dim=0).clamp_min(1e-8)
            gain = (numerator / denominator).clamp(0.70, 1.40)
            gained = x * gain
            gram = gained.transpose(0, 1) @ (gained * mask[:, None])
            cross = gained.transpose(0, 1) @ (y * mask[:, None])
            identity = torch.eye(3, device=iphone.device, dtype=iphone.dtype)
            matrix = torch.linalg.solve(gram + self.ridge * identity, cross + self.ridge * identity).transpose(0, 1)
            matrix = identity + (matrix - identity).clamp(-0.10, 0.10)
            transformed = (gained @ matrix.transpose(0, 1)).reshape(iphone.shape[2], iphone.shape[3], 3).permute(2, 0, 1).unsqueeze(0)
            source_luma = (transformed * luma_weights).sum(dim=1, keepdim=True)[0, 0]
            target_luma = (samsung[index : index + 1] * luma_weights).sum(dim=1, keepdim=True)[0, 0]
            valid = reliable_mask[index, 0].bool()
            sx = source_luma[valid]
            residual = target_luma[valid] - sx
            if sx.numel() == 0:
                curve = x_points.clone()
            else:
                bandwidth = 1.0 / max(2, self.curve_points - 1)
                kernel = torch.exp(-0.5 * ((x_points[:, None] - sx[None, :]) / bandwidth).square())
                residual_points = (kernel * residual[None, :]).sum(dim=1) / kernel.sum(dim=1).clamp_min(1e-8)
                curve = (x_points + residual_points).clamp(0.0, 1.0)
                curve[0], curve[-1] = 0.0, 1.0
                curve = torch.cummax(curve, dim=0).values
                curve[-1] = 1.0
            gains.append(gain)
            matrices.append(matrix)
            curves.append(curve)
        return PairTransformParameters(torch.stack(gains), torch.stack(matrices), torch.stack(curves))

    def refine(
        self,
        *,
        initial: PairTransformParameters,
        iphone: torch.Tensor,
        teacher: torch.Tensor,
        frozen_tm: FrozenSamsungTM,
        alignment: AlignmentQuality,
        roi_mask: torch.Tensor | None,
        alignment_mask: torch.Tensor | None,
        teacher_weight: float,
        steps: int,
        learning_rate: float,
    ) -> PairTransformParameters:
        if not 0.0 < teacher_weight <= 1.0:
            raise ValueError("teacher_weight must lie in (0,1]")
        log_gains = nn.Parameter(torch.log(initial.gains.clamp_min(1e-6)).clone())
        identity = torch.eye(3, device=iphone.device, dtype=iphone.dtype).unsqueeze(0)
        matrix_delta = nn.Parameter((initial.matrix - identity).clone())
        curve_inner = nn.Parameter(initial.curve_y[:, 1:-1].clone())
        optimizer = torch.optim.Adam((log_gains, matrix_delta, curve_inner), lr=learning_rate)
        final = initial
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            gains = torch.exp(log_gains.clamp(math.log(0.70), math.log(1.40)))
            matrix = identity + matrix_delta.clamp(-0.10, 0.10)
            inner = torch.cummax(curve_inner.clamp(0.0, 1.0), dim=1).values
            curve = torch.cat((torch.zeros_like(inner[:, :1]), inner, torch.ones_like(inner[:, :1])), dim=1)
            parameters = PairTransformParameters(gains, matrix, curve)
            adapted = TargetCameraAdapter.apply_explicit(iphone, parameters, confidence=torch.ones(iphone.shape[0]))
            student = frozen_tm(adapted)
            loss = teacher_weight * observable_pair_error(
                student,
                teacher,
                alignment=alignment,
                roi_mask=roi_mask,
                alignment_mask=alignment_mask,
            ) + 0.01 * _parameter_regularization(parameters)
            loss.backward()
            optimizer.step()
            final = PairTransformParameters(gains.detach(), matrix.detach(), curve.detach())
        return final


def _build_teacher_profile(
    source_examples: Sequence[Phase1SourceExample],
    canonicalizer: DeviceCanonicalizer,
    frozen_tm: FrozenSamsungTM,
) -> TeacherMetricProfile:
    errors = []
    for example in source_examples:
        samsung = canonicalizer.canonicalize(example.samsung_image, example.metadata)
        output = frozen_tm(samsung.image)
        metrics, _ = teacher_error_metrics(output, example.samsung_gt)
        errors.append(metrics)
    return TeacherMetricProfile.from_source_errors(errors)


def _prepare_pairs(
    examples: Sequence[Phase1CalibrationExample],
    canonicalizer: DeviceCanonicalizer,
    frozen_tm: FrozenSamsungTM,
    qualifier: TeacherQualifier,
) -> tuple[_PreparedPair, ...]:
    prepared = []
    with torch.no_grad():
        for example in examples:
            iphone = canonicalizer.canonicalize(example.iphone_image, example.iphone_metadata)
            samsung = canonicalizer.canonicalize(example.samsung_image, example.samsung_metadata)
            teacher = frozen_tm(samsung.image)
            metrics, hard_defect = teacher_error_metrics(teacher, example.samsung_gt)
            qualification = qualifier.qualify(metrics, hard_defect)
            baseline = frozen_tm(iphone.image)
            baseline_error = float(
                observable_pair_error(
                    baseline,
                    teacher,
                    alignment=example.alignment.quality,
                    roi_mask=example.roi_mask,
                    alignment_mask=example.alignment_mask,
                ).item()
            )
            prepared.append(
                _PreparedPair(
                    example=example,
                    iphone=iphone.image,
                    samsung=samsung.image,
                    teacher=teacher.detach(),
                    features=extract_phase1_features(iphone, example.iphone_metadata),
                    confidence=torch.tensor([iphone.confidence.overall], dtype=iphone.image.dtype),
                    teacher_weight=qualification.weight,
                    teacher_status=qualification.status,
                    baseline_error=baseline_error,
                )
            )
    return tuple(prepared)


def _fit_pair_targets(
    prepared: Sequence[_PreparedPair],
    solver: ObservablePairSolver,
    frozen_tm: FrozenSamsungTM,
    config: Phase1TrainingConfig,
) -> dict[str, PairTransformParameters]:
    result = {}
    for item in prepared:
        if item.teacher_status is TeacherQualificationStatus.REJECTED:
            continue
        reliable = torch.ones_like(item.iphone[:, :1], dtype=torch.bool)
        if item.example.alignment_mask is not None:
            reliable = reliable & item.example.alignment_mask.bool()
        initial = solver.initialize(item.iphone, item.samsung, reliable)
        result[item.example.pair_id] = solver.refine(
            initial=initial,
            iphone=item.iphone,
            teacher=item.teacher,
            frozen_tm=frozen_tm,
            alignment=item.example.alignment.quality,
            roi_mask=item.example.roi_mask,
            alignment_mask=item.example.alignment_mask,
            teacher_weight=item.teacher_weight,
            steps=config.solver_steps,
            learning_rate=config.solver_learning_rate,
        )
    return result


def _stack_targets(items: Sequence[_PreparedPair], targets: Mapping[str, PairTransformParameters]) -> PairTransformParameters:
    chosen = [targets[item.example.pair_id] for item in items]
    return PairTransformParameters(
        gains=torch.cat([item.gains for item in chosen], dim=0),
        matrix=torch.cat([item.matrix for item in chosen], dim=0),
        curve_y=torch.cat([item.curve_y for item in chosen], dim=0),
    )


def _normalization(items: Sequence[_PreparedPair]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.cat([item.features for item in items], dim=0)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    return features, mean, std


def _train_predictor(
    items: Sequence[_PreparedPair],
    targets: Mapping[str, PairTransformParameters],
    config: Phase1TrainingConfig,
) -> tuple[TargetCameraAdapter, torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("no qualified Phase 1 training pairs")
    features, mean, std = _normalization(items)
    normalized = (features - mean) / std
    confidence = torch.cat([item.confidence for item in items], dim=0)
    adapter = TargetCameraAdapter(len(PHASE1_FEATURE_NAMES), config.hidden_dim)
    fit_pair_parameter_predictor(
        adapter,
        normalized,
        _stack_targets(items, targets),
        confidence=confidence,
        steps=config.predictor_steps,
        learning_rate=config.predictor_learning_rate,
    )
    return adapter, mean, std


def _parameter_margin(parameters: PairTransformParameters) -> float:
    gain_use = torch.log(parameters.gains).abs().max().item() / 0.35
    identity = torch.eye(3, device=parameters.matrix.device, dtype=parameters.matrix.dtype).unsqueeze(0)
    matrix_use = (parameters.matrix - identity).abs().max().item() / 0.10
    return float(1.0 - max(gain_use, matrix_use))


def _evaluate_items(
    items: Sequence[_PreparedPair],
    adapter: TargetCameraAdapter,
    mean: torch.Tensor,
    std: torch.Tensor,
    frozen_tm: FrozenSamsungTM,
) -> tuple[list[float], list[float]]:
    improvements = []
    margins = []
    with torch.no_grad():
        for item in items:
            if item.teacher_status is TeacherQualificationStatus.REJECTED:
                improvements.append(0.0)
                margins.append(0.0)
                continue
            features = (item.features - mean) / std
            output = adapter(item.iphone, features, confidence=item.confidence)
            student = frozen_tm(output.image)
            adapted_error = float(
                observable_pair_error(
                    student,
                    item.teacher,
                    alignment=item.example.alignment.quality,
                    roi_mask=item.example.roi_mask,
                    alignment_mask=item.example.alignment_mask,
                ).item()
            )
            improvements.append(item.baseline_error - adapted_error)
            margins.append(_parameter_margin(PairTransformParameters(output.gains, output.matrix, output.curve_y)))
    return improvements, margins


def _bootstrap_lower(values: Sequence[float], samples: int, seed: int) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return float("-inf")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, tensor.numel(), (samples, tensor.numel()), generator=generator)
    medians = torch.median(tensor[indices], dim=1).values
    return float(torch.quantile(medians, 0.05).item())


def _profile_payload(profile: TeacherMetricProfile) -> dict[str, Any]:
    return {
        "source_count": profile.source_count,
        "thresholds": {
            key: {"p75": threshold.p75, "p90": threshold.p90}
            for key, threshold in profile.thresholds.items()
        },
    }


def _profile_from_payload(payload: Mapping[str, Any]) -> TeacherMetricProfile:
    return TeacherMetricProfile(
        thresholds={
            str(key): TeacherMetricThreshold(p75=float(value["p75"]), p90=float(value["p90"]))
            for key, value in payload["thresholds"].items()
        },
        source_count=int(payload["source_count"]),
    )


def _report_payload(report: Phase1TrainingReport) -> dict[str, Any]:
    return {
        **{key: value for key, value in asdict(report).items() if key != "fold_reports"},
        "fold_reports": [asdict(item) for item in report.fold_reports],
    }


def train_phase1(
    *,
    source_examples: Sequence[Phase1SourceExample],
    calibration_examples: Sequence[Phase1CalibrationExample],
    frozen_tm: FrozenSamsungTM,
    samsung_model_sha256: str,
    source_manifest_sha256: str,
    calibration_manifest_sha256: str,
    artifact_path: Path | str,
    config: Phase1TrainingConfig | None = None,
    canonicalizer: DeviceCanonicalizer | None = None,
) -> Phase1TrainingResult:
    config = config or Phase1TrainingConfig()
    canonicalizer = canonicalizer or DeviceCanonicalizer()
    if len(calibration_examples) != 50:
        raise ValueError("Phase 1 requires exactly 50 calibration pairs")
    development = [item for item in calibration_examples if item.split == "development"]
    locked = [item for item in calibration_examples if item.split == "locked"]
    if len(development) != 40 or len(locked) != 10:
        raise ValueError("Phase 1 requires a frozen 40 development / 10 locked split")
    development_groups = {item.scene_group for item in development}
    locked_groups = {item.scene_group for item in locked}
    if development_groups & locked_groups:
        raise ValueError("locked scene groups must not appear in development")
    torch.manual_seed(config.seed)
    profile = _build_teacher_profile(source_examples, canonicalizer, frozen_tm)
    prepared = _prepare_pairs(calibration_examples, canonicalizer, frozen_tm, TeacherQualifier(profile))
    prepared_development = [item for item in prepared if item.example.split == "development"]
    prepared_locked = [item for item in prepared if item.example.split == "locked"]
    solver = ObservablePairSolver(ridge=config.ridge)
    targets = _fit_pair_targets(prepared_development, solver, frozen_tm, config)
    fold_reports = []
    out_of_fold: dict[str, float] = {}
    folds: tuple[GroupFold, ...] = build_group_folds(calibration_examples, folds=5)
    for fold in folds:
        train_items = [
            item
            for item in prepared_development
            if item.example.scene_group in fold.train_groups and item.example.pair_id in targets
        ]
        validation_items = [
            item for item in prepared_development if item.example.scene_group in fold.validation_groups
        ]
        adapter, mean, std = _train_predictor(train_items, targets, config)
        improvements, _ = _evaluate_items(validation_items, adapter, mean, std, frozen_tm)
        for item, improvement in zip(validation_items, improvements):
            out_of_fold[item.example.pair_id] = improvement
        median = float(torch.median(torch.tensor(improvements)).item()) if improvements else float("-inf")
        fold_reports.append(
            FoldReport(
                fold_index=fold.fold_index,
                train_groups=fold.train_groups,
                validation_groups=fold.validation_groups,
                validation_pairs=len(validation_items),
                improved_pairs=sum(value > 0.0 for value in improvements),
                median_improvement=median,
            )
        )
    development_improvements = [out_of_fold.get(item.example.pair_id, 0.0) for item in prepared_development]
    qualified_development = [item for item in prepared_development if item.example.pair_id in targets]
    final_adapter, feature_mean, feature_std = _train_predictor(qualified_development, targets, config)
    locked_improvements, locked_margins = _evaluate_items(
        prepared_locked, final_adapter, feature_mean, feature_std, frozen_tm
    )
    normalized_development = torch.cat(
        [(item.features - feature_mean) / feature_std for item in qualified_development], dim=0
    )
    support_min = normalized_development.min(dim=0).values
    support_max = normalized_development.max(dim=0).values
    positive_folds = sum(item.median_improvement > 0.0 for item in fold_reports)
    improved_development = sum(value > 0.0 for value in development_improvements)
    bootstrap_lower = _bootstrap_lower(
        development_improvements, config.bootstrap_samples, config.seed
    )
    locked_median = float(torch.median(torch.tensor(locked_improvements)).item())
    reasons = []
    if positive_folds < 4:
        reasons.append("fold_consistency")
    if improved_development < 30:
        reasons.append("development_prevalence")
    if bootstrap_lower <= 0.0:
        reasons.append("development_bootstrap")
    if locked_median <= 0.0:
        reasons.append("locked_direction")
    if any(margin <= 0.0 for margin in locked_margins):
        reasons.append("adapter_boundary")
    qualified_locked = sum(
        item.teacher_status is not TeacherQualificationStatus.REJECTED for item in prepared_locked
    )
    if len(qualified_development) < 30 or qualified_locked < 8:
        reasons.append("teacher_qualification_coverage")
    report = Phase1TrainingReport(
        passed=not reasons,
        positive_folds=positive_folds,
        improved_development_pairs=improved_development,
        development_bootstrap_lower=bootstrap_lower,
        locked_median_improvement=locked_median,
        qualified_development_pairs=len(qualified_development),
        qualified_locked_pairs=qualified_locked,
        fold_reports=tuple(fold_reports),
        reasons=tuple(reasons),
    )
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "feature_names": list(PHASE1_FEATURE_NAMES),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "support_min": support_min,
        "support_max": support_max,
        "adapter": {
            "feature_dim": len(PHASE1_FEATURE_NAMES),
            "hidden_dim": config.hidden_dim,
            "curve_points": final_adapter.curve_points,
            "max_log_gain": final_adapter.max_log_gain,
            "max_matrix_delta": final_adapter.max_matrix_delta,
            "state_dict": final_adapter.state_dict(),
        },
        "samsung_model_sha256": samsung_model_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "phase1_passed": report.passed,
        "training_config": asdict(config),
        "validation_report": _report_payload(report),
        "teacher_profile": _profile_payload(profile),
        "data_mode": config.data_mode,
    }
    torch.save(payload, artifact_path)
    return Phase1TrainingResult(report=report, artifact_path=artifact_path)


def load_phase1_artifact(
    path: Path | str, *, expected_model_sha256: str | None = None
) -> Phase1Artifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = {
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
    }
    if not isinstance(payload, Mapping) or set(payload) != expected or int(payload["schema_version"]) != 1:
        raise ValueError("Phase 1 artifact schema is invalid")
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
    return Phase1Artifact(
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
        teacher_profile=_profile_from_payload(payload["teacher_profile"]),
        data_mode=str(payload["data_mode"]),
    )


def calibration_support_distance(features: torch.Tensor, artifact: Phase1Artifact) -> float:
    normalized = (features - artifact.feature_mean) / artifact.feature_std
    below = (artifact.support_min - normalized).clamp_min(0.0)
    above = (normalized - artifact.support_max).clamp_min(0.0)
    return float(torch.sqrt((below.square() + above.square()).mean()).item())


def run_phase1_inference(
    *,
    image: torch.Tensor,
    metadata,
    frozen_tm: FrozenSamsungTM,
    artifact: Phase1Artifact,
    canonicalizer: DeviceCanonicalizer | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not artifact.phase1_passed:
        raise ValueError("Phase 1 artifact did not pass locked-holdout acceptance")
    canonicalizer = canonicalizer or DeviceCanonicalizer()
    canonical = canonicalizer.canonicalize(image, metadata)
    features = extract_phase1_features(canonical, metadata)
    support_distance = calibration_support_distance(features, artifact)
    normalized = (features - artifact.feature_mean) / artifact.feature_std
    confidence = torch.tensor([canonical.confidence.overall], dtype=canonical.image.dtype)
    with torch.no_grad():
        adapted = artifact.adapter(canonical.image, normalized, confidence=confidence)
        output = frozen_tm(adapted.image)
    manifest = {
        "schema_version": 1,
        "phase1_status": "pass",
        "phase2_executed": False,
        "samsung_model_sha256": artifact.samsung_model_sha256,
        "source_manifest_sha256": artifact.source_manifest_sha256,
        "calibration_manifest_sha256": artifact.calibration_manifest_sha256,
        "input_sample_id": metadata.sample_id,
        "canonicalization_confidence": canonical.confidence.overall,
        "calibration_support_distance": support_distance,
        "adapter_parameter_bound_margin": _parameter_margin(
            PairTransformParameters(adapted.gains, adapted.matrix, adapted.curve_y)
        ),
        "real_data_effectiveness_verified": artifact.data_mode == "real" and artifact.phase1_passed,
    }
    return output, manifest


def evaluate_phase1_artifact(
    *,
    calibration_examples: Sequence[Phase1CalibrationExample],
    frozen_tm: FrozenSamsungTM,
    artifact: Phase1Artifact,
    canonicalizer: DeviceCanonicalizer | None = None,
) -> dict[str, Any]:
    canonicalizer = canonicalizer or DeviceCanonicalizer()
    qualifier = TeacherQualifier(artifact.teacher_profile)
    prepared = _prepare_pairs(calibration_examples, canonicalizer, frozen_tm, qualifier)
    locked = [item for item in prepared if item.example.split == "locked"]
    improvements, margins = _evaluate_items(
        locked, artifact.adapter, artifact.feature_mean, artifact.feature_std, frozen_tm
    )
    return {
        "locked_pairs": len(locked),
        "improved_locked_pairs": sum(value > 0.0 for value in improvements),
        "locked_median_improvement": float(torch.median(torch.tensor(improvements)).item()),
        "minimum_parameter_bound_margin": min(margins) if margins else float("-inf"),
        "phase1_artifact_passed": artifact.phase1_passed,
    }
