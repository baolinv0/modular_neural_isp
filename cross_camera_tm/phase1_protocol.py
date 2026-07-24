from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import phase1_training as core
from .adapters import LUMA_WEIGHTS, PairTransformParameters, TargetCameraAdapter
from .canonicalization import DeviceCanonicalizer
from .phase1 import FrozenSamsungTM, TeacherQualificationStatus, TeacherQualifier
from .phase1_data import (
    PHASE1_FEATURE_NAMES,
    GroupFold,
    Phase1CalibrationExample,
    Phase1SourceExample,
    build_group_folds,
)
from .phase1_training import (
    FoldReport,
    Phase1Artifact,
    Phase1TrainingReport,
    Phase1TrainingResult,
    calibration_support_distance,
    evaluate_phase1_artifact,
    load_phase1_artifact,
    run_phase1_inference,
)


@dataclass(frozen=True)
class Phase1TrainingConfig:
    solver_steps: int = 24
    solver_learning_rate: float = 0.03
    hidden_dim: int = 4
    bootstrap_samples: int = 1000
    seed: int = 17
    ridge: float = 1e-3
    data_mode: str = "real"

    def __post_init__(self) -> None:
        if self.solver_steps < 1 or self.bootstrap_samples < 50:
            raise ValueError("solver and bootstrap counts are too small")
        if self.solver_learning_rate <= 0 or self.ridge <= 0:
            raise ValueError("solver learning rate and ridge must be positive")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.data_mode not in {"real", "synthetic"}:
            raise ValueError("data_mode must be real or synthetic")


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _non_finite_error(name: str) -> ValueError:
    return ValueError(f"NON_FINITE_PHASE1_EVIDENCE: {name}")


def _require_finite_values(values: Sequence[float], name: str, *, allow_empty: bool = False) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(float(value)) for value in values):
        raise _non_finite_error(name)


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise _non_finite_error(name)


def _require_finite_parameter_targets(
    targets: Mapping[str, PairTransformParameters],
    name: str,
) -> None:
    for pair_id, parameters in targets.items():
        for field_name, tensor in (
            ("gains", parameters.gains),
            ("matrix", parameters.matrix),
            ("curve_y", parameters.curve_y),
        ):
            if not torch.isfinite(tensor).all():
                raise _non_finite_error(f"{name}:{pair_id}:{field_name}")


def _require_finite_prepared(items: Sequence[Any]) -> None:
    for item in items:
        if not math.isfinite(float(item.baseline_error)):
            raise _non_finite_error(f"prepared:{item.example.pair_id}:baseline_error")
        for field_name, tensor in (
            ("iphone", item.iphone),
            ("samsung", item.samsung),
            ("teacher", item.teacher),
            ("features", item.features),
            ("confidence", item.confidence),
        ):
            _require_finite_tensor(tensor, f"prepared:{item.example.pair_id}:{field_name}")


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True).clamp(0.0, 1.0)


def _metric_errors(
    output: torch.Tensor,
    teacher: torch.Tensor,
    roi_mask: torch.Tensor | None,
) -> dict[str, float | None]:
    output_luma = _luma(output)
    teacher_luma = _luma(teacher)
    quantiles = output.new_tensor((0.1, 0.25, 0.5, 0.75, 0.9))
    output_q = torch.quantile(torch.log(output_luma.clamp_min(1e-6)).flatten(), quantiles)
    teacher_q = torch.quantile(torch.log(teacher_luma.clamp_min(1e-6)).flatten(), quantiles)
    global_error = float((output_q - teacher_q).abs().mean().item())
    output_headroom = 1.0 - torch.quantile(output_luma.flatten(), 0.99)
    teacher_headroom = 1.0 - torch.quantile(teacher_luma.flatten(), 0.99)
    output_clip = (output_luma >= 0.995).float().mean()
    teacher_clip = (teacher_luma >= 0.995).float().mean()
    highlight_error = float(
        ((output_headroom - teacher_headroom).abs() + (output_clip - teacher_clip).abs()).item()
    )
    roi_error: float | None = None
    if roi_mask is not None:
        mask = roi_mask.to(dtype=output.dtype)
        if mask.sum().item() > 0:
            output_values = output_luma[mask.bool()]
            teacher_values = teacher_luma[mask.bool()]
            roi_q = output.new_tensor((0.25, 0.5, 0.75))
            roi_error = float(
                (
                    torch.quantile(torch.log(output_values.clamp_min(1e-6)), roi_q)
                    - torch.quantile(torch.log(teacher_values.clamp_min(1e-6)), roi_q)
                )
                .abs()
                .mean()
                .item()
            )
    return {"global_tone": global_error, "highlight": highlight_error, "roi": roi_error}


def _raw_parameter_targets(
    adapter: TargetCameraAdapter,
    targets: PairTransformParameters,
    confidence: torch.Tensor,
) -> torch.Tensor:
    gate = confidence.reshape(-1, 1).clamp(0.2, 1.0)
    gain_ratio = torch.log(targets.gains.clamp_min(1e-6)) / (adapter.max_log_gain * gate)
    raw_gain = torch.atanh(gain_ratio.clamp(-0.999, 0.999))

    identity = torch.eye(3, device=targets.matrix.device, dtype=targets.matrix.dtype).unsqueeze(0)
    matrix_ratio = (targets.matrix - identity).reshape(-1, 9) / (
        adapter.max_matrix_delta * gate
    )
    raw_matrix = torch.atanh(matrix_ratio.clamp(-0.999, 0.999))

    increments = torch.diff(targets.curve_y, dim=1).clamp_min(1e-5)
    increments = increments / increments.sum(dim=1, keepdim=True)
    raw_curve_gated = torch.log(torch.expm1(increments).clamp_min(1e-8))
    raw_curve = raw_curve_gated / gate
    result = torch.cat((raw_gain, raw_matrix, raw_curve), dim=1)
    _require_finite_tensor(result, "raw_parameter_targets")
    return result


def _canonicalize_component_signs(components: torch.Tensor) -> torch.Tensor:
    result = components.clone()
    for index in range(result.shape[0]):
        pivot = int(result[index].abs().argmax().item())
        if result[index, pivot].item() < 0.0:
            result[index].mul_(-1.0)
    return result


def _fit_ridge_predictor(
    items,
    targets,
    config: Phase1TrainingConfig,
) -> tuple[TargetCameraAdapter, torch.Tensor, torch.Tensor]:
    """Fit z->theta with fold-local PCA features and teacher-weighted ridge."""

    if not items:
        raise ValueError("no qualified Phase 1 training pairs")
    features, mean, std = core._normalization(items)
    _require_finite_tensor(features, "predictor_features")
    _require_finite_tensor(mean, "predictor_feature_mean")
    _require_finite_tensor(std, "predictor_feature_std")
    normalized = (features - mean) / std
    _require_finite_tensor(normalized, "predictor_normalized_features")
    confidence = torch.cat([item.confidence for item in items], dim=0)
    teacher_weights = torch.tensor(
        [item.teacher_weight for item in items],
        dtype=normalized.dtype,
        device=normalized.device,
    ).clamp_min(1e-3)
    parameter_targets = core._stack_targets(items, targets)
    _require_finite_parameter_targets({"stacked": parameter_targets}, "predictor_targets")
    adapter = TargetCameraAdapter(
        len(PHASE1_FEATURE_NAMES),
        config.hidden_dim,
    ).to(normalized)

    with torch.no_grad():
        linear = adapter.predictor[0]
        linear.weight.zero_()
        linear.bias.zero_()
        _, _, right_vectors = torch.linalg.svd(normalized, full_matrices=False)
        _require_finite_tensor(right_vectors, "predictor_pca")
        component_count = min(config.hidden_dim, right_vectors.shape[0])
        components = _canonicalize_component_signs(right_vectors[:component_count])
        linear.weight[:component_count].copy_(components)

        hidden = adapter.predictor(normalized)
        design = torch.cat(
            (hidden, torch.ones(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)),
            dim=1,
        )
        raw_targets = _raw_parameter_targets(adapter, parameter_targets, confidence)
        root_weight = torch.sqrt(teacher_weights).unsqueeze(1)
        weighted_design = design * root_weight
        weighted_targets = raw_targets * root_weight
        gram = weighted_design.transpose(0, 1) @ weighted_design
        cross = weighted_design.transpose(0, 1) @ weighted_targets
        regularizer = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        regularizer[-1, -1] = 0.0
        ridge_strength = max(config.ridge * len(items), 0.1)
        coefficients = torch.linalg.solve(
            gram + ridge_strength * regularizer,
            cross,
        )
        _require_finite_tensor(coefficients, "predictor_coefficients")
        adapter.head.weight.copy_(coefficients[:-1].transpose(0, 1))
        adapter.head.bias.copy_(coefficients[-1])
    adapter.eval()
    for name, tensor in adapter.state_dict().items():
        _require_finite_tensor(tensor, f"predictor_state:{name}")
    return adapter, mean, std


def _locked_metric_summary(
    items,
    adapter,
    mean: torch.Tensor,
    std: torch.Tensor,
    frozen_tm: FrozenSamsungTM,
) -> dict[str, Any]:
    improvements: dict[str, list[float]] = {
        "global_tone": [],
        "highlight": [],
        "roi": [],
    }
    with torch.no_grad():
        for item in items:
            if item.teacher_status is TeacherQualificationStatus.REJECTED:
                continue
            features = (item.features - mean) / std
            adapted_input = adapter(item.iphone, features, confidence=item.confidence)
            baseline = frozen_tm(item.iphone)
            student = frozen_tm(adapted_input.image)
            baseline_errors = _metric_errors(baseline, item.teacher, item.example.roi_mask)
            adapted_errors = _metric_errors(student, item.teacher, item.example.roi_mask)
            for name in improvements:
                before = baseline_errors[name]
                after = adapted_errors[name]
                if before is not None and after is not None:
                    improvement = float(before - after)
                    if not math.isfinite(improvement):
                        raise _non_finite_error(
                            f"locked_metric:{item.example.pair_id}:{name}"
                        )
                    improvements[name].append(improvement)
    summary: dict[str, Any] = {}
    for name, values in improvements.items():
        _require_finite_values(values, f"locked_metric:{name}", allow_empty=True)
        summary[f"{name}_pairs"] = len(values)
        summary[f"{name}_median_improvement"] = (
            float(torch.median(torch.tensor(values)).item())
            if values
            else float("-inf")
        )
    return summary


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
    """Train Phase 1 with fold-local pair targets, normalization and PCA."""

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

    backbone_before = _state_sha256(frozen_tm.module)
    torch.manual_seed(config.seed)
    profile = core._build_teacher_profile(source_examples, canonicalizer, frozen_tm)
    prepared = core._prepare_pairs(
        calibration_examples,
        canonicalizer,
        frozen_tm,
        TeacherQualifier(profile),
    )
    _require_finite_prepared(prepared)
    prepared_development = [item for item in prepared if item.example.split == "development"]
    prepared_locked = [item for item in prepared if item.example.split == "locked"]
    solver = core.ObservablePairSolver(ridge=config.ridge)

    fold_reports: list[FoldReport] = []
    out_of_fold: dict[str, float] = {}
    folds: tuple[GroupFold, ...] = build_group_folds(calibration_examples, folds=5)
    for fold in folds:
        fold_candidates = [
            item
            for item in prepared_development
            if item.example.scene_group in fold.train_groups
        ]
        fold_targets = core._fit_pair_targets(fold_candidates, solver, frozen_tm, config)
        _require_finite_parameter_targets(fold_targets, f"fold_{fold.fold_index}_targets")
        train_items = [
            item
            for item in fold_candidates
            if item.example.pair_id in fold_targets
        ]
        validation_items = [
            item
            for item in prepared_development
            if item.example.scene_group in fold.validation_groups
        ]
        adapter, mean, std = _fit_ridge_predictor(train_items, fold_targets, config)
        improvements, margins = core._evaluate_items(
            validation_items,
            adapter,
            mean,
            std,
            frozen_tm,
        )
        _require_finite_values(improvements, f"fold_{fold.fold_index}_improvements")
        _require_finite_values(margins, f"fold_{fold.fold_index}_margins")
        for item, improvement in zip(validation_items, improvements):
            out_of_fold[item.example.pair_id] = improvement
        median = float(torch.median(torch.tensor(improvements)).item())
        if not math.isfinite(median):
            raise _non_finite_error(f"fold_{fold.fold_index}_median")
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

    development_improvements = [
        out_of_fold.get(item.example.pair_id, 0.0)
        for item in prepared_development
    ]
    _require_finite_values(development_improvements, "development_improvements")
    final_targets = core._fit_pair_targets(prepared_development, solver, frozen_tm, config)
    _require_finite_parameter_targets(final_targets, "final_pair_targets")
    qualified_development = [
        item
        for item in prepared_development
        if item.example.pair_id in final_targets
    ]
    final_adapter, feature_mean, feature_std = _fit_ridge_predictor(
        qualified_development,
        final_targets,
        config,
    )
    locked_improvements, locked_margins = core._evaluate_items(
        prepared_locked,
        final_adapter,
        feature_mean,
        feature_std,
        frozen_tm,
    )
    _require_finite_values(locked_improvements, "locked_improvements")
    _require_finite_values(locked_margins, "locked_margins")
    locked_metrics = _locked_metric_summary(
        prepared_locked,
        final_adapter,
        feature_mean,
        feature_std,
        frozen_tm,
    )

    normalized_development = torch.cat(
        [(item.features - feature_mean) / feature_std for item in qualified_development],
        dim=0,
    )
    _require_finite_tensor(normalized_development, "normalized_development")
    support_min = normalized_development.min(dim=0).values
    support_max = normalized_development.max(dim=0).values
    _require_finite_tensor(support_min, "support_min")
    _require_finite_tensor(support_max, "support_max")
    positive_folds = sum(item.median_improvement > 0.0 for item in fold_reports)
    improved_development = sum(value > 0.0 for value in development_improvements)
    bootstrap_lower = core._bootstrap_lower(
        development_improvements,
        config.bootstrap_samples,
        config.seed,
    )
    if not math.isfinite(bootstrap_lower):
        raise _non_finite_error("development_bootstrap_lower")
    locked_median = float(torch.median(torch.tensor(locked_improvements)).item())
    if not math.isfinite(locked_median):
        raise _non_finite_error("locked_median_improvement")
    backbone_unchanged = backbone_before == _state_sha256(frozen_tm.module)

    reasons: list[str] = []
    if positive_folds < 4:
        reasons.append("fold_consistency")
    if improved_development < 30:
        reasons.append("development_prevalence")
    if bootstrap_lower <= 0.0:
        reasons.append("development_bootstrap")
    if locked_median <= 0.0:
        reasons.append("locked_direction")
    if locked_metrics["global_tone_median_improvement"] <= 0.0:
        reasons.append("locked_global_tone")
    if locked_metrics["highlight_median_improvement"] < -1e-6:
        reasons.append("locked_highlight_regression")
    if locked_metrics["roi_pairs"] < 2:
        reasons.append("locked_roi_coverage")
    elif locked_metrics["roi_median_improvement"] <= 0.0:
        reasons.append("locked_roi_direction")
    if any(margin <= 0.0 for margin in locked_margins):
        reasons.append("adapter_boundary")
    qualified_locked = sum(
        item.teacher_status is not TeacherQualificationStatus.REJECTED
        for item in prepared_locked
    )
    if len(qualified_development) < 30 or qualified_locked < 8:
        reasons.append("teacher_qualification_coverage")
    if not backbone_unchanged:
        reasons.append("samsung_backbone_changed")

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
    validation_payload = core._report_payload(report)
    validation_payload["locked_metric_summary"] = locked_metrics
    validation_payload["samsung_backbone_unchanged"] = backbone_unchanged
    validation_payload["samsung_backbone_state_sha256"] = backbone_before
    validation_payload["parameter_predictor"] = "teacher_weighted_ridge_on_fold_pca_features"
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
        "validation_report": validation_payload,
        "teacher_profile": core._profile_payload(profile),
        "data_mode": config.data_mode,
    }
    torch.save(payload, artifact_path)
    return Phase1TrainingResult(report=report, artifact_path=artifact_path)


__all__ = [
    "FoldReport",
    "Phase1Artifact",
    "Phase1TrainingConfig",
    "Phase1TrainingReport",
    "Phase1TrainingResult",
    "calibration_support_distance",
    "evaluate_phase1_artifact",
    "load_phase1_artifact",
    "run_phase1_inference",
    "train_phase1",
]
