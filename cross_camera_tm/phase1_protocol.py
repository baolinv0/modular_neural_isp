from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from . import phase1_training as core
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
    Phase1TrainingConfig,
    Phase1TrainingReport,
    Phase1TrainingResult,
    calibration_support_distance,
    evaluate_phase1_artifact,
    load_phase1_artifact,
    run_phase1_inference,
)


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
    """Train Phase 1 with fold-local pair targets and normalization.

    Pair-specific parameters for a validation scene are never computed or
    consumed by that fold's training path. The final artifact is fitted only
    after out-of-fold behavior has been measured and the development protocol
    is frozen.
    """

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
    profile = core._build_teacher_profile(source_examples, canonicalizer, frozen_tm)
    prepared = core._prepare_pairs(
        calibration_examples,
        canonicalizer,
        frozen_tm,
        TeacherQualifier(profile),
    )
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
        fold_targets = core._fit_pair_targets(
            fold_candidates,
            solver,
            frozen_tm,
            config,
        )
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
        adapter, mean, std = core._train_predictor(train_items, fold_targets, config)
        improvements, _ = core._evaluate_items(
            validation_items,
            adapter,
            mean,
            std,
            frozen_tm,
        )
        for item, improvement in zip(validation_items, improvements):
            out_of_fold[item.example.pair_id] = improvement
        median = (
            float(torch.median(torch.tensor(improvements)).item())
            if improvements
            else float("-inf")
        )
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

    final_targets = core._fit_pair_targets(
        prepared_development,
        solver,
        frozen_tm,
        config,
    )
    qualified_development = [
        item
        for item in prepared_development
        if item.example.pair_id in final_targets
    ]
    final_adapter, feature_mean, feature_std = core._train_predictor(
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

    normalized_development = torch.cat(
        [
            (item.features - feature_mean) / feature_std
            for item in qualified_development
        ],
        dim=0,
    )
    support_min = normalized_development.min(dim=0).values
    support_max = normalized_development.max(dim=0).values
    positive_folds = sum(item.median_improvement > 0.0 for item in fold_reports)
    improved_development = sum(value > 0.0 for value in development_improvements)
    bootstrap_lower = core._bootstrap_lower(
        development_improvements,
        config.bootstrap_samples,
        config.seed,
    )
    locked_median = float(torch.median(torch.tensor(locked_improvements)).item())

    reasons: list[str] = []
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
        item.teacher_status is not TeacherQualificationStatus.REJECTED
        for item in prepared_locked
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
        "validation_report": core._report_payload(report),
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
