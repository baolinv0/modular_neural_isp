"""Multi-checkpoint evaluation for same-scene non-pixel-aligned Photofinishing adaptation."""

from .config import EvaluationConfig, ExperimentGroup, ModelSpec, load_experiment_config
from .data import EvaluationRecord, load_evaluation_manifest
from .decision import DecisionThresholds
from .metrics import compute_all_metrics, compute_color_metrics, compute_luminance_metrics

__all__ = [
    "DecisionThresholds",
    "EvaluationConfig",
    "EvaluationRecord",
    "ExperimentGroup",
    "ModelSpec",
    "compute_all_metrics",
    "compute_color_metrics",
    "compute_luminance_metrics",
    "load_evaluation_manifest",
    "load_experiment_config",
]
