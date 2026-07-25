"""Unpaired same-scene style adaptation for the photofinishing module."""

from .contracts import (
    RegionPair,
    StylePairRecord,
    Stage1LossWeights,
    Stage2LossWeights,
    TwoStageTrainingConfig,
    load_manifest,
    validate_disjoint_manifests,
)
from .data import UnpairedStyleDataset
from .losses import Stage1UnpairedLoss, Stage2UnpairedLoss
from .stages import configure_trainable_stage, trainable_parameter_names

__all__ = [
    "RegionPair",
    "StylePairRecord",
    "Stage1LossWeights",
    "Stage2LossWeights",
    "TwoStageTrainingConfig",
    "load_manifest",
    "validate_disjoint_manifests",
    "UnpairedStyleDataset",
    "Stage1UnpairedLoss",
    "Stage2UnpairedLoss",
    "configure_trainable_stage",
    "trainable_parameter_names",
]
