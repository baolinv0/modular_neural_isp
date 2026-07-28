"""Configuration contracts for multi-checkpoint non-aligned evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint and the metadata required to reconstruct it."""

    name: str
    checkpoint: Path
    run_config: Optional[Path] = None
    use_3d_lut: bool = False


@dataclass(frozen=True)
class ExperimentGroup:
    """A matched pretrained -> Stage-1 -> Stage-2 experiment chain."""

    name: str
    pretrained: ModelSpec
    stage1: ModelSpec
    stage2: Tuple[ModelSpec, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    groups: Tuple[ExperimentGroup, ...]


def _resolve_existing(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value.strip())
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _model_spec(
    root: Path,
    payload: Mapping[str, Any],
    *,
    default_name: str,
    label: str,
) -> ModelSpec:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    raw_name = payload.get("name", default_name)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"{label}.name must be non-empty")
    checkpoint = _resolve_existing(root, payload.get("checkpoint"), f"{label}.checkpoint")
    raw_run_config = payload.get("run_config")
    run_config = None
    if raw_run_config not in (None, ""):
        run_config = _resolve_existing(root, raw_run_config, f"{label}.run_config")
    use_3d_lut = payload.get("use_3d_lut", False)
    if not isinstance(use_3d_lut, bool):
        raise ValueError(f"{label}.use_3d_lut must be boolean")
    return ModelSpec(
        name=raw_name.strip(),
        checkpoint=checkpoint,
        run_config=run_config,
        use_3d_lut=use_3d_lut,
    )


def load_experiment_config(path: str | Path) -> EvaluationConfig:
    """Loads and validates the JSON experiment configuration.

    Paths are resolved relative to the JSON file. Checkpoints and explicit run
    configs must exist so evaluation fails before any expensive inference.
    """

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON experiment config: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Experiment config root must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version={schema_version!r}; expected 1")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("Experiment config requires a non-empty groups list")

    root = config_path.parent
    groups = []
    seen_groups: set[str] = set()
    for index, raw_group in enumerate(raw_groups):
        label = f"groups[{index}]"
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"{label} must be an object")
        raw_name = raw_group.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{label}.name must be non-empty")
        name = raw_name.strip()
        if name in seen_groups:
            raise ValueError(f"duplicate group name: {name}")
        seen_groups.add(name)

        pretrained = _model_spec(root, raw_group.get("pretrained", {}), default_name="pretrained", label=f"{label}.pretrained")
        stage1 = _model_spec(root, raw_group.get("stage1", {}), default_name="stage1", label=f"{label}.stage1")
        raw_stage2 = raw_group.get("stage2")
        if not isinstance(raw_stage2, list) or not raw_stage2:
            raise ValueError(f"{label}.stage2 must be a non-empty list")
        stage2_specs = []
        seen_stage2: set[str] = set()
        for stage2_index, stage2_payload in enumerate(raw_stage2):
            spec = _model_spec(
                root,
                stage2_payload,
                default_name=f"stage2_{stage2_index}",
                label=f"{label}.stage2[{stage2_index}]",
            )
            if spec.name in seen_stage2:
                raise ValueError(f"duplicate stage2 name in group {name}: {spec.name}")
            if spec.name in {pretrained.name, stage1.name}:
                raise ValueError(f"stage2 label conflicts with pretrained/stage1 in group {name}: {spec.name}")
            seen_stage2.add(spec.name)
            stage2_specs.append(spec)
        groups.append(ExperimentGroup(name=name, pretrained=pretrained, stage1=stage1, stage2=tuple(stage2_specs)))

    return EvaluationConfig(schema_version=1, groups=tuple(groups))
