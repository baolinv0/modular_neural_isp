from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .canonicalization import CanonicalizationConfig


def _strict_fields(payload: Mapping[str, Any], expected: set[str], section: str) -> None:
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown:
        raise ValueError(f"unknown {section} fields: " + ",".join(unknown))
    if missing:
        raise ValueError(f"missing {section} fields: " + ",".join(missing))


@dataclass(frozen=True)
class Phase2Config:
    enabled: bool
    minimum_eligible_samples: int


@dataclass(frozen=True)
class RoutingConfig:
    pixel_route_enabled: bool


@dataclass(frozen=True)
class ModelConfig:
    samsung_checkpoint: str | None
    qwen3_vl_checkpoint: str | None
    qwen_image_edit_checkpoint: str | None
    internvl_checkpoint: str | None
    ovis_checkpoint: str | None
    require_real_model: bool


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    mode: str
    seed: int
    phase2: Phase2Config
    routing: RoutingConfig
    models: ModelConfig
    canonicalization: CanonicalizationConfig

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PipelineConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("configuration must be a mapping")
        _strict_fields(
            payload,
            {"schema_version", "mode", "seed", "phase2", "routing", "models", "canonicalization"},
            "root",
        )
        if int(payload["schema_version"]) != 2:
            raise ValueError("schema_version must be 2")
        mode = str(payload["mode"])
        if mode not in {"synthetic_canary", "real"}:
            raise ValueError("mode must be synthetic_canary or real")

        phase_payload = payload["phase2"]
        routing_payload = payload["routing"]
        model_payload = payload["models"]
        canonical_payload = payload["canonicalization"]
        if not all(isinstance(value, Mapping) for value in (phase_payload, routing_payload, model_payload, canonical_payload)):
            raise TypeError("nested configuration sections must be mappings")
        _strict_fields(phase_payload, {"enabled", "minimum_eligible_samples"}, "phase2")
        _strict_fields(routing_payload, {"pixel_route_enabled"}, "routing")
        _strict_fields(
            model_payload,
            {
                "samsung_checkpoint",
                "qwen3_vl_checkpoint",
                "qwen_image_edit_checkpoint",
                "internvl_checkpoint",
                "ovis_checkpoint",
                "require_real_model",
            },
            "models",
        )
        _strict_fields(
            canonical_payload,
            {
                "exposure_scale_min",
                "exposure_scale_max",
                "reliable_dark_threshold",
                "highlight_threshold",
            },
            "canonicalization",
        )
        minimum = int(phase_payload["minimum_eligible_samples"])
        if minimum < 50:
            raise ValueError("Phase 2 requires at least 50 eligible target samples")
        models = ModelConfig(
            samsung_checkpoint=None if model_payload["samsung_checkpoint"] is None else str(model_payload["samsung_checkpoint"]),
            qwen3_vl_checkpoint=None if model_payload["qwen3_vl_checkpoint"] is None else str(model_payload["qwen3_vl_checkpoint"]),
            qwen_image_edit_checkpoint=None if model_payload["qwen_image_edit_checkpoint"] is None else str(model_payload["qwen_image_edit_checkpoint"]),
            internvl_checkpoint=None if model_payload["internvl_checkpoint"] is None else str(model_payload["internvl_checkpoint"]),
            ovis_checkpoint=None if model_payload["ovis_checkpoint"] is None else str(model_payload["ovis_checkpoint"]),
            require_real_model=bool(model_payload["require_real_model"]),
        )
        if mode == "real" and (not models.require_real_model or not models.samsung_checkpoint):
            raise ValueError("real mode requires an explicit Samsung checkpoint and require_real_model=true")
        if mode == "synthetic_canary" and models.require_real_model:
            raise ValueError("synthetic canary cannot claim a real model")
        return cls(
            schema_version=2,
            mode=mode,
            seed=int(payload["seed"]),
            phase2=Phase2Config(bool(phase_payload["enabled"]), minimum),
            routing=RoutingConfig(bool(routing_payload["pixel_route_enabled"])),
            models=models,
            canonicalization=CanonicalizationConfig(**{key: float(value) for key, value in canonical_payload.items()}),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
