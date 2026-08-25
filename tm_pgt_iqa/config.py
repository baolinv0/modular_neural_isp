from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class LabelConfig:
    background: int = 0
    face: int = 1
    skin: int = 2


@dataclass
class ScoreWeights:
    exposure: float = 0.20
    dynamic_range: float = 0.20
    face_tone: float = 0.25
    face_background: float = 0.15
    naturalness: float = 0.20


@dataclass
class GuardConfig:
    face_clip_reject: float = 0.10
    face_dark_reject: float = 0.35
    global_clip_reject: float = 0.20
    halo_reject: float = 0.20
    skin_chroma_min: float = 4.0
    skin_chroma_max: float = 55.0
    pool_outlier_z: float = 3.5


@dataclass
class ThresholdConfig:
    certified_score: float = 82.0
    usable_score: float = 65.0
    qwen_reject_confidence: float = 0.75


@dataclass
class VLMConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen3.8"
    timeout_s: float = 60.0
    max_tokens: int = 300
    temperature: float = 0.0


@dataclass
class IQAConfig:
    labels: LabelConfig = field(default_factory=LabelConfig)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    guards: GuardConfig = field(default_factory=GuardConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_dataclass(obj, values: dict):
    for key, value in values.items():
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None) -> IQAConfig:
    cfg = IQAConfig()
    if path is None:
        return cfg
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config files") from exc
        values = yaml.safe_load(text) or {}
    else:
        values = json.loads(text)
    return _merge_dataclass(cfg, values)
