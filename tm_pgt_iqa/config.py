from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class LabelConfig:
    background: int = 0
    face: int = 1
    skin: int = 2
    human: int = 3


@dataclass
class RetinexConfig:
    gaussian_radius: float = 31.0
    eps: float = 1e-4
    levels_ev: dict[str, float] = field(default_factory=lambda: {
        "a_m30": -0.45,
        "a_m20": -0.30,
        "a_m10": -0.15,
        "a_000": 0.00,
        "a_p10": 0.15,
        "a_p20": 0.30,
        "a_p30": 0.45,
        "a_p40": 0.60,
    })


@dataclass
class LocalFaceTMConfig:
    lift_ev: dict[str, float] = field(default_factory=lambda: {
        "low": 0.30,
        "mid": 0.60,
        "high": 0.90,
    })
    dilation_radius: int = 8
    smoothing_radius: float = 4.0


@dataclass
class ToneShapeConfig:
    median_ev_tolerance: float = 0.10
    shadow_preserve_gamma: float = 1.12
    soft_highlight_gamma: float = 0.88
    smoothing_radius: float = 3.0


@dataclass
class GainConfig:
    max_gain: float = 4.0
    max_clip_ratio: float = 0.01
    search_steps: int = 24


@dataclass
class QwenEditConfig:
    enabled: bool = False
    base_url: str | None = None
    model: str | None = None
    timeout_s: float = 120.0


@dataclass
class CandidateGenerationConfig:
    retinex: RetinexConfig = field(default_factory=RetinexConfig)
    local_face_tm: LocalFaceTMConfig = field(default_factory=LocalFaceTMConfig)
    tone_shape: ToneShapeConfig = field(default_factory=ToneShapeConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    qwen_edit: QwenEditConfig = field(default_factory=QwenEditConfig)


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
class SemanticConfig:
    enabled: bool = True
    top_k: int = 5
    reject_confidence: float = 0.80
    pairwise_min_confidence: float = 0.65
    equivalent_margin: float = 1.0


@dataclass
class IQAConfig:
    labels: LabelConfig = field(default_factory=LabelConfig)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    guards: GuardConfig = field(default_factory=GuardConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    candidate_generation: CandidateGenerationConfig = field(default_factory=CandidateGenerationConfig)

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
