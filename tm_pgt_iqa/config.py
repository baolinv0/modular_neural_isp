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
class ObjectiveMetricConfig:
    """Calibratable bands for the small deterministic V2 objective score.

    Keeping these values in configuration makes later human-ranking calibration
    a data change instead of a code change.  The defaults intentionally retain
    the V1 scoring behaviour.
    """

    exposure_band: tuple[float, float, float, float] = (-5.0, -3.1, -1.4, -0.35)
    highlight_headroom_band: tuple[float, float, float, float] = (0.0, 0.015, 0.18, 0.35)
    shadow_separation_band: tuple[float, float, float, float] = (0.0, 0.005, 0.12, 0.25)
    midtone_span_band: tuple[float, float, float, float] = (0.02, 0.08, 0.45, 0.70)
    face_tone_span_band: tuple[float, float, float, float] = (0.15, 0.55, 2.8, 4.5)
    face_shadow_band: tuple[float, float, float, float] = (0.05, 0.20, 1.5, 2.5)
    face_highlight_band: tuple[float, float, float, float] = (0.05, 0.20, 1.5, 2.5)
    face_background_band: tuple[float, float, float, float] = (-3.0, -1.2, 2.0, 3.5)
    global_log_std_band: tuple[float, float, float, float] = (0.15, 0.45, 2.0, 3.2)
    global_log_skew_center: float = -0.2
    global_log_skew_sigma: float = 1.6
    dynamic_range_components: tuple[float, float, float] = (0.35, 0.25, 0.40)
    face_tone_components: tuple[float, float, float] = (0.45, 0.275, 0.275)
    naturalness_components: tuple[float, float] = (0.60, 0.40)
    dynamic_range_clip_scale: float = 0.20


@dataclass
class GuardConfig:
    face_clip_reject: float = 0.10
    face_dark_reject: float = 0.35
    global_clip_reject: float = 0.20
    # ``halo_reject`` is retained as a backwards-compatible config spelling.
    # Halo is a soft warning in V2 and never causes a hard guard failure.
    halo_reject: float = 0.20
    # ``None`` means: use the legacy ``halo_reject`` threshold as warning
    # threshold.  New configs may set this independently.
    halo_warning: float | None = None
    face_highlight_threshold: float = 0.98
    face_dark_threshold: float = 0.02
    skin_chroma_min: float = 4.0
    skin_chroma_max: float = 55.0
    pool_outlier_z: float = 3.5
    preservation_low_frequency_suspicious: float = 0.22
    preservation_low_frequency_fail: float = 0.38
    preservation_edge_agreement_suspicious: float = 0.60
    preservation_edge_agreement_fail: float = 0.35
    preservation_face_correlation_suspicious: float = 0.70
    preservation_face_correlation_fail: float = 0.35
    preservation_hue_shift_suspicious_deg: float = 18.0
    preservation_hue_shift_fail_deg: float = 35.0
    preservation_edge_percentile: float = 85.0


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
    objective: ObjectiveMetricConfig = field(default_factory=ObjectiveMetricConfig)
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
