"""Strict data and configuration contracts for unpaired style adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class RegionPair:
  """Independent masks for a semantic region in the two non-aligned images."""

  input_mask: str
  reference_mask: str
  weight: float = 1.0

  def __post_init__(self) -> None:
    if not self.input_mask or not self.reference_mask:
      raise ValueError("region mask paths must be non-empty")
    if not math.isfinite(self.weight) or self.weight <= 0:
      raise ValueError("region weight must be finite and positive")

  @classmethod
  def from_dict(cls, payload: Mapping[str, object]) -> "RegionPair":
    allowed = {"input_mask", "reference_mask", "weight"}
    unknown = set(payload) - allowed
    if unknown:
      raise ValueError(f"unknown region fields: {sorted(unknown)}")
    return cls(
      input_mask=str(payload["input_mask"]),
      reference_mask=str(payload["reference_mask"]),
      weight=float(payload.get("weight", 1.0)),
    )


@dataclass(frozen=True)
class StylePairRecord:
  """One same-scene, non-pixel-aligned training pair."""

  sample_id: str
  scene_group: str
  input_path: str
  reference_path: str
  split: str
  confidence: float = 1.0
  regions: Dict[str, RegionPair] = field(default_factory=dict)
  metadata: Dict[str, object] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.sample_id.strip():
      raise ValueError("sample_id must be non-empty")
    if not self.scene_group.strip():
      raise ValueError("scene_group must be non-empty")
    if self.split not in {"train", "validation", "test"}:
      raise ValueError("split must be train, validation, or test")
    if not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
      raise ValueError("confidence must be finite and in (0, 1]")
    if not self.input_path or not self.reference_path:
      raise ValueError("input_path and reference_path are required")

  @classmethod
  def from_dict(cls, payload: Mapping[str, object]) -> "StylePairRecord":
    allowed = {
      "sample_id", "scene_group", "input_path", "reference_path", "split",
      "confidence", "regions", "metadata",
    }
    unknown = set(payload) - allowed
    if unknown:
      raise ValueError(f"unknown sample fields: {sorted(unknown)}")
    required = {"sample_id", "scene_group", "input_path", "reference_path", "split"}
    missing = required - set(payload)
    if missing:
      raise ValueError(f"missing sample fields: {sorted(missing)}")
    regions_payload = payload.get("regions", {})
    if not isinstance(regions_payload, Mapping):
      raise ValueError("regions must be an object")
    regions = {
      str(name): RegionPair.from_dict(value)
      for name, value in regions_payload.items()
    }
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
      raise ValueError("metadata must be an object")
    return cls(
      sample_id=str(payload["sample_id"]),
      scene_group=str(payload["scene_group"]),
      input_path=str(payload["input_path"]),
      reference_path=str(payload["reference_path"]),
      split=str(payload["split"]),
      confidence=float(payload.get("confidence", 1.0)),
      regions=regions,
      metadata=dict(metadata),
    )


@dataclass(frozen=True)
class Stage1LossWeights:
  exposure: float = 1.0
  luminance_cdf: float = 1.0
  percentiles: float = 1.0
  tone_regions: float = 0.5
  semantic_regions: float = 1.0
  edge_anchor: float = 0.25
  high_frequency_anchor: float = 0.10
  residual_anchor: float = 0.05


@dataclass(frozen=True)
class Stage2LossWeights:
  chroma_histogram: float = 1.0
  chroma_moments: float = 0.5
  saturation_cdf: float = 0.5
  semantic_regions: float = 1.0
  luminance_preserve: float = 2.0
  edge_anchor: float = 0.10
  lut_identity: float = 0.02
  lut_total_variation: float = 0.02
  lut_bound: float = 0.10


@dataclass(frozen=True)
class TwoStageTrainingConfig:
  stage1_epochs: int = 5
  stage2_epochs: int = 5
  stage1_learning_rate: float = 1e-5
  stage2_learning_rate: float = 1e-5
  weight_decay: float = 0.0
  gradient_clip_norm: float = 5.0
  histogram_bins: int = 32
  chroma_bins: int = 16
  histogram_sigma: float = 0.03
  chroma_sigma: float = 0.04
  stage1: Stage1LossWeights = field(default_factory=Stage1LossWeights)
  stage2: Stage2LossWeights = field(default_factory=Stage2LossWeights)

  def __post_init__(self) -> None:
    for name in ("stage1_epochs", "stage2_epochs", "histogram_bins", "chroma_bins"):
      if getattr(self, name) <= 0:
        raise ValueError(f"{name} must be positive")
    for name in (
      "stage1_learning_rate", "stage2_learning_rate", "gradient_clip_norm",
      "histogram_sigma", "chroma_sigma",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
      raise ValueError("weight_decay must be finite and non-negative")

  def to_dict(self) -> Dict[str, object]:
    return asdict(self)

  @classmethod
  def from_dict(cls, payload: Mapping[str, object]) -> "TwoStageTrainingConfig":
    data = dict(payload)
    stage1_payload = data.pop("stage1", {})
    stage2_payload = data.pop("stage2", {})
    return cls(
      stage1=Stage1LossWeights(**stage1_payload),
      stage2=Stage2LossWeights(**stage2_payload),
      **data,
    )


def _resolve_path(path: str, base_dir: Path) -> str:
  candidate = Path(path)
  if not candidate.is_absolute():
    candidate = base_dir / candidate
  return str(candidate.resolve())


def load_manifest(path: str, *, require_files: bool = True) -> List[StylePairRecord]:
  """Load a strict JSONL manifest and resolve paths relative to the manifest."""
  manifest_path = Path(path).resolve()
  records: List[StylePairRecord] = []
  seen_ids = set()
  with manifest_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      try:
        payload = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
      record = StylePairRecord.from_dict(payload)
      if record.sample_id in seen_ids:
        raise ValueError(f"duplicate sample_id: {record.sample_id}")
      seen_ids.add(record.sample_id)
      regions = {
        name: RegionPair(
          input_mask=_resolve_path(region.input_mask, manifest_path.parent),
          reference_mask=_resolve_path(region.reference_mask, manifest_path.parent),
          weight=region.weight,
        )
        for name, region in record.regions.items()
      }
      resolved = StylePairRecord(
        sample_id=record.sample_id,
        scene_group=record.scene_group,
        input_path=_resolve_path(record.input_path, manifest_path.parent),
        reference_path=_resolve_path(record.reference_path, manifest_path.parent),
        split=record.split,
        confidence=record.confidence,
        regions=regions,
        metadata=record.metadata,
      )
      if require_files:
        paths = [resolved.input_path, resolved.reference_path]
        for region in resolved.regions.values():
          paths.extend([region.input_mask, region.reference_mask])
        missing = [item for item in paths if not Path(item).is_file()]
        if missing:
          raise FileNotFoundError(f"missing files for {record.sample_id}: {missing}")
      records.append(resolved)
  if not records:
    raise ValueError("manifest contains no samples")
  return records


def validate_disjoint_manifests(
    train_records: Sequence[StylePairRecord],
    validation_records: Sequence[StylePairRecord],
) -> None:
  """Fail if samples, scenes, or exact file identities cross the split boundary."""
  train_ids = {record.sample_id for record in train_records}
  val_ids = {record.sample_id for record in validation_records}
  if train_ids & val_ids:
    raise ValueError(f"sample IDs cross train/validation: {sorted(train_ids & val_ids)}")
  train_scenes = {record.scene_group for record in train_records}
  val_scenes = {record.scene_group for record in validation_records}
  if train_scenes & val_scenes:
    raise ValueError(f"scene groups cross train/validation: {sorted(train_scenes & val_scenes)}")
  train_files = {(record.input_path, record.reference_path) for record in train_records}
  val_files = {(record.input_path, record.reference_path) for record in validation_records}
  if train_files & val_files:
    raise ValueError("exact input/reference pairs cross train/validation")
