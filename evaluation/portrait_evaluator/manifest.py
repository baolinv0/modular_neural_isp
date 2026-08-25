from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from portrait_evaluator.models import SceneSpec


@dataclass(slots=True)
class EvaluationManifest:
    path: Path
    scenes: list[SceneSpec]


def _resolve(path: str, root: Path, override: Path | None) -> Path:
    raw = Path(path)
    if override is not None:
        return (override / raw.name).resolve()
    if raw.is_absolute():
        return raw.resolve()
    return (root / raw).resolve()


def _bbox_map(value: Any) -> dict[str, tuple[float, float, float, float]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, tuple[float, float, float, float]] = {}
    for role, box in value.items():
        if isinstance(box, (list, tuple)) and len(box) == 4:
            out[str(role)] = tuple(float(x) for x in box)
    return out


def load_manifest(path: Path | str, *, source_override: Path | None = None, baseline_override: Path | None = None, candidate_override: Path | None = None, reference_override: Path | None = None) -> EvaluationManifest:
    path = Path(path).resolve(); payload = yaml.safe_load(path.read_text()) or {}
    scenes = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("manifest must contain a non-empty scenes list")
    parsed: list[SceneSpec] = []; seen: set[str] = set(); root = path.parent
    for item in scenes:
        if not isinstance(item, dict): raise ValueError("each scene must be a mapping")
        sid = str(item.get("id", "")).strip()
        if not sid or sid in seen: raise ValueError(f"scene id is missing or duplicated: {sid!r}")
        seen.add(sid)
        for role in ("source", "baseline", "candidate", "reference"):
            if not item.get(role): raise ValueError(f"scene {sid} missing {role}")
        tags = tuple(str(t) for t in item.get("tags", []) if str(t)); family = str(item.get("family") or (tags[0] if tags else "all")); split = str(item.get("split", "optimization"))
        if split not in {"optimization", "holdout"}: raise ValueError(f"scene {sid} has unsupported split {split}")
        parsed.append(SceneSpec(id=sid, source=_resolve(str(item["source"]), root, source_override), baseline=_resolve(str(item["baseline"]), root, baseline_override), candidate=_resolve(str(item["candidate"]), root, candidate_override), reference=_resolve(str(item["reference"]), root, reference_override), tags=tags, family=family, split=split, face_bbox=_bbox_map(item.get("face_bbox")), aligned_reference=bool(item.get("aligned_reference", False))))
    return EvaluationManifest(path, parsed)
