from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

from .constants import LEVELS


def _scene_digest(scene_names: Sequence[str]) -> str:
    payload = "\n".join(sorted(scene_names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def create_or_load_split_manifest(
    scene_names: Sequence[str],
    path: str | Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Create one scene-disjoint train/validation split or validate an existing one."""
    names = sorted(scene_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("scene_names must be non-empty and unique")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    path = Path(path)
    expected_set = set(names)
    digest = _scene_digest(names)

    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        train = manifest.get("train")
        val = manifest.get("val")
        if not isinstance(train, list) or not isinstance(val, list):
            raise ValueError(f"Invalid split manifest: {path}")
        if set(train) & set(val):
            raise ValueError("Split manifest contains train/validation overlap")
        if set(train) | set(val) != expected_set:
            raise ValueError("Split manifest does not match the current scene set")
        stored_digest = manifest.get("scene_digest")
        if stored_digest is not None and stored_digest != digest:
            raise ValueError("Split manifest scene digest mismatch")
        return manifest

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or val_fraction == 0.0:
        val_count = 0
    else:
        val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_fraction)))
    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    manifest = {
        "version": 1,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "scene_digest": digest,
        "train": train,
        "val": val,
    }
    _atomic_write_json(path, manifest)
    return manifest


def write_sample_index(dataset, path: str | Path) -> dict:
    """Write a deterministic JSON index for audit and cross-method reuse."""
    entries = []
    for index, sample in enumerate(dataset.samples):
        if len(sample) == 3:
            scene_name, low_index, high_index = sample
            low_name, alpha_low = LEVELS[low_index]
            high_name, alpha_high = LEVELS[high_index]
            entries.append(
                {
                    "index": index,
                    "scene_name": scene_name,
                    "level_low": low_name,
                    "alpha_low": alpha_low,
                    "level_high": high_name,
                    "alpha_high": alpha_high,
                }
            )
        elif len(sample) == 2:
            scene_name, level_index = sample
            level_name, alpha = LEVELS[level_index]
            entries.append(
                {
                    "index": index,
                    "scene_name": scene_name,
                    "level_name": level_name,
                    "alpha": alpha,
                }
            )
        else:
            raise ValueError(f"Unsupported sample tuple: {sample}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        "version": 1,
        "count": len(entries),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "samples": entries,
    }
    _atomic_write_json(Path(path), payload)
    return payload
