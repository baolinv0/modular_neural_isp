from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from adjustTM.constants import LEVELS


REAL_LEVEL = "a_000"
FLUX_ENDPOINTS = {"a_m100", "a_p100"}


def semantic_group(level_name: str) -> str:
    valid = {name for name, _ in LEVELS}
    if level_name not in valid:
        raise KeyError(f"Unknown brightness level: {level_name}")
    if level_name == REAL_LEVEL:
        return "real_camera"
    if level_name in FLUX_ENDPOINTS:
        return "flux_endpoint"
    return "retinex_intermediate"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_image_metadata(path: Path, *, require_uint16: bool) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected three-channel image, got {image.shape}: {path}")
    if require_uint16 and image.dtype != np.uint16:
        raise TypeError(f"Input must be uint16 PNG, got {image.dtype}: {path}")
    if not require_uint16 and image.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"GT must be uint8 or uint16 PNG, got {image.dtype}: {path}")
    height, width = image.shape[:2]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "dtype": str(image.dtype),
        "height": int(height),
        "width": int(width),
        "channels": 3,
    }


def _load_scene_list(scene_list: str | Path | Iterable[str] | None, input_names: list[str]) -> list[str]:
    if scene_list is None:
        return input_names
    if isinstance(scene_list, (str, Path)):
        payload = json.loads(Path(scene_list).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("scenes", payload.get("test", payload.get("scene_names")))
        if not isinstance(payload, list):
            raise ValueError("Scene list JSON must be a list or contain scenes/test/scene_names")
        requested = [str(item) for item in payload]
    else:
        requested = [str(item) for item in scene_list]
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate scene names in scene list")
    missing = sorted(set(requested) - set(input_names))
    if missing:
        raise FileNotFoundError(f"Scene-list inputs not found: {missing[:10]}")
    return sorted(requested)


def build_manifest(
    input_dir: str | Path,
    gt_root: str | Path,
    scene_list: str | Path | Iterable[str] | None = None,
    *,
    scene_tags: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    gt_root = Path(gt_root)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not gt_root.is_dir():
        raise NotADirectoryError(gt_root)
    input_names = sorted(path.name for path in input_dir.glob("*.png") if path.is_file())
    if not input_names:
        raise ValueError(f"No PNG inputs found in {input_dir}")
    selected = _load_scene_list(scene_list, input_names)
    scenes: list[dict[str, Any]] = []
    for scene_name in selected:
        input_meta = _read_image_metadata(input_dir / scene_name, require_uint16=True)
        gt_records: dict[str, dict[str, Any]] = {}
        for level_name, alpha in LEVELS:
            level_dir = gt_root / level_name
            if not level_dir.is_dir():
                raise NotADirectoryError(level_dir)
            path = level_dir / scene_name
            if not path.is_file():
                raise FileNotFoundError(path)
            meta = _read_image_metadata(path, require_uint16=False)
            if (meta["height"], meta["width"]) != (input_meta["height"], input_meta["width"]):
                raise ValueError(
                    f"Spatial mismatch for {scene_name}/{level_name}: "
                    f"input={(input_meta['height'], input_meta['width'])}, "
                    f"gt={(meta['height'], meta['width'])}"
                )
            meta.update({"alpha": float(alpha), "semantic_group": semantic_group(level_name)})
            gt_records[level_name] = meta
        scenes.append(
            {
                "scene_id": scene_name,
                "input": input_meta,
                "gt": gt_records,
                "tags": sorted((scene_tags or {}).get(scene_name, [])),
            }
        )
    manifest = {
        "manifest_version": 1,
        "input_dir": str(input_dir.resolve()),
        "gt_root": str(gt_root.resolve()),
        "scene_count": len(scenes),
        "levels": {name: alpha for name, alpha in LEVELS},
        "scenes": scenes,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    return manifest


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
