"""Machine-readable reports and visual comparison panels."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


def _finite_or_blank(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def build_metric_rows(
    group: str,
    sample_id: str,
    scene_tags: Sequence[str],
    model_metrics: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    rows = []
    for model_name, metrics in model_metrics.items():
        row: dict[str, object] = {
            "group": group,
            "sample_id": sample_id,
            "scene_tags": ";".join(scene_tags),
            "model": model_name,
        }
        row.update({name: float(value) for name, value in metrics.items()})
        rows.append(row)
    return rows


def _increment_row(
    group: str,
    sample_id: str,
    scene_tags: Sequence[str],
    transition: str,
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, object]:
    common = sorted(set(before) & set(after))
    row: dict[str, object] = {
        "group": group,
        "sample_id": sample_id,
        "scene_tags": ";".join(scene_tags),
        "transition": transition,
    }
    for metric in common:
        left = float(before[metric])
        right = float(after[metric])
        row[metric] = round(left - right, 12) if math.isfinite(left) and math.isfinite(right) else float("nan")
    return row


def build_increment_rows(
    group: str,
    sample_id: str,
    scene_tags: Sequence[str],
    pretrained_metrics: Mapping[str, float],
    stage1_metrics: Mapping[str, float],
    stage2_metrics: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    rows = [
        _increment_row(
            group,
            sample_id,
            scene_tags,
            "pretrained_to_stage1",
            pretrained_metrics,
            stage1_metrics,
        )
    ]
    for name, metrics in stage2_metrics.items():
        rows.append(_increment_row(
            group,
            sample_id,
            scene_tags,
            f"stage1_to_{name}",
            stage1_metrics,
            metrics,
        ))
    return rows


def _fieldnames(rows: Sequence[Mapping[str, object]], leading: Sequence[str]) -> list[str]:
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set(leading)
    return list(leading) + sorted(all_fields - set(leading))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], leading: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fieldnames(rows, leading)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite_or_blank(row.get(key, "")) for key in fields})


def _flatten_summary(value: object, prefix: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_summary(item, next_prefix))
    elif isinstance(value, (list, tuple)):
        rows.append({"path": prefix, "value": json.dumps(_json_safe(value), ensure_ascii=False)})
    else:
        rows.append({"path": prefix, "value": _finite_or_blank(value)})
    return rows


def write_reports(
    output_dir: str | Path,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    increment_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    decisions: Mapping[str, object],
) -> None:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "per_sample_metrics.csv",
        metric_rows,
        leading=("group", "sample_id", "scene_tags", "model"),
    )
    _write_csv(
        output / "incremental_improvements.csv",
        increment_rows,
        leading=("group", "sample_id", "scene_tags", "transition"),
    )
    safe_summary = _json_safe(summary)
    safe_decisions = _json_safe(decisions)
    (output / "summary.json").write_text(
        json.dumps(safe_summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    (output / "decisions.json").write_text(
        json.dumps(safe_decisions, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(output / "summary.csv", _flatten_summary(safe_summary), leading=("path", "value"))


def render_comparison_panel(
    images: Mapping[str, np.ndarray],
    output_path: str | Path,
    *,
    tile_size: int = 384,
) -> None:
    if not images:
        raise ValueError("At least one image is required for a comparison panel")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    label_height = max(24, tile_size // 12)
    tiles = []
    for label, image in images.items():
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Panel image {label!r} must have shape [H,W,3]")
        rgb = np.clip(rgb, 0.0, 1.0)
        resized = cv2.resize(rgb, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tile = np.zeros((tile_size + label_height, tile_size, 3), dtype=np.uint8)
        tile[:tile_size] = np.rint(resized * 255.0).astype(np.uint8)
        cv2.putText(
            tile,
            str(label),
            (5, tile_size + label_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, tile_size / 900.0),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    panel_rgb = np.concatenate(tiles, axis=1)
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write comparison panel: {path}")
