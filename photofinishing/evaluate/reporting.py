"""Machine-readable reports and visual comparison panels."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


_NON_INCREMENTAL_METRICS = {"signed_ev_error"}


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
    common = sorted((set(before) & set(after)) - _NON_INCREMENTAL_METRICS)
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


def _format_number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def _build_markdown_report(summary: Mapping[str, object], decisions: Mapping[str, object]) -> str:
    cross = decisions.get("cross_group", {}) if isinstance(decisions, Mapping) else {}
    stage1 = cross.get("stage1", {}) if isinstance(cross, Mapping) else {}
    necessity = cross.get("stage2_necessity", {}) if isinstance(cross, Mapping) else {}
    variants = cross.get("stage2", {}) if isinstance(cross, Mapping) else {}
    recommendation = cross.get("recommended_stage2", {}) if isinstance(cross, Mapping) else {}
    expansion = decisions.get("data_expansion", {}) if isinstance(decisions, Mapping) else {}
    lines = [
        "# Non-Aligned Photofinishing Evaluation Report",
        "",
        "## Overall answers",
        "",
        f"- **Stage 1 effectiveness:** `{stage1.get('status', 'undetermined')}` — {stage1.get('reason', 'No decision evidence.')}",
        f"- **Stage 2 necessity:** `{necessity.get('status', 'undetermined')}` — {necessity.get('reason', 'No repeated-reference evidence.')}",
        f"- **Recommended Stage 2:** `{recommendation.get('recommended') or 'none'}` — {recommendation.get('reason', 'No variant passed all gates.')}",
        "",
        "## Stage 2 variants",
        "",
        "| Variant | Status | Primary median gain | Win rate | 95% CI | Luminance preserved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if isinstance(variants, Mapping) and variants:
        for name, decision in variants.items():
            primary = decision.get("primary", {}) if isinstance(decision, Mapping) else {}
            lines.append(
                "| {name} | {status} | {median} | {win} | [{low}, {high}] | {guard} |".format(
                    name=name,
                    status=decision.get("status", "undetermined"),
                    median=_format_number(primary.get("median")),
                    win=_format_number(primary.get("win_rate")),
                    low=_format_number(primary.get("ci95_low")),
                    high=_format_number(primary.get("ci95_high")),
                    guard=decision.get("luminance_preserved", "N/A"),
                )
            )
    else:
        lines.append("| none | undetermined | N/A | N/A | N/A | N/A |")

    lines.extend([
        "",
        "## Data sufficiency",
        "",
        f"- **Expand evaluation data:** `{expansion.get('expand_evaluation_data', 'undetermined')}`",
    ])
    for reason in expansion.get("evaluation_reasons", []) if isinstance(expansion, Mapping) else []:
        lines.append(f"  - {reason}")
    lines.append(f"- **Expand training data:** `{expansion.get('expand_training_data', 'undetermined')}`")
    for reason in expansion.get("training_reasons", []) if isinstance(expansion, Mapping) else []:
        lines.append(f"  - {reason}")
    target_slices = expansion.get("target_slices", []) if isinstance(expansion, Mapping) else []
    lines.append(f"- **Target slices:** {', '.join(target_slices) if target_slices else 'none identified'}")

    lines.extend([
        "",
        "## How to read the evidence",
        "",
        "- Stage 1 is judged from `pretrained -> stage1` improvements in absolute EV error, luminance quantiles, tone shape, and clipping safety.",
        "- Stage 2 is judged only from `stage1 -> stage2` incremental color improvement; inherited Stage-1 gains do not count.",
        "- The primary Stage-2 metric is luminance-conditioned CbCr sliced Wasserstein distance.",
        "- A Stage-2 variant must improve color while preserving Stage-1 brightness, tone shape, and clipping.",
        "- Stage-2 necessity is only conclusive when repeated reference images establish a capture/non-alignment noise floor.",
        "",
        "## Output files",
        "",
        "- `per_sample_metrics.csv`: absolute distance of every model to the reference for every scene.",
        "- `incremental_improvements.csv`: paired Stage-1 and Stage-2 gains; positive values indicate improvement.",
        "- `summary.json` / `summary.csv`: mean, median, tail, win-rate, and confidence-interval evidence.",
        "- `decisions.json`: complete machine-readable decisions and reasons.",
        "- `panels/`: visual Input / Pretrained / Stage1 / Stage2 / Reference comparisons.",
    ])
    return "\n".join(lines) + "\n"


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
    (output / "report.md").write_text(
        _build_markdown_report(safe_summary, safe_decisions), encoding="utf-8"
    )


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
