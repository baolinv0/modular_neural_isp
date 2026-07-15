from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        if isinstance(raw, list):
            payload = raw
        elif isinstance(raw, Mapping):
            for key in ("records", "rows", "scores", "candidates"):
                if isinstance(raw.get(key), list):
                    payload = raw[key]
                    break
            else:
                raise ValueError(f"Unsupported row container: {path}")
        else:
            raise ValueError(f"Unsupported row container: {path}")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ValueError(f"All rows must be JSON objects: {path}")
    return [dict(row) for row in payload]


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    scene_id = row.get("scene_id", row.get("scene_name"))
    method = row.get("method", row.get("method_name"))
    level = row.get("level", row.get("level_name"))
    if scene_id is None or method is None or level is None:
        raise ValueError(f"Candidate row must contain scene_id, method and level: {row}")
    return str(scene_id), str(method), str(level)


def merge_candidate_scores(
    inference_records: str | Path,
    iqa_scores: str | Path,
    output_path: str | Path,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Join adjustTM inference records with candidate-level IQA outputs.

    The join key is ``(scene_id, method, level)``. This keeps the selected
    teacher tied to the exact controllable checkpoint and alpha level that
    produced it.
    """

    inference = _read_rows(inference_records)
    scores = _read_rows(iqa_scores)
    if method is not None:
        inference = [row for row in inference if str(row.get("method")) == method]
        scores = [row for row in scores if str(row.get("method", method)) == method]
    score_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in scores:
        if method is not None and "method" not in row:
            row["method"] = method
        key = _key(row)
        if key in score_map:
            raise ValueError(f"Duplicate IQA score for {key}")
        score_map[key] = row

    missing: list[tuple[str, str, str]] = []
    merged: list[dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()
    for record in inference:
        key = _key(record)
        score = score_map.get(key)
        if score is None:
            missing.append(key)
            continue
        used.add(key)
        row = dict(record)
        row.update(score)
        # Inference provenance must not be overridden by a scorer path alias.
        row["scene_id"], row["method"], row["level"] = key
        row["alpha"] = float(record["alpha"])
        row["output_path"] = str(record["output_path"])
        if "overall_score" not in row and "score" in row:
            row["overall_score"] = float(row["score"])
        merged.append(row)
    if missing:
        raise ValueError(f"Missing IQA scores for {missing[:10]}")
    unused = sorted(set(score_map) - used)
    if unused:
        raise ValueError(f"IQA scores contain unmatched candidates: {unused[:10]}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "version": 1,
        "method": method,
        "inference_count": len(inference),
        "score_count": len(scores),
        "merged_count": len(merged),
        "output_path": str(output_path),
    }
