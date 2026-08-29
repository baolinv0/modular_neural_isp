"""Auditable report and visualization writers for TM PGT IQA V2.

This module deliberately has no evaluator dependency.  The batch CLI gives it
plain candidate records and result dictionaries, so an offline report can also
be written for a mocked or future semantic backend.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _replace_candidate_paths(value: Any, candidate_ids: Mapping[str, str]) -> Any:
    """Translate evaluator-private image paths to stable manifest IDs."""
    if isinstance(value, str):
        return candidate_ids.get(value, value)
    if isinstance(value, list):
        return [_replace_candidate_paths(item, candidate_ids) for item in value]
    if isinstance(value, dict):
        return {str(key): _replace_candidate_paths(item, candidate_ids) for key, item in value.items()}
    return value


def _candidate_result(result: Any, candidate_id: str) -> dict:
    value = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    value["candidate"] = candidate_id
    value["candidate_id"] = candidate_id
    return value


def _grid(path: Path, items: Iterable[tuple[Path, str]], *, columns: int = 4) -> None:
    """Write a small durable contact sheet without optional visualization deps."""
    entries = list(items)
    if not entries:
        raise ValueError("a report grid needs at least one image")
    thumb_w, thumb_h, caption_h, margin = 220, 180, 34, 10
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * (thumb_w + margin) + margin, rows * (thumb_h + caption_h + margin) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image_path, caption) in enumerate(entries):
        x = margin + (index % columns) * (thumb_w + margin)
        y = margin + (index // columns) * (thumb_h + caption_h + margin)
        with Image.open(image_path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            offset_x = x + (thumb_w - preview.width) // 2
            offset_y = y + (thumb_h - preview.height) // 2
            canvas.paste(preview, (offset_x, offset_y))
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + caption_h), fill="white")
        draw.text((x + 3, y + thumb_h + 3), caption[:42], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def write_scene_artifacts(
    output_root: str | Path,
    *,
    scene_id: str,
    source_path: str | Path,
    candidates: list[Mapping[str, Any]],
    results: list[Any],
    selection: Mapping[str, Any],
    semantic: Mapping[str, Any] | None,
    ranking: Mapping[str, Any] | None,
) -> dict:
    """Materialize one scene's copied pool, semantic evidence, and grids.

    ``candidates`` contains ``candidate_id``, ``image_path`` and ``manifest``.
    The manifest remains the authority for candidate identity throughout the
    report; image filenames are never used to infer a production family.
    """
    root = Path(output_root)
    source = Path(source_path)
    candidate_ids = {str(item["image_path"]): str(item["candidate_id"]) for item in candidates}
    by_id = {str(item["candidate_id"]): item for item in candidates}
    copied_pool = root / "candidates" / scene_id
    copied_pool.mkdir(parents=True, exist_ok=True)
    for item in candidates:
        image = Path(item["image_path"])
        manifest_path = Path(item["manifest_path"])
        shutil.copy2(image, copied_pool / image.name)
        shutil.copy2(manifest_path, copied_pool / manifest_path.name)

    public_results = []
    for result in sorted(results, key=lambda item: (-float(item.quality_score), str(item.candidate))):
        candidate_id = candidate_ids[str(result.candidate)]
        public_results.append(_candidate_result(result, candidate_id))
    public_selection = _replace_candidate_paths(dict(selection), candidate_ids)
    public_ranking = _replace_candidate_paths(dict(ranking) if ranking else None, candidate_ids)
    public_semantic = _replace_candidate_paths(dict(semantic) if semantic else None, candidate_ids)
    selected_id = public_selection.get("selected")
    selected_candidate = next((item for item in candidates if item["candidate_id"] == selected_id), None)
    selected_rel = None
    if selected_candidate is not None:
        selected_path = root / "selected" / f"{scene_id}{Path(selected_candidate['image_path']).suffix.lower()}"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_candidate["image_path"], selected_path)
        selected_rel = _relative(root, selected_path)

    semantic_dir = root / "semantic" / scene_id
    _json_write(semantic_dir / "scene.json", {
        "source": source.name,
        "live_qwen_conformance": "NOT_COMPLETE",
        "scene": public_semantic.get("scene") if public_semantic else None,
    })
    _json_write(semantic_dir / "candidate_judgments.json", {
        "source": source.name,
        "live_qwen_conformance": "NOT_COMPLETE",
        "candidates": [
            {"candidate_id": result["candidate_id"], "semantic": result.get("semantic"), "tm_only": (result.get("semantic") or {}).get("tm_only")}
            for result in public_results
        ],
    })
    _json_write(semantic_dir / "pairwise.json", {
        "source": source.name,
        "live_qwen_conformance": "NOT_COMPLETE",
        "tournament": public_ranking,
    })

    viz_dir = root / "viz"
    candidate_grid = viz_dir / f"{scene_id}_candidate_grid.jpg"
    ranking_grid = viz_dir / f"{scene_id}_ranking_grid.jpg"
    failure_grid = viz_dir / f"{scene_id}_failure_grid.jpg"
    _grid(candidate_grid, [(source, "Source")] + [
        (Path(item["image_path"]), str(item["candidate_id"])) for item in candidates
    ])
    rank_ids = []
    if public_ranking:
        winner = public_ranking.get("winner")
        if winner:
            rank_ids.append(winner)
        rank_ids.extend(candidate for candidate in public_ranking.get("equivalent_top_set", []) if candidate not in rank_ids)
    rank_ids.extend(result["candidate_id"] for result in public_results if result["candidate_id"] not in rank_ids)
    _grid(ranking_grid, [(source, "Source")] + [
        (Path(next(item["image_path"] for item in candidates if item["candidate_id"] == candidate_id)), f"#{index + 1} {candidate_id}")
        for index, candidate_id in enumerate(rank_ids)
    ])
    failures = [
        result for result in public_results
        if result.get("pgt_class") == "REJECT" or not (result.get("guards") or {}).get("passed", True)
    ]
    failure_items = [(Path(by_id[str(result["candidate_id"])]["image_path"]), f"FAIL {result['candidate_id']}") for result in failures]
    if not failure_items:
        failure_items = [(source, "No rejected candidates")]
    _grid(failure_grid, failure_items)

    return {
        "source": source.name,
        "selected": selected_id,
        "selected_image": selected_rel,
        **public_selection,
        "semantic_scene": public_semantic.get("scene") if public_semantic else None,
        "semantic_ranking": public_ranking,
        "ranking": rank_ids,
        "live_qwen_conformance": "NOT_COMPLETE",
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "family": item["manifest"]["family"],
                "parameters": item["manifest"]["parameters"],
                "source": item["manifest"]["source"],
            }
            for item in candidates
        ],
        "results": public_results,
        "artifacts": {
            "candidate_pool": _relative(root, copied_pool),
            "semantic": _relative(root, semantic_dir),
            "candidate_grid": _relative(root, candidate_grid),
            "ranking_grid": _relative(root, ranking_grid),
            "failure_grid": _relative(root, failure_grid),
        },
    }


def write_batch_report(output_root: str | Path, scenes: list[Mapping[str, Any]], *, report_path: str | Path | None = None) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "TM_PGT_IQA_V2",
        "live_qwen_conformance": "NOT_COMPLETE",
        "count": len(scenes),
        "certified": sum(scene.get("pgt_class") == "CERTIFIED_PGT" for scene in scenes),
        "usable": sum(scene.get("pgt_class") == "USABLE_PGT" for scene in scenes),
        "rejected": sum(scene.get("pgt_class") == "REJECT" for scene in scenes),
        "scenes": list(scenes),
    }
    target = Path(report_path) if report_path is not None else root / "report.json"
    _json_write(target, report)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "source", "selected", "pgt_class", "training_weight", "selection_confidence", "objective_score", "candidate_count",
        ])
        writer.writeheader()
        for scene in scenes:
            selected = next((result for result in scene.get("results", []) if result.get("candidate_id") == scene.get("selected")), {})
            writer.writerow({
                "source": scene.get("source"),
                "selected": scene.get("selected"),
                "pgt_class": scene.get("pgt_class"),
                "training_weight": scene.get("training_weight"),
                "selection_confidence": scene.get("selection_confidence"),
                "objective_score": (selected.get("metrics") or {}).get("overall", selected.get("quality_score")),
                "candidate_count": len(scene.get("results", [])),
            })
    return target
