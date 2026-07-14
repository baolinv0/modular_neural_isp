from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .case_assets import automatic_crop_boxes, materialize_gallery_assets
from .case_render import _CASE_TITLES, _render_index, render_case_page, render_scene_browser
from .case_selection import read_jsonl, select_case_sets


def build_case_gallery(
    *,
    manifest: Mapping[str, Any],
    output_root: str | Path,
    reference_records: Sequence[Mapping[str, Any]],
    dense_records: Sequence[Mapping[str, Any]],
    control_records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    levels: Sequence[str],
    focus_method: str,
    comparison_baseline: str,
    output_dir: str | Path,
    vlm_records: Sequence[Mapping[str, Any]] | None = None,
    representative_count: int = 6,
    best_count: int = 6,
    failure_count: int = 6,
    disagreement_count: int = 6,
    asset_mode: str = "copy",
    crop_fraction: float = 0.25,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_sets = select_case_sets(
        reference_records, control_records,
        focus_method=focus_method, comparison_baseline=comparison_baseline,
        representative_count=representative_count, best_count=best_count,
        failure_count=failure_count, disagreement_count=disagreement_count,
        vlm_records=vlm_records,
    )
    selected_scenes = list(dict.fromkeys(scene for scenes in case_sets.values() for scene in scenes))
    asset_index = materialize_gallery_assets(
        manifest=manifest, output_root=output_root, methods=methods, levels=levels,
        output_dir=output_dir, selected_scenes=selected_scenes,
        asset_mode=asset_mode, crop_fraction=crop_fraction,
    )
    outputs: dict[str, str] = {}
    index_path = output_dir / "index.html"
    index_path.write_text(_render_index(case_sets, focus_method=focus_method, comparison_baseline=comparison_baseline), encoding="utf-8")
    outputs["index"] = str(index_path)
    for key, title in _CASE_TITLES.items():
        path = output_dir / f"{key}.html"
        path.write_text(render_case_page(
            title=title, scene_ids=case_sets.get(key, []), asset_index=asset_index,
            reference_records=reference_records, dense_records=dense_records,
            methods=methods, levels=levels,
        ), encoding="utf-8")
        outputs[key] = str(path)
    browser_path = output_dir / "scene_browser.html"
    browser_path.write_text(render_scene_browser(asset_index), encoding="utf-8")
    outputs["browser"] = str(browser_path)
    case_index_path = output_dir / "case_index.json"
    case_index_path.write_text(json.dumps({
        "focus_method": focus_method, "comparison_baseline": comparison_baseline,
        "methods": list(methods), "levels": list(levels), "case_sets": case_sets,
        "crop_fraction": crop_fraction,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["case_index"] = str(case_index_path)
    return outputs


__all__ = [
    "automatic_crop_boxes", "build_case_gallery", "materialize_gallery_assets",
    "read_jsonl", "select_case_sets",
]
