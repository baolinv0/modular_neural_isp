from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

_CASE_TITLES = {
    "representative_cases": "Representative cases",
    "best_improvements": "Where learning helps",
    "failure_cases": "Failure cases",
    "metric_vlm_disagreement": "Metric–perception disagreement",
}


def _reference_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result = {}
    for row in records:
        result[(str(row["scene_id"]), str(row["method"]), str(row["level"]))] = row.get(
            "metrics", {}
        )
    return result


def _target_curve_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in records:
        metrics = row.get("metrics", {})
        if "target_mean_log_luma" not in metrics or "alpha" not in row:
            continue
        scene_id = str(row["scene_id"])
        level = str(row["level"])
        grouped[scene_id].setdefault(
            level,
            (float(row["alpha"]), float(metrics["target_mean_log_luma"])),
        )
    return {
        scene_id: sorted(values.values())
        for scene_id, values in grouped.items()
    }


def _dense_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    result: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        result[(str(row["scene_id"]), str(row["method"]))].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: float(row["alpha"]))
    return result


def _metric_badges(metrics: Mapping[str, Any]) -> str:
    keys = (
        "log_luma_mae",
        "rgb_ssim",
        "lpips",
        "clip_ratio",
        "deep_shadow_ratio",
    )
    badges = []
    for key in keys:
        if key in metrics:
            value = float(metrics[key])
            badges.append(
                f"<span class='badge'>{html.escape(key)}={value:.4g}</span>"
            )
    return "".join(badges)


def _svg_line_chart(
    *,
    title: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    width: int = 720,
    height: int = 260,
) -> str:
    all_points = [
        (x, y)
        for points in series.values()
        for x, y in points
        if math.isfinite(x) and math.isfinite(y)
    ]
    if not all_points:
        return ""
    min_x, max_x = min(x for x, _ in all_points), max(x for x, _ in all_points)
    min_y, max_y = min(y for _, y in all_points), max(y for _, y in all_points)
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        padding = max(1e-6, abs(max_y) * 0.05 + 0.01)
        min_y -= padding
        max_y += padding
    pad_left, pad_top, pad_right, pad_bottom = 58, 28, 18, 42
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = pad_left + (x - min_x) / (max_x - min_x) * plot_width
        py = pad_top + (max_y - y) / (max_y - min_y) * plot_height
        return px, py

    palette = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#475569",
        "#be123c",
    )
    lines = []
    legend = []
    for index, (name, points) in enumerate(series.items()):
        projected = [
            project(point)
            for point in points
            if math.isfinite(point[0]) and math.isfinite(point[1])
        ]
        if not projected:
            continue
        color = palette[index % len(palette)]
        path = " ".join(
            ("M" if point_index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for point_index, (x, y) in enumerate(projected)
        )
        circles = "".join(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.5' fill='{color}'/>"
            for x, y in projected
        )
        lines.append(
            f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>{circles}"
        )
        legend.append(
            f"<span><i style='background:{color}'></i>{html.escape(name)}</span>"
        )
    grid = []
    labels = []
    for tick in range(5):
        fraction = tick / 4
        y = pad_top + fraction * plot_height
        value = max_y - fraction * (max_y - min_y)
        grid.append(
            f"<line x1='{pad_left}' y1='{y:.2f}' x2='{width-pad_right}' y2='{y:.2f}' stroke='#e5e7eb'/>"
        )
        labels.append(
            f"<text x='{pad_left-8}' y='{y+4:.2f}' text-anchor='end' font-size='11'>{value:.3g}</text>"
        )
    for tick in range(5):
        fraction = tick / 4
        x = pad_left + fraction * plot_width
        value = min_x + fraction * (max_x - min_x)
        labels.append(
            f"<text x='{x:.2f}' y='{height-15}' text-anchor='middle' font-size='11'>{value:.2g}</text>"
        )
    return (
        f"<div class='chart'><h4>{html.escape(title)}</h4>"
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        + "".join(grid)
        + f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height-pad_bottom}' stroke='#374151'/>"
        + f"<line x1='{pad_left}' y1='{height-pad_bottom}' x2='{width-pad_right}' y2='{height-pad_bottom}' stroke='#374151'/>"
        + "".join(labels)
        + "".join(lines)
        + "</svg><div class='legend'>"
        + "".join(legend)
        + "</div></div>"
    )


def _card(label: str, image_path: str, badges: str = "") -> str:
    return (
        "<figure class='image-card'>"
        f"<figcaption>{html.escape(label)}</figcaption>"
        f"<a href='{html.escape(image_path)}'>"
        f"<img loading='lazy' src='{html.escape(image_path)}' alt='{html.escape(label)}'></a>"
        f"<div class='badges'>{badges}</div></figure>"
    )


def _base_style() -> str:
    return """
<style>
:root{font-family:Inter,Arial,sans-serif;color:#172033;background:#f5f7fb}body{margin:0}.page{max-width:1500px;margin:auto;padding:28px}
a{color:#1d4ed8}.nav{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0 24px}.case{background:white;border:1px solid #dbe2ef;border-radius:14px;padding:22px;margin:0 0 28px;box-shadow:0 4px 18px #0f172a0d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.trajectory{display:flex;gap:8px;overflow-x:auto;padding:8px 0}.trajectory .image-card{min-width:150px}.image-card{margin:0;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:8px}.image-card img{display:block;width:100%;height:180px;object-fit:contain;background:#111;border-radius:6px}.trajectory img,.crop-grid img{height:135px}.image-card figcaption{font-weight:650;margin-bottom:7px;word-break:break-word}.badge{display:inline-block;background:#e8eefc;border-radius:999px;padding:3px 7px;margin:5px 4px 0 0;font-size:11px}.section{margin-top:22px}.level-block{border-top:1px solid #e5e7eb;padding-top:12px;margin-top:12px}.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}.chart{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:8px}.chart svg{width:100%;height:auto}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}.legend i{display:inline-block;width:12px;height:3px;margin-right:5px;vertical-align:middle}.tags span{display:inline-block;background:#dcfce7;padding:4px 8px;border-radius:999px;margin-right:6px}.crop-group{border-top:1px dashed #cbd5e1;margin-top:14px;padding-top:12px}.crop-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.summary-card{display:inline-block;min-width:220px;background:#fff;border:1px solid #dbe2ef;border-radius:12px;padding:16px;margin:8px}.browser-controls{position:sticky;top:0;background:#f5f7fb;padding:12px 0;z-index:2;display:flex;gap:12px;flex-wrap:wrap}select{padding:8px;border-radius:8px;border:1px solid #cbd5e1}
</style>
"""


def _nav() -> str:
    return (
        "<nav class='nav'><a href='index.html'>Overview</a>"
        "<a href='representative_cases.html'>Representative</a>"
        "<a href='best_improvements.html'>Learning helps</a>"
        "<a href='failure_cases.html'>Failures</a>"
        "<a href='metric_vlm_disagreement.html'>Metric–VLM disagreement</a>"
        "<a href='scene_browser.html'>Full scene browser</a></nav>"
    )


def _render_scene_case(
    scene_id: str,
    *,
    asset_index: Mapping[str, Any],
    reference_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    dense_lookup: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    target_curve_lookup: Mapping[str, Sequence[tuple[float, float]]],
    methods: Sequence[str],
    levels: Sequence[str],
) -> str:
    scene = asset_index["scenes"][scene_id]
    tags = "".join(
        f"<span>{html.escape(tag)}</span>" for tag in scene.get("tags", [])
    )
    parts = [
        f"<section class='case'><h2>{html.escape(scene_id)}</h2>"
        f"<div class='tags'>{tags}</div>"
    ]
    parts.append("<div class='section'><h3>Cross-method comparison</h3>")
    for level in levels:
        parts.append(
            f"<div class='level-block'><h4>{html.escape(level)}</h4><div class='grid'>"
        )
        parts.append(_card(f"Target · {level}", scene["target"][level]))
        for method in methods:
            metrics = reference_lookup.get((scene_id, method, level), {})
            parts.append(
                _card(
                    f"{method} · {level}",
                    scene["methods"][method][level],
                    _metric_badges(metrics),
                )
            )
        parts.append("</div></div>")
    parts.append("</div>")

    parts.append("<div class='section'><h3>Nine-level trajectory</h3>")
    parts.append("<h4>Target trajectory</h4><div class='trajectory'>")
    for level in levels:
        parts.append(_card(level, scene["target"][level]))
    parts.append("</div>")
    for method in methods:
        parts.append(f"<h4>{html.escape(method)}</h4><div class='trajectory'>")
        for level in levels:
            parts.append(
                _card(
                    level,
                    scene["methods"][method][level],
                    _metric_badges(
                        reference_lookup.get((scene_id, method, level), {})
                    ),
                )
            )
        parts.append("</div>")
    parts.append("</div>")

    if scene.get("crops"):
        parts.append("<div class='section'><h3>Local crops</h3>")
        crop_levels = list(
            dict.fromkeys([levels[0], levels[len(levels) // 2], levels[-1]])
        )
        for crop_name, crop_entry in scene["crops"].items():
            parts.append(
                f"<div class='crop-group'><h4>{html.escape(crop_name)}</h4>"
            )
            for level in crop_levels:
                parts.append(
                    f"<h5>{html.escape(level)}</h5><div class='crop-grid'>"
                )
                parts.append(
                    _card(f"Target · {level}", crop_entry["target"][level])
                )
                for method in methods:
                    parts.append(
                        _card(
                            f"{method} · {level}",
                            crop_entry["methods"][method][level],
                        )
                    )
                parts.append("</div>")
            parts.append("</div>")
        parts.append("</div>")

    chart_specs = [
        ("mean_log_luma", "Mean log luminance"),
        ("clip_ratio", "Clipping ratio"),
        ("deep_shadow_ratio", "Deep-shadow ratio"),
        ("chroma_rg_drift_from_zero", "Chroma drift from alpha zero"),
    ]
    parts.append("<div class='section'><h3>Control curves</h3><div class='chart-grid'>")
    for key, title in chart_specs:
        series: dict[str, Sequence[tuple[float, float]]] = {}
        if key == "mean_log_luma" and target_curve_lookup.get(scene_id):
            series["Target (nine-level)"] = list(target_curve_lookup[scene_id])
        for method in methods:
            points = [
                (float(row["alpha"]), float(row[key]))
                for row in dense_lookup.get((scene_id, method), [])
                if key in row
            ]
            if points:
                series[method] = points
        parts.append(_svg_line_chart(title=title, series=series))
    parts.append("</div></div></section>")
    return "".join(parts)


def render_case_page(
    *,
    title: str,
    scene_ids: Sequence[str],
    asset_index: Mapping[str, Any],
    reference_records: Sequence[Mapping[str, Any]],
    dense_records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    levels: Sequence[str],
) -> str:
    reference_lookup = _reference_index(reference_records)
    dense_lookup = _dense_index(dense_records)
    target_curve_lookup = _target_curve_index(reference_records)
    cases = "".join(
        _render_scene_case(
            scene_id,
            asset_index=asset_index,
            reference_lookup=reference_lookup,
            dense_lookup=dense_lookup,
            target_curve_lookup=target_curve_lookup,
            methods=methods,
            levels=levels,
        )
        for scene_id in scene_ids
    )
    if not cases:
        cases = "<p>No cases available for this category.</p>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>{_base_style()}</head>"
        f"<body><main class='page'><h1>{html.escape(title)}</h1>"
        f"{_nav()}{cases}</main></body></html>"
    )


def render_scene_browser(asset_index: Mapping[str, Any]) -> str:
    data = json.dumps(asset_index, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Full scene browser</title>{_base_style()}</head><body><main class='page'><h1>Full scene browser</h1>{_nav()}
<div class='browser-controls'><label>Scene <select id='scene'></select></label><label>Level <select id='level'></select></label><label>Trajectory method <select id='method'></select></label></div>
<div id='comparison'></div><div id='trajectory'></div></main><script>
const DATA={data};
const sceneSelect=document.getElementById('scene'),levelSelect=document.getElementById('level'),methodSelect=document.getElementById('method');
Object.keys(DATA.scenes).forEach(v=>sceneSelect.add(new Option(v,v))); DATA.levels.forEach(v=>levelSelect.add(new Option(v,v))); DATA.methods.forEach(v=>methodSelect.add(new Option(v,v)));
function card(label,path){{return `<figure class="image-card"><figcaption>${{label}}</figcaption><a href="${{path}}"><img src="${{path}}"></a></figure>`}}
function render(){{const s=DATA.scenes[sceneSelect.value],l=levelSelect.value,m=methodSelect.value;let comparison=`<section class="case"><h2>${{sceneSelect.value}} · ${{l}}</h2><div class="grid">${{card('Target',s.target[l])}}`;DATA.methods.forEach(x=>comparison+=card(x,s.methods[x][l]));comparison+='</div></section>';document.getElementById('comparison').innerHTML=comparison;let trajectory=`<section class="case"><h2>${{m}} · full trajectory</h2><div class="trajectory">`;DATA.levels.forEach(x=>trajectory+=card(x,s.methods[m][x]));trajectory+='</div></section>';document.getElementById('trajectory').innerHTML=trajectory;}}
[sceneSelect,levelSelect,methodSelect].forEach(x=>x.addEventListener('change',render));render();
</script></body></html>"""


def _render_index(
    case_sets: Mapping[str, Sequence[str]],
    *,
    focus_method: str,
    comparison_baseline: str,
) -> str:
    cards = []
    for key, title in _CASE_TITLES.items():
        scenes = case_sets.get(key, [])
        scene_preview = ", ".join(html.escape(scene) for scene in scenes[:5]) or "No cases"
        cards.append(
            f"<a class='summary-card' href='{key}.html'><h2>{html.escape(title)}</h2>"
            f"<strong>{len(scenes)} scenes</strong><p>{scene_preview}</p></a>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>adjustTM visual evidence</title>{_base_style()}</head>"
        "<body><main class='page'><h1>adjustTM Visual Evidence</h1>"
        f"<p>Focus method: <strong>{html.escape(focus_method)}</strong>; "
        f"comparison baseline: <strong>{html.escape(comparison_baseline)}</strong>.</p>"
        f"{_nav()}<div>{''.join(cards)}"
        "<a class='summary-card' href='scene_browser.html'><h2>Full scene browser</h2>"
        "<p>Inspect every scene, level and method.</p></a></div>"
        "</main></body></html>"
    )
