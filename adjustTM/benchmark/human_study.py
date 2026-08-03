from __future__ import annotations

import hashlib
import html
import json
import math
import random
import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def select_stratified_scenes(scenes: Sequence[Mapping[str, object]], *, count: int, seed: int) -> list[str]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    tagged: dict[str, list[str]] = defaultdict(list)
    untagged: list[str] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        tags = [str(tag) for tag in scene.get("tags", [])]
        if not tags:
            untagged.append(scene_id)
        for tag in tags:
            tagged[tag].append(scene_id)
    for values in tagged.values():
        rng.shuffle(values)
    rng.shuffle(untagged)
    tags = sorted(tagged)
    rng.shuffle(tags)
    selected: list[str] = []
    while len(selected) < min(count, len(scenes)):
        progressed = False
        for tag in tags:
            while tagged[tag] and tagged[tag][0] in selected:
                tagged[tag].pop(0)
            if tagged[tag] and len(selected) < count:
                selected.append(tagged[tag].pop(0))
                progressed = True
        if not progressed:
            break
    remaining = [str(scene["scene_id"]) for scene in scenes if str(scene["scene_id"]) not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def _materialize_file(source: str | Path, destination: Path, mode: str) -> None:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unknown asset mode: {mode}")


def materialize_blinded_assets(
    trials: Sequence[Mapping[str, object]],
    study_root: str | Path,
    *,
    mode: str = "copy",
) -> list[dict[str, object]]:
    study_root = Path(study_root)
    output = json.loads(json.dumps(list(trials)))
    reference_map: dict[str, str] = {}
    for trial in output:
        for field, prefix in (("center_image", "center"), ("target_image", "target")):
            source = trial.get(field)
            if not source:
                continue
            source_key = str(Path(source).resolve())
            if source_key not in reference_map:
                suffix = Path(source).suffix.lower() or ".png"
                digest = hashlib.sha256(source_key.encode()).hexdigest()[:16]
                relative = f"assets/{prefix}_{digest}{suffix}"
                _materialize_file(source, study_root / relative, mode)
                reference_map[source_key] = relative
            trial[field] = reference_map[source_key]
        for candidate in trial["candidates"]:
            source = candidate["image"]
            suffix = Path(source).suffix.lower() or ".png"
            relative = f"assets/{candidate['blind_asset_id']}{suffix}"
            _materialize_file(source, study_root / relative, mode)
            candidate["image"] = relative
    return output


def _blind_id(seed: int, trial_id: str, method: str) -> str:
    digest = hashlib.sha256(f"{seed}:{trial_id}:{method}".encode()).hexdigest()
    return digest[:12]


def build_balanced_trials(
    *,
    scene_ids: Sequence[str],
    levels: Sequence[str],
    methods: Sequence[str],
    candidates_per_trial: int,
    blocks_per_scene_level: int,
    seed: int,
    study_type: str,
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    if not (2 <= candidates_per_trial <= len(methods)):
        raise ValueError("candidates_per_trial must be between 2 and method count")
    if study_type not in {"intent_match", "naturalness"}:
        raise ValueError("Unknown study type")
    rng = random.Random(seed)
    counts = {method: 0 for method in methods}
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    trials: list[dict[str, object]] = []
    mapping: dict[str, dict[str, str]] = {}
    for scene_id in sorted(scene_ids):
        for level in levels:
            for block in range(blocks_per_scene_level):
                trial_id = f"{study_type}_{len(trials):06d}"
                available = list(methods)
                rng.shuffle(available)
                selected: list[str] = []
                while len(selected) < candidates_per_trial:
                    candidates = [method for method in available if method not in selected]
                    candidates.sort(key=lambda method: (
                        counts[method],
                        sum(pair_counts[tuple(sorted((method, existing)))] for existing in selected),
                        rng.random(),
                    ))
                    selected.append(candidates[0])
                rng.shuffle(selected)
                trial_map: dict[str, str] = {}
                candidates_payload = []
                for position, method in enumerate(selected, start=1):
                    candidate_id = f"C{position}"
                    blind = _blind_id(seed, trial_id, method)
                    trial_map[candidate_id] = method
                    candidates_payload.append({"candidate_id": candidate_id, "blind_asset_id": blind})
                    counts[method] += 1
                for i, first in enumerate(selected):
                    for second in selected[i + 1:]:
                        pair_counts[tuple(sorted((first, second)))] += 1
                trials.append({
                    "trial_id": trial_id,
                    "study_type": study_type,
                    "scene_id": scene_id,
                    "level": level,
                    "block": block,
                    "candidates": candidates_payload,
                })
                mapping[trial_id] = trial_map
    return trials, mapping


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def response_quality_control(
    responses: Iterable[Mapping[str, object]],
    *,
    attention_threshold: float = 0.8,
    repeat_consistency_threshold: float = 0.6,
    minimum_seconds: float = 1.0,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for response in responses:
        grouped[str(response["rater_id"])].append(response)
    results: dict[str, dict[str, object]] = {}
    for rater_id, rows in grouped.items():
        attention = [_to_bool(row.get("attention_correct")) for row in rows if "attention_correct" in row]
        repeat = [_to_bool(row.get("repeat_consistent")) for row in rows if "repeat_consistent" in row]
        times = [float(row.get("duration_seconds", 0.0)) for row in rows]
        attention_rate = float(np.mean(attention)) if attention else 1.0
        repeat_rate = float(np.mean(repeat)) if repeat else 1.0
        fast_rate = float(np.mean(np.asarray(times) < minimum_seconds)) if times else 0.0
        include = attention_rate >= attention_threshold and repeat_rate >= repeat_consistency_threshold and fast_rate <= 0.5
        results[rater_id] = {
            "attention_rate": attention_rate,
            "repeat_consistency_rate": repeat_rate,
            "fast_response_rate": fast_rate,
            "included": bool(include),
        }
    return results


def responses_to_pairwise(
    responses: Iterable[Mapping[str, object]],
    method_map: Mapping[str, Mapping[str, str]],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for response in responses:
        trial_id = str(response["trial_id"])
        mapping = method_map[trial_id]
        best = mapping[str(response["best_candidate_id"])]
        worst_id = response.get("worst_candidate_id")
        for method in mapping.values():
            if method != best:
                pairs.append((best, method))
        if worst_id is not None:
            worst = mapping[str(worst_id)]
            for method in mapping.values():
                if method != worst:
                    pairs.append((method, worst))
    return pairs


def bradley_terry_scores(pairwise_wins: Iterable[tuple[str, str]], *, max_iterations: int = 10000, tolerance: float = 1e-10) -> dict[str, float]:
    pairs = list(pairwise_wins)
    methods = sorted({method for pair in pairs for method in pair})
    if not methods:
        return {}
    index = {method: i for i, method in enumerate(methods)}
    wins = np.zeros(len(methods), dtype=np.float64)
    games = np.zeros((len(methods), len(methods)), dtype=np.float64)
    for winner, loser in pairs:
        i, j = index[winner], index[loser]
        wins[i] += 1.0
        games[i, j] += 1.0
        games[j, i] += 1.0
    ability = np.ones(len(methods), dtype=np.float64)
    for _ in range(max_iterations):
        denominator = np.zeros_like(ability)
        for i in range(len(methods)):
            for j in range(len(methods)):
                if i != j and games[i, j] > 0:
                    denominator[i] += games[i, j] / max(ability[i] + ability[j], 1e-12)
        updated = np.where(denominator > 0, np.maximum(wins, 1e-6) / denominator, ability)
        updated /= np.exp(np.mean(np.log(np.maximum(updated, 1e-12))))
        if np.max(np.abs(np.log(updated) - np.log(ability))) < tolerance:
            ability = updated
            break
        ability = updated
    logits = np.log(np.maximum(ability, 1e-12))
    logits -= logits.mean()
    return {method: float(logits[index[method]]) for method in methods}


def render_study_html(trials: Sequence[Mapping[str, object]], *, title: str) -> str:
    payload = json.dumps(list(trials), ensure_ascii=False).replace("</", "<\\/")
    safe_title = html.escape(title)
    return f"""<!doctype html>
<meta charset=\"utf-8\"><title>{safe_title}</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7}} .panel{{background:white;padding:18px;border-radius:8px;max-width:1500px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}} img{{max-width:100%;max-height:380px;object-fit:contain;background:#111}}
.candidate{{border:2px solid #ddd;padding:8px}} label{{display:block;margin:4px 0}} button{{padding:10px 18px;margin:8px}}
</style>
<div class=\"panel\"><h1>{safe_title}</h1><label>Rater ID <input id=\"rater\"></label><div id=\"trial\"></div>
<button onclick=\"previous()\">Previous</button><button onclick=\"saveAndNext()\">Save & Next</button><button onclick=\"downloadCSV()\">Download CSV</button></div>
<script>
const trials={payload}; let index=0; const answers={{}}; let started=Date.now();
function esc(s){{return String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));}}
function render(){{const t=trials[index]; started=Date.now(); let refs=`<h2>${{index+1}} / ${{trials.length}} · ${{esc(t.study_type)}} · ${{esc(t.level)}}</h2><div class=grid><div><b>Center</b><br><img src=\"${{esc(t.center_image)}}\"></div>`;
if(t.target_image) refs+=`<div><b>Target</b><br><img src=\"${{esc(t.target_image)}}\"></div>`; refs+='</div><h3>Anonymous candidates</h3><div class=grid>';
for(const c of t.candidates) refs+=`<div class=candidate><b>${{esc(c.candidate_id)}}</b><br><img src=\"${{esc(c.image)}}\"><label><input type=radio name=best_candidate_id value=\"${{esc(c.candidate_id)}}\"> Best</label><label><input type=radio name=worst_candidate_id value=\"${{esc(c.candidate_id)}}\"> Worst</label></div>`;
refs+='</div>'; document.getElementById('trial').innerHTML=refs; const a=answers[t.trial_id]; if(a){{for(const k of ['best_candidate_id','worst_candidate_id']){{const e=document.querySelector(`input[name=${{k}}][value=\"${{a[k]}}\"]`); if(e)e.checked=true;}}}}}}
function chosen(name){{const e=document.querySelector(`input[name=${{name}}]:checked`);return e?e.value:'';}}
function save(){{const t=trials[index],best=chosen('best_candidate_id'),worst=chosen('worst_candidate_id');if(!best||!worst||best===worst){{alert('Choose distinct best and worst candidates');return false;}}answers[t.trial_id]={{trial_id:t.trial_id,study_type:t.study_type,scene_id:t.scene_id,level:t.level,rater_id:document.getElementById('rater').value,best_candidate_id:best,worst_candidate_id:worst,duration_seconds:(Date.now()-started)/1000}};return true;}}
function saveAndNext(){{if(!save())return;if(index<trials.length-1)index++;render();}} function previous(){{if(index>0)index--;render();}}
function downloadCSV(){{if(!answers[trials[index].trial_id])save();const rows=Object.values(answers),keys=['trial_id','study_type','scene_id','level','rater_id','best_candidate_id','worst_candidate_id','duration_seconds'];let csv=keys.join(',')+'\\n'+rows.map(r=>keys.map(k=>JSON.stringify(r[k]??'')).join(',')).join('\\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='human_responses.csv';a.click();}}
render();
</script>"""
