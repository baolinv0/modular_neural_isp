from __future__ import annotations

import argparse
import json
from pathlib import Path

from .human_study import build_balanced_trials, materialize_blinded_assets, render_study_html, select_stratified_scenes
from .schemas import read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic blinded intent/naturalness human-study trials")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", default=["a_m100", "a_m050", "a_p050", "a_p100"])
    parser.add_argument("--scene-count", type=int, default=32)
    parser.add_argument("--candidates-per-trial", type=int, default=4)
    parser.add_argument("--blocks-per-scene-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    scenes = select_stratified_scenes(manifest["scenes"], count=args.scene_count, seed=args.seed)
    scene_lookup = {scene["scene_id"]: scene for scene in manifest["scenes"]}
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_mapping = {}
    for study_type in ("intent_match", "naturalness"):
        trials, mapping = build_balanced_trials(
            scene_ids=scenes, levels=args.levels, methods=args.methods,
            candidates_per_trial=args.candidates_per_trial,
            blocks_per_scene_level=args.blocks_per_scene_level,
            seed=args.seed + (0 if study_type == "intent_match" else 1), study_type=study_type,
        )
        for trial in trials:
            scene = scene_lookup[trial["scene_id"]]
            trial["center_image"] = scene["gt"]["a_000"]["path"]
            if study_type == "intent_match":
                trial["target_image"] = scene["gt"][trial["level"]]["path"]
            for candidate in trial["candidates"]:
                method = mapping[trial["trial_id"]][candidate["candidate_id"]]
                candidate["image"] = str(output_root / method / trial["level"] / trial["scene_id"])
        trials = materialize_blinded_assets(trials, output_dir, mode=args.asset_mode)
        with (output_dir / f"{study_type}_trials.jsonl").open("w", encoding="utf-8") as handle:
            for trial in trials:
                handle.write(json.dumps(trial, ensure_ascii=False) + "\n")
        combined_mapping.update(mapping)
        (output_dir / f"{study_type}.html").write_text(
            render_study_html(trials, title=f"adjustTM {study_type}"), encoding="utf-8"
        )
    write_json_atomic(output_dir / "blinded_method_map.json", combined_mapping)


if __name__ == "__main__":
    main()
