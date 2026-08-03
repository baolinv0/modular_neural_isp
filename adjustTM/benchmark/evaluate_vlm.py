from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

from adjustTM.constants import LEVELS
from .schemas import read_json
from .vlm import INTENT_SCORES, NATURALNESS_SCORES, aggregate_repeats, build_prompt


def build_tasks(manifest, output_root: str | Path, methods, levels, kinds):
    output_root = Path(output_root)
    alpha_map = dict(LEVELS)
    tasks = []
    for scene in manifest["scenes"]:
        scene_id = scene["scene_id"]
        center = scene["gt"]["a_000"]["path"]
        for level in levels:
            for method in methods:
                candidate = output_root / method / level / scene_id
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                for kind in kinds:
                    prompt = build_prompt(kind, alpha=alpha_map[level])
                    task = {
                        "task_id": f"{kind}:{scene_id}:{level}:{method}",
                        "kind": kind,
                        "scene_id": scene_id,
                        "level": level,
                        "method": method,
                        "alpha": alpha_map[level],
                        "prompt": prompt,
                        "required_scores": list(INTENT_SCORES if kind == "intent" else NATURALNESS_SCORES),
                        "response_contract": {"score_range": [1, 5], "confidence_range": [0, 1]},
                        "images": {"center": center, "candidate": str(candidate)},
                    }
                    if kind == "intent":
                        task["images"]["target"] = scene["gt"][level]["path"]
                    tasks.append(task)
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or execute separate VLM intent/naturalness evaluations")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", default=["a_m100", "a_m050", "a_p050", "a_p100"])
    parser.add_argument("--kinds", nargs="+", choices=["intent", "naturalness"], default=["intent", "naturalness"])
    parser.add_argument("--tasks-output", required=True)
    parser.add_argument("--backend-command", help="Executable reading one task JSON from stdin and returning one response JSON")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--responses-output")
    parser.add_argument("--aggregate-output")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = build_tasks(read_json(args.manifest), args.output_root, args.methods, args.levels, args.kinds)
    task_path = Path(args.tasks_output)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    with task_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    if not args.backend_command:
        print(f"exported {len(tasks)} VLM tasks")
        return
    responses = []
    command = shlex.split(args.backend_command)
    for task in tasks:
        for repeat in range(args.repeats):
            last_error = None
            for attempt in range(args.max_retries + 1):
                try:
                    completed = subprocess.run(command, input=json.dumps(task), capture_output=True, text=True, check=True)
                    response = json.loads(completed.stdout)
                    required = INTENT_SCORES if task["kind"] == "intent" else NATURALNESS_SCORES
                    aggregate_repeats([response], required_scores=required)
                    responses.append({
                        "task_id": task["task_id"], "repeat": repeat, "attempt": attempt,
                        "response": response, "kind": task["kind"], "raw_stdout": completed.stdout,
                    })
                    last_error = None
                    break
                except Exception as error:
                    last_error = error
            if last_error is not None:
                raise RuntimeError(f"VLM backend failed for {task['task_id']} repeat={repeat}") from last_error
    response_path = Path(args.responses_output or task_path.with_name("vlm_responses.jsonl"))
    with response_path.open("w", encoding="utf-8") as handle:
        for response in responses:
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
    grouped = defaultdict(list)
    kinds = {}
    for row in responses:
        grouped[row["task_id"]].append(row["response"])
        kinds[row["task_id"]] = row["kind"]
    aggregates = []
    for task_id, rows in grouped.items():
        required = INTENT_SCORES if kinds[task_id] == "intent" else NATURALNESS_SCORES
        aggregates.append({"task_id": task_id, "kind": kinds[task_id], **aggregate_repeats(rows, required_scores=required)})
    aggregate_path = Path(args.aggregate_output or task_path.with_name("vlm_aggregate.jsonl"))
    with aggregate_path.open("w", encoding="utf-8") as handle:
        for row in aggregates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
