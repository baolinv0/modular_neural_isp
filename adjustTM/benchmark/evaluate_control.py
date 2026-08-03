from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import control_curve_metrics


def _read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def evaluate_dense_records(records, *, dead_zone_threshold: float, jump_ratio_threshold: float):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["scene_id"])].append(record)
    results = []
    for (method, scene_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["alpha"]))
        metrics = control_curve_metrics(
            [row["alpha"] for row in rows], [row["mean_log_luma"] for row in rows],
            dead_zone_threshold=dead_zone_threshold, jump_ratio_threshold=jump_ratio_threshold,
        )
        results.append({"method": method, "scene_id": scene_id, "metrics": metrics})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GT-independent dense brightness-control behavior")
    parser.add_argument("--dense-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dead-zone-threshold", type=float, default=0.002)
    parser.add_argument("--jump-ratio-threshold", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_dense_records(
        _read_jsonl(args.dense_records), dead_zone_threshold=args.dead_zone_threshold,
        jump_ratio_threshold=args.jump_ratio_threshold
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
