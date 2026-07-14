from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .human_study import bradley_terry_scores, response_quality_control, responses_to_pairwise
from .schemas import read_json, write_json_atomic


def _read_responses(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC and analyze blinded human-study responses")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--method-map", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses = _read_responses(args.responses)
    mapping = read_json(args.method_map)
    qc = response_quality_control(responses)
    included = [row for row in responses if qc[str(row["rater_id"])]["included"]]
    grouped = defaultdict(list)
    for row in included:
        grouped[str(row.get("study_type", "unknown"))].append(row)
    rankings = {}
    for study_type, rows in grouped.items():
        rankings[study_type] = bradley_terry_scores(responses_to_pairwise(rows, mapping))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "rater_qc.json", qc)
    write_json_atomic(output_dir / "rankings.json", rankings)
    write_json_atomic(output_dir / "analysis_summary.json", {"response_count": len(responses), "included_count": len(included)})


if __name__ == "__main__":
    main()
