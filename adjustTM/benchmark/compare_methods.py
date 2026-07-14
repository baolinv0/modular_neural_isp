from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .schemas import read_json, write_json_atomic
from .statistics import bootstrap_mean_ci, worst_cvar, paired_comparison, holm_adjust


def _read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(reference_records, control_records, main_methods, diagnostic_methods, comparison_baselines=None):
    summary = {"main_methods": {}, "diagnostic_methods": {}, "comparisons": {}}
    reference_scene = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in reference_records:
        scene_id = str(row["scene_id"])
        for metric, value in row["metrics"].items():
            if metric in {"pred_mean_log_luma", "target_mean_log_luma"}:
                continue
            key = f"{row['semantic_group']}.{metric}"
            reference_scene[row["method"]][key][scene_id].append(float(value))
    control_scene = defaultdict(lambda: defaultdict(dict))
    for row in control_records:
        scene_id = str(row["scene_id"])
        for metric, value in row["metrics"].items():
            control_scene[row["method"]][metric][scene_id] = float(value)

    reference_maps = defaultdict(dict)
    control_maps = defaultdict(dict)
    for method, keyed in reference_scene.items():
        for key, scene_rows in keyed.items():
            reference_maps[method][key] = {scene_id: float(np.mean(values)) for scene_id, values in scene_rows.items()}
    for method, keyed in control_scene.items():
        for key, scene_rows in keyed.items():
            control_maps[method][key] = dict(scene_rows)

    for method in list(main_methods) + list(diagnostic_methods):
        payload = {"reference": {}, "control": {}}
        for key, scene_map in reference_maps.get(method, {}).items():
            values = [value for _, value in sorted(scene_map.items())]
            low, high = bootstrap_mean_ci(values, samples=2000, seed=42)
            lower_is_better = not any(token in key for token in ("psnr", "ssim", "spearman"))
            payload["reference"][key] = {
                "scene_count": len(values),
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "ci_low": low, "ci_high": high,
                "worst_5pct_cvar": worst_cvar(values, higher_is_better=not lower_is_better),
            }
        for key, scene_map in control_maps.get(method, {}).items():
            values = [value for _, value in sorted(scene_map.items())]
            low, high = bootstrap_mean_ci(values, samples=2000, seed=42)
            higher_is_better = key in {"strict_scene_pass", "negative_range", "positive_range", "total_range", "range_balance"}
            payload["control"][key] = {
                "scene_count": len(values),
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "ci_low": low, "ci_high": high,
                "worst_5pct_cvar": worst_cvar(values, higher_is_better=higher_is_better),
            }
        target = "main_methods" if method in main_methods else "diagnostic_methods"
        summary[target][method] = payload

    baselines = list(comparison_baselines or [
        name for name in ("frozen_baseline", "exposure_global", "gamma_global") if name in main_methods
    ])
    raw_pvalues = {}
    for method in main_methods:
        if method in baselines:
            continue
        method_comparisons = {}
        for baseline in baselines:
            comparison_key = f"vs_{baseline}"
            metric_results = {}
            for key in sorted(set(reference_maps.get(method, {})) & set(reference_maps.get(baseline, {}))):
                lower_is_better = not any(token in key for token in ("psnr", "ssim", "spearman"))
                result = paired_comparison(
                    reference_maps[method][key], reference_maps[baseline][key],
                    lower_is_better=lower_is_better, seed=42, bootstrap_samples=2000, permutation_samples=5000,
                )
                full_key = f"reference.{key}"
                metric_results[full_key] = result
                raw_pvalues[f"{method}|{comparison_key}|{full_key}"] = result["permutation_p"]
            for key in sorted(set(control_maps.get(method, {})) & set(control_maps.get(baseline, {}))):
                higher_is_better = key in {"strict_scene_pass", "negative_range", "positive_range", "total_range", "range_balance"}
                result = paired_comparison(
                    control_maps[method][key], control_maps[baseline][key],
                    lower_is_better=not higher_is_better, seed=42, bootstrap_samples=2000, permutation_samples=5000,
                )
                full_key = f"control.{key}"
                metric_results[full_key] = result
                raw_pvalues[f"{method}|{comparison_key}|{full_key}"] = result["permutation_p"]
            method_comparisons[comparison_key] = metric_results
        summary["comparisons"][method] = method_comparisons
    adjusted = holm_adjust(raw_pvalues) if raw_pvalues else {}
    for joined, value in adjusted.items():
        method, comparison_key, metric_key = joined.split("|", 2)
        summary["comparisons"][method][comparison_key][metric_key]["holm_adjusted_p"] = value
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate scene-level adjustTM benchmark metrics")
    parser.add_argument("--reference-records", required=True)
    parser.add_argument("--control-records", required=True)
    parser.add_argument("--main-methods", nargs="+", required=True)
    parser.add_argument("--diagnostic-methods", nargs="*", default=[])
    parser.add_argument("--comparison-baselines", nargs="*", default=None)
    parser.add_argument("--protocol")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(
        _read_jsonl(args.reference_records), _read_jsonl(args.control_records),
        args.main_methods, args.diagnostic_methods, comparison_baselines=args.comparison_baselines
    )
    result["protocol"] = read_json(args.protocol) if args.protocol else {}
    write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
