"""Evaluate pretrained, Stage-1, and Stage-2 checkpoints on non-aligned references."""
from __future__ import annotations

import argparse
import gc
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

from .config import ExperimentGroup, load_experiment_config
from .data import EvaluationRecord, load_evaluation_manifest, load_record_images
from .decision import (
    DecisionThresholds,
    decide_stage1,
    decide_stage2_effectiveness,
    decide_stage2_necessity,
    recommend_data_expansion,
    recommend_stage2_variant,
)
from .metrics import compute_all_metrics, linear_to_srgb
from .model_loader import infer_rgb, load_model_from_spec, resolve_device
from .reporting import build_increment_rows, build_metric_rows, render_comparison_panel, write_reports
from .statistics import summarize_improvements


PRIMARY_COLOR_METRIC = "luminance_conditioned_cbcr_swd"
DIAGNOSTIC_METRICS = {"signed_ev_error"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple pretrained/Stage-1/Stage-2 Photofinishing checkpoints "
            "against same-scene non-pixel-aligned references."
        )
    )
    parser.add_argument("--config", required=True, help="JSON file defining checkpoint groups")
    parser.add_argument("--manifest", required=True, help="Extended same-scene evaluation CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input-mode", choices=["linear_srgb", "raw_metadata"], default="linear_srgb")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--min-win-rate", type=float, default=0.60)
    parser.add_argument(
        "--panel-limit",
        type=int,
        default=50,
        help="Maximum panels per experiment group; 0 saves every sample",
    )
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--output-bit-depth", choices=[8, 16], type=int, default=8)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("image-size must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if args.min_count <= 0:
        raise ValueError("min-count must be positive")
    if not 0.5 <= args.min_win_rate <= 1.0:
        raise ValueError("min-win-rate must be in [0.5, 1.0]")
    if args.panel_limit < 0:
        raise ValueError("panel-limit must be non-negative")


def evaluate_output_set(
    outputs: Mapping[str, np.ndarray],
    reference: np.ndarray,
    *,
    reference_repeat: Optional[np.ndarray] = None,
    output_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    input_semantic_masks: Optional[Mapping[str, np.ndarray]] = None,
    reference_semantic_masks: Optional[Mapping[str, np.ndarray]] = None,
) -> tuple[dict[str, dict[str, float]], Optional[dict[str, float]]]:
    """Computes all non-aligned metrics for precomputed model outputs."""

    required = {"pretrained", "stage1"}
    missing = required - set(outputs)
    if missing:
        raise ValueError(f"outputs missing required model labels: {sorted(missing)}")
    metrics = {
        name: compute_all_metrics(
            image,
            reference,
            output_mask=output_mask,
            reference_mask=reference_mask,
            input_semantic_masks=input_semantic_masks,
            reference_semantic_masks=reference_semantic_masks,
        )
        for name, image in outputs.items()
    }
    noise = None
    if reference_repeat is not None:
        noise = compute_all_metrics(
            reference_repeat,
            reference,
            output_mask=reference_mask,
            reference_mask=reference_mask,
            input_semantic_masks=reference_semantic_masks,
            reference_semantic_masks=reference_semantic_masks,
        )
    return metrics, noise


def _mean_median(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _summarize_model_metrics(
    model_sample_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for model_name, samples in model_sample_metrics.items():
        metric_names = sorted(set().union(*(metrics.keys() for metrics in samples.values()))) if samples else []
        result[model_name] = {
            metric: _mean_median([sample.get(metric, float("nan")) for sample in samples.values()])
            for metric in metric_names
        }
    return result


def _summarize_transitions(
    transition_sample_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for transition, samples in transition_sample_metrics.items():
        metric_names = sorted(set().union(*(metrics.keys() for metrics in samples.values()))) if samples else []
        result[transition] = {
            metric: summarize_improvements(
                np.asarray([sample.get(metric, float("nan")) for sample in samples.values()]),
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for metric in metric_names
            if metric not in DIAGNOSTIC_METRICS
        }
    return result


def _summarize_noise(noise_sample_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    metric_names = sorted(set().union(*(metrics.keys() for metrics in noise_sample_metrics.values()))) if noise_sample_metrics else []
    return {
        metric: _mean_median([sample.get(metric, float("nan")) for sample in noise_sample_metrics.values()])
        for metric in metric_names
    }


def _average_across_groups(
    grouped: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Averages seed/group results per sample before statistical testing."""

    accumulator: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for group_payload in grouped.values():
        for label, samples in group_payload.items():
            for sample_id, metrics in samples.items():
                for metric, value in metrics.items():
                    if math.isfinite(float(value)):
                        accumulator[label][sample_id][metric].append(float(value))
    averaged: dict[str, dict[str, dict[str, float]]] = {}
    for label, samples in accumulator.items():
        averaged[label] = {}
        for sample_id, metrics in samples.items():
            averaged[label][sample_id] = {
                metric: float(np.mean(values)) for metric, values in metrics.items() if values
            }
    return averaged


def _slice_failures(
    sample_improvements: Mapping[str, Mapping[str, float]],
    records_by_id: Mapping[str, EvaluationRecord],
    *,
    metric: str = PRIMARY_COLOR_METRIC,
) -> dict[str, dict[str, float | int]]:
    by_tag: dict[str, list[float]] = defaultdict(list)
    for sample_id, metrics in sample_improvements.items():
        value = float(metrics.get(metric, float("nan")))
        if not math.isfinite(value):
            continue
        tags = records_by_id[sample_id].scene_tags or ("untagged",)
        for tag in tags:
            by_tag[tag].append(value)
    return {
        tag: {
            "count": len(values),
            "median": float(np.median(values)),
            "negative_rate": float(np.mean(np.asarray(values) < 0)),
        }
        for tag, values in sorted(by_tag.items())
    }


def _save_rgb(path: Path, image: np.ndarray, bit_depth: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if bit_depth == 8:
        encoded = np.rint(clipped * 255.0).astype(np.uint8)
    elif bit_depth == 16:
        encoded = np.rint(clipped * 65535.0).astype(np.uint16)
    else:
        raise ValueError("bit_depth must be 8 or 16")
    if not cv2.imwrite(str(path), cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to save output image: {path}")


def _load_group_models(group: ExperimentGroup, device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, object]]:
    models: dict[str, torch.nn.Module] = {}
    metadata: dict[str, object] = {}
    pretrained, pretrained_meta = load_model_from_spec(group.pretrained, role="pretrained", device=device)
    stage1, stage1_meta = load_model_from_spec(group.stage1, role="stage1", device=device)
    models["pretrained"] = pretrained
    models["stage1"] = stage1
    metadata["pretrained"] = pretrained_meta
    metadata["stage1"] = stage1_meta
    for stage2_spec in group.stage2:
        model, model_meta = load_model_from_spec(stage2_spec, role="stage2", device=device)
        label = f"stage2/{stage2_spec.name}"
        models[label] = model
        metadata[label] = model_meta
    return models, metadata


def _group_decisions(
    *,
    transition_summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
    model_sample_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    noise_sample_metrics: Mapping[str, Mapping[str, float]],
    stage2_names: Sequence[str],
    thresholds: DecisionThresholds,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    stage1_summary = transition_summaries.get("pretrained_to_stage1", {})
    stage1_decision = decide_stage1(stage1_summary, thresholds)
    stage1_color = np.asarray([
        metrics.get(PRIMARY_COLOR_METRIC, float("nan"))
        for metrics in model_sample_metrics.get("stage1", {}).values()
    ])
    noise = None
    if noise_sample_metrics:
        noise = np.asarray([
            noise_sample_metrics.get(sample_id, {}).get(PRIMARY_COLOR_METRIC, float("nan"))
            for sample_id in model_sample_metrics.get("stage1", {})
        ])
    necessity = decide_stage2_necessity(
        stage1_color,
        noise,
        thresholds,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    stage2_decisions = {
        name: decide_stage2_effectiveness(
            transition_summaries.get(f"stage1_to_{name}", {}), thresholds
        )
        for name in stage2_names
    }
    return {
        "stage1": stage1_decision,
        "stage2_necessity": necessity,
        "stage2": stage2_decisions,
        "recommended_stage2": recommend_stage2_variant(stage2_decisions),
    }


def _metric_definitions() -> dict[str, str]:
    return {
        "signed_ev_error": "Signed output-minus-reference median log-luminance bias in EV; diagnostic only.",
        "absolute_ev_error": "Absolute median brightness error in EV.",
        "log_luma_quantile_mae": "Mean error over p01/p05/p10/p25/p50/p75/p90/p95/p99 log-luminance quantiles.",
        "tone_shape_mae": "Quantile-curve error after removing median exposure; measures tone shape rather than simple brightening.",
        "log_luma_w1": "One-dimensional Wasserstein distance between log-luminance distributions.",
        "shadow_ratio_error": "Absolute difference in pixels below linear luminance 0.03.",
        "highlight_ratio_error": "Absolute difference in pixels above linear luminance 0.95.",
        "clipping_ratio_error": "Absolute difference in near-0/near-1 RGB clipping ratio.",
        "cbcr_swd": "Sliced Wasserstein distance between complete CbCr distributions.",
        "luminance_conditioned_cbcr_swd": "Mean CbCr SWD in shadow, midtone, and highlight bands; Stage-2 primary metric.",
        "chroma_mean_error": "Euclidean distance between mean CbCr vectors.",
        "chroma_covariance_error": "Frobenius distance between CbCr covariance matrices.",
        "saturation_w1": "Wasserstein distance between CIELAB chroma-magnitude distributions.",
        "neutral_axis_error": "Difference between mean a*b* of low-chroma pixels.",
        "semantic_*_lab_swd": "Optional non-aligned Lab a*b* SWD inside separately supplied semantic masks.",
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    config = load_experiment_config(args.config)
    records = load_evaluation_manifest(args.manifest, split=args.split)
    records_by_id = {record.sample_id: record for record in records}
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = DecisionThresholds(min_count=args.min_count, min_win_rate=args.min_win_rate)

    all_metric_rows: list[dict[str, object]] = []
    all_increment_rows: list[dict[str, object]] = []
    group_model_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    group_transition_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    group_noise_metrics: dict[str, dict[str, dict[str, float]]] = {}
    model_metadata: dict[str, object] = {}
    per_group_decisions: dict[str, object] = {}
    per_group_slice_failures: dict[str, object] = {}

    for group in config.groups:
        models, metadata = _load_group_models(group, device)
        model_metadata[group.name] = metadata
        sample_model_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        sample_transition_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        sample_noise_metrics: dict[str, dict[str, float]] = {}
        panel_count = 0

        for record in records:
            loaded = load_record_images(record, image_size=args.image_size, input_mode=args.input_mode)
            input_image = loaded["input"]
            reference = loaded["reference"]
            repeat = loaded["reference_repeat"]
            outputs = {label: infer_rgb(model, input_image, device) for label, model in models.items()}
            metrics, noise = evaluate_output_set(
                outputs,
                reference,
                reference_repeat=repeat,
                output_mask=loaded["input_valid_mask"],
                reference_mask=loaded["reference_valid_mask"],
                input_semantic_masks=loaded["input_semantic_masks"],
                reference_semantic_masks=loaded["reference_semantic_masks"],
            )
            for model_name, values in metrics.items():
                sample_model_metrics[model_name][record.sample_id] = values
            if noise is not None:
                sample_noise_metrics[record.sample_id] = noise

            stage2_metrics = {
                spec.name: metrics[f"stage2/{spec.name}"] for spec in group.stage2
            }
            metric_rows = build_metric_rows(group.name, record.sample_id, record.scene_tags, metrics)
            increment_rows = build_increment_rows(
                group.name,
                record.sample_id,
                record.scene_tags,
                metrics["pretrained"],
                metrics["stage1"],
                stage2_metrics,
            )
            all_metric_rows.extend(metric_rows)
            all_increment_rows.extend(increment_rows)
            for row in increment_rows:
                transition = str(row["transition"])
                sample_transition_metrics[transition][record.sample_id] = {
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"group", "sample_id", "scene_tags", "transition"}
                }

            should_save_panel = not args.no_panels and (args.panel_limit == 0 or panel_count < args.panel_limit)
            if should_save_panel:
                panel_images = {
                    "Input": linear_to_srgb(input_image),
                    "Pretrained": outputs["pretrained"],
                    "Stage1": outputs["stage1"],
                }
                for spec in group.stage2:
                    panel_images[f"Stage2/{spec.name}"] = outputs[f"stage2/{spec.name}"]
                panel_images["Reference"] = reference
                if repeat is not None:
                    panel_images["ReferenceRepeat"] = repeat
                render_comparison_panel(
                    panel_images,
                    output_dir / "panels" / group.name / f"{record.sample_id}.jpg",
                )
                panel_count += 1

            if args.save_outputs:
                for label, image in outputs.items():
                    safe_label = label.replace("/", "_")
                    _save_rgb(
                        output_dir / "outputs" / group.name / safe_label / f"{record.sample_id}.png",
                        image,
                        args.output_bit_depth,
                    )

        group_model_metrics[group.name] = dict(sample_model_metrics)
        group_transition_metrics[group.name] = dict(sample_transition_metrics)
        group_noise_metrics[group.name] = sample_noise_metrics
        transition_summary = _summarize_transitions(
            sample_transition_metrics,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        stage2_names = [spec.name for spec in group.stage2]
        per_group_decisions[group.name] = _group_decisions(
            transition_summaries=transition_summary,
            model_sample_metrics=sample_model_metrics,
            noise_sample_metrics=sample_noise_metrics,
            stage2_names=stage2_names,
            thresholds=thresholds,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        per_group_slice_failures[group.name] = {
            name: _slice_failures(
                sample_transition_metrics.get(f"stage1_to_{name}", {}), records_by_id
            )
            for name in stage2_names
        }

        del models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cross_model_metrics = _average_across_groups(group_model_metrics)
    cross_transition_metrics = _average_across_groups(group_transition_metrics)
    cross_noise_candidates = _average_across_groups({
        group: {"noise": samples} for group, samples in group_noise_metrics.items()
    })
    cross_noise_metrics = cross_noise_candidates.get("noise", {})
    cross_transition_summary = _summarize_transitions(
        cross_transition_metrics,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    stage2_names = sorted(
        transition.removeprefix("stage1_to_")
        for transition in cross_transition_metrics
        if transition.startswith("stage1_to_")
    )
    cross_decisions = _group_decisions(
        transition_summaries=cross_transition_summary,
        model_sample_metrics=cross_model_metrics,
        noise_sample_metrics=cross_noise_metrics,
        stage2_names=stage2_names,
        thresholds=thresholds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    cross_slices = {
        name: _slice_failures(cross_transition_metrics.get(f"stage1_to_{name}", {}), records_by_id)
        for name in stage2_names
    }
    group_primary_medians = {
        f"{group}/{name}": float(
            per_group_decisions[group]["stage2"][name].get("primary", {}).get("median", float("nan"))
        )
        for group in per_group_decisions
        for name in per_group_decisions[group]["stage2"]
    }
    recommended = cross_decisions["recommended_stage2"].get("recommended")
    selected_slice_failures = cross_slices.get(str(recommended), {}) if recommended else {
        tag: {
            "count": max(int(payload.get("count", 0)) for payload in variants),
            "negative_rate": max(float(payload.get("negative_rate", 0.0)) for payload in variants),
            "median": min(float(payload.get("median", 0.0)) for payload in variants),
        }
        for tag in sorted(set().union(*(slices.keys() for slices in cross_slices.values())))
        for variants in [[slices[tag] for slices in cross_slices.values() if tag in slices]]
    }
    data_expansion = recommend_data_expansion(
        sample_count=len(records),
        stage2_necessity=cross_decisions["stage2_necessity"],
        stage2_decisions=cross_decisions["stage2"],
        group_primary_medians=group_primary_medians,
        slice_failures=selected_slice_failures,
        thresholds=thresholds,
    )

    summary = {
        "schema_version": 1,
        "split": args.split,
        "num_unique_scenes": len(records),
        "num_experiment_groups": len(config.groups),
        "metric_direction": "lower_is_better; incremental improvement = earlier distance - later distance",
        "primary_stage2_metric": PRIMARY_COLOR_METRIC,
        "metric_definitions": _metric_definitions(),
        "model_metadata": model_metadata,
        "groups": {
            group.name: {
                "model_distances": _summarize_model_metrics(group_model_metrics[group.name]),
                "incremental_improvements": _summarize_transitions(
                    group_transition_metrics[group.name],
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                ),
                "reference_noise_floor": _summarize_noise(group_noise_metrics[group.name]),
                "slice_failures": per_group_slice_failures[group.name],
            }
            for group in config.groups
        },
        "cross_group": {
            "model_distances": _summarize_model_metrics(cross_model_metrics),
            "incremental_improvements": cross_transition_summary,
            "reference_noise_floor": _summarize_noise(cross_noise_metrics),
            "slice_failures": cross_slices,
        },
    }
    decisions = {
        "thresholds": {
            "min_count": thresholds.min_count,
            "min_win_rate": thresholds.min_win_rate,
            "primary_stage2_metric": PRIMARY_COLOR_METRIC,
        },
        "groups": per_group_decisions,
        "cross_group": cross_decisions,
        "data_expansion": data_expansion,
    }
    write_reports(
        output_dir,
        metric_rows=all_metric_rows,
        increment_rows=all_increment_rows,
        summary=summary,
        decisions=decisions,
    )
    print(f"Evaluation complete: {output_dir}")
    print(f"Stage 1: {cross_decisions['stage1']['status']}")
    print(f"Stage 2 necessity: {cross_decisions['stage2_necessity']['status']}")
    print(f"Recommended Stage 2: {cross_decisions['recommended_stage2']['recommended']}")
    print(f"Expand evaluation data: {data_expansion['expand_evaluation_data']}")
    print(f"Expand training data: {data_expansion['expand_training_data']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
