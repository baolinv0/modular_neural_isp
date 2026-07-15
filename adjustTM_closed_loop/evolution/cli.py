from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .adjudication import AdjudicationConfig, adjudicate_evolution
from .candidate_selection import CandidateSelector, SelectionConfig, load_candidate_scores
from .policy import fit_policy_from_manifest
from .teacher_manifest import build_teacher_manifest


def _write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _merge_scores(args: argparse.Namespace) -> int:
    from .score_merge import merge_candidate_scores

    report = merge_candidate_scores(
        args.inference_records, args.iqa_scores, args.output, method=args.method
    )
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".summary.json")
    _write_json(report_path, report)
    print(args.output)
    return 0


def _select_teachers(args: argparse.Namespace) -> int:
    candidates = load_candidate_scores(args.scores)
    selector = CandidateSelector(SelectionConfig(
        min_improvement=args.min_improvement,
        min_confidence=args.min_confidence,
        allow_boundary=args.allow_boundary,
        boundary_weight=args.boundary_weight,
        score_tolerance=args.score_tolerance,
        improvement_scale=args.improvement_scale,
    ))
    selections = selector.select_all(candidates)
    if args.teacher_evaluator_id:
        selections = [
            replace(item, metadata={**dict(item.metadata), "teacher_evaluator_id": args.teacher_evaluator_id})
            for item in selections
        ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "selections.jsonl", [item.to_dict() for item in selections])
    manifest_path = build_teacher_manifest(
        selections,
        input_dir=args.input_dir,
        output_path=output_dir / "teacher_manifest.jsonl",
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        validate_output_paths=not args.skip_output_validation,
    )
    improved = [item for item in selections if item.status == "improved"]
    anchors = [item for item in selections if item.status != "improved"]
    summary = {
        "version": 1,
        "candidate_count": len(candidates),
        "scene_count": len(selections),
        "improved_scene_count": len(improved),
        "baseline_anchor_count": len(anchors),
        "improved_rate": len(improved) / max(len(selections), 1),
        "mean_selected_alpha": sum(item.selected_alpha for item in selections) / max(len(selections), 1),
        "mean_improvement": sum(item.score_delta for item in improved) / max(len(improved), 1),
        "teacher_manifest": str(manifest_path),
        "selection_config": {
            "min_improvement": args.min_improvement,
            "min_confidence": args.min_confidence,
            "allow_boundary": args.allow_boundary,
            "boundary_weight": args.boundary_weight,
            "score_tolerance": args.score_tolerance,
            "improvement_scale": args.improvement_scale,
            "teacher_evaluator_id": args.teacher_evaluator_id,
        },
    }
    _write_json(output_dir / "selection_summary.json", summary)
    print(output_dir / "selection_summary.json")
    return 0


def _fit_policy(args: argparse.Namespace) -> int:
    policy, report = fit_policy_from_manifest(
        args.teacher_manifest,
        ridge=args.ridge,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        ood_threshold=args.ood_threshold,
    )
    policy.save(args.output_policy)
    report.update({
        "policy": str(Path(args.output_policy)),
        "teacher_manifest": str(Path(args.teacher_manifest)),
        "ridge": args.ridge,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "ood_threshold": args.ood_threshold,
    })
    _write_json(args.output_report, report)
    print(args.output_report)
    return 0


def _render_policy(args: argparse.Namespace) -> int:
    from .rendering import render_policy_outputs

    rows = render_policy_outputs(
        teacher_manifest=args.teacher_manifest,
        policy_path=args.policy,
        methods_config=args.methods_config,
        method_name=args.method_name,
        output_dir=args.output_dir,
        device=args.device,
        max_side=args.max_side if args.max_side > 0 else None,
        multiple=args.multiple,
    )
    summary = {
        "version": 1,
        "scene_count": len(rows),
        "ood_fallback_count": sum(not row["in_domain"] for row in rows),
        "mean_abs_predicted_alpha": sum(abs(row["predicted_alpha"]) for row in rows) / max(len(rows), 1),
        "records": str(Path(args.output_dir) / "render_records.jsonl"),
    }
    _write_json(Path(args.output_dir) / "render_summary.json", summary)
    print(Path(args.output_dir) / "render_summary.json")
    return 0


def _distill_baseline(args: argparse.Namespace) -> int:
    from .baseline_distillation import (
        BaselineDistillationConfig,
        DistillationLossConfig,
        distill_fixed_baseline,
    )

    report = distill_fixed_baseline(BaselineDistillationConfig(
        teacher_manifest=Path(args.teacher_manifest),
        baseline_checkpoint=Path(args.baseline_checkpoint),
        output_dir=Path(args.output_dir),
        train_modules=tuple(args.train_modules.split(",")),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        parameter_anchor_weight=args.parameter_anchor_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        loss=DistillationLossConfig(
            lambda_luminance=args.lambda_luminance,
            lambda_gradient=args.lambda_gradient,
            lambda_chroma=args.lambda_chroma,
        ),
    ))
    print(report["checkpoint"])
    return 0


def _adjudicate(args: argparse.Namespace) -> int:
    report = adjudicate_evolution(
        args.teacher_manifest,
        args.baseline_scores,
        args.student_scores,
        AdjudicationConfig(
            min_target_mean_delta=args.min_target_mean_delta,
            min_target_win_rate=args.min_target_win_rate,
            min_overall_mean_delta=args.min_overall_mean_delta,
            max_anchor_mean_regression=args.max_anchor_mean_regression,
            max_anchor_regression_rate=args.max_anchor_regression_rate,
            regression_epsilon=args.regression_epsilon,
            reject_new_hard_failures=not args.allow_new_hard_failures,
            evaluation_splits=tuple(item.strip() for item in args.evaluation_splits.split(",") if item.strip()),
            require_anchor_scenes=not args.allow_no_anchors,
            require_independent_evaluator=not args.allow_self_evaluation,
        ),
    )
    _write_json(args.output, report)
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tm-baseline-evolution",
        description="IQA-selected controllable-TM teacher distillation for one fixed automatic baseline",
    )
    sub = parser.add_subparsers(dest="command", required=True)


    merge = sub.add_parser("merge-scores", help="Join generated candidate records with candidate-level IQA scores")
    merge.add_argument("--inference-records", required=True)
    merge.add_argument("--iqa-scores", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--method")
    merge.add_argument("--report")
    merge.set_defaults(handler=_merge_scores)

    select = sub.add_parser("select-teachers", help="Select one safe IQA teacher candidate per scene")
    select.add_argument("--scores", required=True, help="Candidate-level IQA JSON or JSONL")
    select.add_argument("--input-dir", required=True, help="16-bit linear RGB scene inputs")
    select.add_argument("--output-dir", required=True)
    select.add_argument("--min-improvement", type=float, default=0.03)
    select.add_argument("--min-confidence", type=float, default=0.70)
    select.add_argument("--allow-boundary", action=argparse.BooleanOptionalAction, default=True)
    select.add_argument("--boundary-weight", type=float, default=0.35)
    select.add_argument("--score-tolerance", type=float, default=1e-6)
    select.add_argument("--improvement-scale", type=float, default=0.15)
    select.add_argument("--split-seed", type=int, default=42)
    select.add_argument("--validation-fraction", type=float, default=0.15)
    select.add_argument("--test-fraction", type=float, default=0.15)
    select.add_argument("--skip-output-validation", action="store_true")
    select.add_argument("--teacher-evaluator-id")
    select.set_defaults(handler=_select_teachers)

    fit = sub.add_parser("fit-policy", help="Distill teacher alpha choices into an automatic scene policy")
    fit.add_argument("--teacher-manifest", required=True)
    fit.add_argument("--output-policy", required=True)
    fit.add_argument("--output-report", required=True)
    fit.add_argument("--ridge", type=float, default=1e-2)
    fit.add_argument("--alpha-min", type=float, default=-1.0)
    fit.add_argument("--alpha-max", type=float, default=1.0)
    fit.add_argument("--ood-threshold", type=float, default=6.0)
    fit.set_defaults(handler=_fit_policy)

    render = sub.add_parser("render-policy", help="Render automatic-policy and exact-baseline outputs")
    render.add_argument("--teacher-manifest", required=True)
    render.add_argument("--policy", required=True)
    render.add_argument("--methods-config", required=True)
    render.add_argument("--method-name", required=True)
    render.add_argument("--output-dir", required=True)
    render.add_argument("--device", default="cpu")
    render.add_argument("--max-side", type=int, default=512)
    render.add_argument("--multiple", type=int, default=16)
    render.set_defaults(handler=_render_policy)

    distill = sub.add_parser("distill-baseline", help="Fine-tune one fixed Gain/GTM baseline checkpoint")
    distill.add_argument("--teacher-manifest", required=True)
    distill.add_argument("--baseline-checkpoint", required=True)
    distill.add_argument("--output-dir", required=True)
    distill.add_argument("--train-modules", default="gain,gtm")
    distill.add_argument("--image-size", type=int, default=512)
    distill.add_argument("--batch-size", type=int, default=4)
    distill.add_argument("--epochs", type=int, default=10)
    distill.add_argument("--learning-rate", type=float, default=1e-5)
    distill.add_argument("--weight-decay", type=float, default=1e-6)
    distill.add_argument("--parameter-anchor-weight", type=float, default=1e-5)
    distill.add_argument("--num-workers", type=int, default=0)
    distill.add_argument("--seed", type=int, default=42)
    distill.add_argument("--device", default="cpu")
    distill.add_argument("--amp", action="store_true")
    distill.add_argument("--lambda-luminance", type=float, default=1.0)
    distill.add_argument("--lambda-gradient", type=float, default=0.2)
    distill.add_argument("--lambda-chroma", type=float, default=0.1)
    distill.set_defaults(handler=_distill_baseline)

    adjudicate = sub.add_parser("adjudicate", help="Accept only target improvement without regression or new defects")
    adjudicate.add_argument("--teacher-manifest", required=True)
    adjudicate.add_argument("--baseline-scores", required=True)
    adjudicate.add_argument("--student-scores", required=True)
    adjudicate.add_argument("--output", required=True)
    adjudicate.add_argument("--min-target-mean-delta", type=float, default=0.03)
    adjudicate.add_argument("--min-target-win-rate", type=float, default=0.50)
    adjudicate.add_argument("--min-overall-mean-delta", type=float, default=0.0)
    adjudicate.add_argument("--max-anchor-mean-regression", type=float, default=0.01)
    adjudicate.add_argument("--max-anchor-regression-rate", type=float, default=0.10)
    adjudicate.add_argument("--regression-epsilon", type=float, default=0.01)
    adjudicate.add_argument("--allow-new-hard-failures", action="store_true")
    adjudicate.add_argument("--evaluation-splits", default="test")
    adjudicate.add_argument("--allow-no-anchors", action="store_true")
    adjudicate.add_argument("--allow-self-evaluation", action="store_true")
    adjudicate.set_defaults(handler=_adjudicate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
