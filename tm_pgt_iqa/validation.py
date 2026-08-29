"""Offline human-ranking validation for TM PGT IQA V2.

The helpers intentionally consume compact annotations rather than prescribe a
labeling UI.  Annotation JSON is either a list of scene records or
``{"scenes": [...]}``, where each record contains ``source``, a ranked list
under ``ranking`` (or ``human_ranking``), and optional ``accepted`` for
certified-precision measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _scenes(value: Any) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("scenes", [])
    if not isinstance(value, list):
        raise ValueError("annotations and reports must contain a scene list")
    return [dict(item) for item in value]


def load_human_annotations(path: str | Path) -> dict:
    return {"scenes": _scenes(json.loads(Path(path).read_text(encoding="utf-8")))}


def kendall_tau(predicted: Sequence[str], human: Sequence[str]) -> float | None:
    """Kendall tau-a over candidates ranked by both sides; return None if N<2."""
    human_order = {value: index for index, value in enumerate(human)}
    common = [value for value in predicted if value in human_order]
    if len(common) < 2:
        return None
    pred_order = {value: index for index, value in enumerate(common)}
    concordant = discordant = 0
    for index, first in enumerate(common):
        for second in common[index + 1:]:
            pred_sign = pred_order[first] - pred_order[second]
            human_sign = human_order[first] - human_order[second]
            if pred_sign * human_sign > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else None


def _prediction_ranking(scene: Mapping[str, Any]) -> list[str]:
    if isinstance(scene.get("ranking"), list):
        return [str(value) for value in scene["ranking"]]
    tournament = scene.get("semantic_ranking") or {}
    ranking = []
    if tournament.get("winner"):
        ranking.append(str(tournament["winner"]))
    ranking.extend(str(value) for value in tournament.get("equivalent_top_set", []) if str(value) not in ranking)
    ranking.extend(
        str(item.get("candidate_id"))
        for item in scene.get("results", [])
        if item.get("candidate_id") is not None and str(item["candidate_id"]) not in ranking
    )
    if not ranking and scene.get("selected"):
        ranking.append(str(scene["selected"]))
    return ranking


def evaluate_human_annotations(report: Mapping[str, Any], annotations: Mapping[str, Any]) -> dict:
    """Return the required Kendall tau, Top-2 accuracy and certified precision."""
    scene_by_source = {str(scene.get("source")): scene for scene in _scenes(report)}
    taus: list[float] = []
    top2_hits: list[bool] = []
    certified: list[bool] = []
    unmatched: list[str] = []
    for annotation in _scenes(annotations):
        source = str(annotation.get("source"))
        scene = scene_by_source.get(source)
        if scene is None:
            unmatched.append(source)
            continue
        human = annotation.get("ranking", annotation.get("human_ranking", []))
        if not isinstance(human, list) or not human:
            continue
        human = [str(value) for value in human]
        predicted = _prediction_ranking(scene)
        tau = kendall_tau(predicted, human)
        if tau is not None:
            taus.append(tau)
        selected = str(scene.get("selected")) if scene.get("selected") is not None else None
        if selected is not None:
            top2_hits.append(selected in human[:2])
        if scene.get("pgt_class") == "CERTIFIED_PGT" and "accepted" in annotation:
            certified.append(bool(annotation["accepted"]))
    return {
        "annotated_scenes": len(taus) if taus else len(top2_hits),
        "kendall_tau": sum(taus) / len(taus) if taus else None,
        "top2_accuracy": sum(top2_hits) / len(top2_hits) if top2_hits else None,
        "certified_precision": sum(certified) / len(certified) if certified else None,
        "certified_count": len(certified),
        "unmatched_sources": unmatched,
        "live_qwen_conformance": "NOT_COMPLETE",
    }


def ablation_template() -> dict:
    """A schema-only plan for the mandatory Objective/Qwen and family ablations."""
    return {
        "live_qwen_conformance": "NOT_COMPLETE",
        "experiments": [
            {"name": "A_objective_only", "qwen": "off", "families": "full"},
            {"name": "B_objective_plus_scene", "qwen": "scene", "families": "full"},
            {"name": "C_objective_plus_naturalness", "qwen": "naturalness", "families": "full"},
            {"name": "D_objective_plus_pairwise", "qwen": "pairwise", "families": "full"},
            {"name": "E_full_system", "qwen": "full", "families": "full"},
            {"name": "retinex_only", "qwen": "off", "families": ["retinex"]},
            {"name": "retinex_local", "qwen": "off", "families": ["retinex", "local_face_tm"]},
            {"name": "retinex_local_shape", "qwen": "off", "families": ["retinex", "local_face_tm", "tone_shape"]},
            {"name": "retinex_qwen", "qwen": "full", "families": ["retinex", "qwen_edit"]},
            {"name": "full_candidate_pool", "qwen": "full", "families": "full"},
        ],
        "metrics": ["kendall_tau", "top2_accuracy", "certified_precision", "pgt_usable_rate"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score TM PGT outputs against human ranking annotations.")
    parser.add_argument("--report", help="Batch report.json produced by tm_pgt_iqa")
    parser.add_argument("--annotations", help="Human-ranking JSON annotations")
    parser.add_argument("--output", help="Validation JSON output")
    parser.add_argument("--write-ablation-template", default=None, help="Write an ablation experiment template and exit")
    args = parser.parse_args(argv)
    if args.write_ablation_template:
        Path(args.write_ablation_template).write_text(json.dumps(ablation_template(), indent=2) + "\n", encoding="utf-8")
        return 0
    if not (args.report and args.annotations and args.output):
        parser.error("--report, --annotations and --output are required unless writing the ablation template")
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    metrics = evaluate_human_annotations(report, load_human_annotations(args.annotations))
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
