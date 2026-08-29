"""Batch entrypoint for the TM PGT IQA V2 production pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .candidate_generation.generate_candidates import CandidateManifest
from .config import load_config
from .evaluator import TMPGTEvaluator
from .reporting import write_batch_report, write_scene_artifacts


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"not a supported image: {path}")
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    yield from (child for child in sorted(path.iterdir()) if child.suffix.lower() in IMAGE_EXTS)


def _resolve_mask(image: Path, masks: Path, explicit: Path | None = None) -> Path:
    root = explicit if explicit is not None else masks
    if root.is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(root)
    for candidate in (root / image.name, root / f"{image.stem}_mask.png", root / f"{image.stem}.png"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no semantic label map found for {image.name} under {root}")


def _pool_for_source(candidates_root: Path, source: Path) -> Path:
    nested = candidates_root / source.stem
    return nested if nested.is_dir() else candidates_root


def _find_candidate_image(manifest_path: Path, candidate_id: str) -> Path:
    # Adjacent image wins; it lets a manifest be named independently while
    # keeping the candidate directory self-contained.
    for extension in IMAGE_EXTS:
        sibling = manifest_path.with_suffix(extension)
        if sibling.exists():
            return sibling
    for extension in IMAGE_EXTS:
        by_id = manifest_path.parent / f"{candidate_id}{extension}"
        if by_id.exists():
            return by_id
    raise FileNotFoundError(f"manifest {manifest_path} has no candidate image")


def _discover_candidates(pool: Path, *, strict_manifest: bool, expected_source: Path | None) -> list[dict]:
    if not pool.is_dir():
        raise FileNotFoundError(pool)
    records: list[dict] = []
    for manifest_path in sorted(pool.glob("*.json")):
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if strict_manifest:
            manifest = CandidateManifest.from_dict(value)
            if expected_source is not None and manifest.source != expected_source.name:
                raise ValueError(
                    f"manifest {manifest_path} belongs to {manifest.source!r}, not source {expected_source.name!r}"
                )
        else:
            # Legacy ``--images`` accepted minimal sidecars. Production
            # ``--candidates`` always takes the strict authoritative path.
            image_guess = _find_candidate_image(manifest_path, manifest_path.stem)
            manifest = CandidateManifest(
                candidate_id=str(value.get("candidate_id", image_guess.stem)),
                family=str(value.get("family", "unknown")),
                parameters=dict(value.get("parameters", {})),
                source=str(value.get("source", expected_source.name if expected_source else "legacy")),
            )
        image = _find_candidate_image(manifest_path, manifest.candidate_id)
        records.append({
            "candidate_id": manifest.candidate_id,
            "image_path": str(image),
            "manifest_path": str(manifest_path),
            "manifest": manifest.to_dict(),
        })
    if not records:
        raise FileNotFoundError(f"no candidate manifests found in {pool}")
    ids = [record["candidate_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate candidate IDs in {pool}")
    return records


def _output_paths(value: str) -> tuple[Path, Path]:
    requested = Path(value)
    if requested.suffix.lower() == ".json":
        return requested.parent, requested
    return requested, requested / "report.json"


def _run_scene(
    evaluator: TMPGTEvaluator,
    *,
    source_path: Path,
    source_mask: Path,
    candidates: list[dict],
    no_vlm: bool,
) -> tuple[list, dict | None, dict | None, dict]:
    mask_paths: dict[str, str] = {}
    results = []
    for record in candidates:
        image_path = Path(record["image_path"])
        # A TM candidate inherits Source semantics. Evaluator-side resizing
        # handles candidate output dimensions without rewriting segmentation.
        mask_paths[str(image_path)] = str(source_mask)
        results.append(evaluator.evaluate_one(
            image_path,
            source_mask,
            run_vlm=False,  # source-aware calls are batched below
            source_path=source_path,
            family=str(record["manifest"]["family"]),
        ))
    evaluator.apply_pool_consistency(results)
    semantic = ranking = None
    if not no_vlm and evaluator.semantic_judge is not None:
        semantic = evaluator.review_semantic_topk(results, source_path, source_mask, mask_paths)
        ranking = evaluator.rank_candidates(results, source_path, mask_paths)
    return results, semantic, ranking, evaluator.finalize_selection(results, ranking)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch front-portrait TM pseudo-GT evaluator and selector.")
    parser.add_argument("--source", default=None, help="Source image or source directory (required for V2 production mode)")
    parser.add_argument("--candidates", default=None, help="Candidate root with authoritative JSON manifests; <root>/<source-stem>/ is supported")
    parser.add_argument("--images", default=None, help="Legacy alias for one candidate pool; accepts legacy minimal sidecars")
    parser.add_argument("--masks", required=True, help="Source label map or directory")
    parser.add_argument("--source-mask", default=None, help="Explicit source label map or directory")
    parser.add_argument("--config", default=None, help="JSON/YAML config; defaults are used when omitted")
    parser.add_argument("--output", required=True, help="Output directory (or legacy report.json path)")
    parser.add_argument("--no-vlm", action="store_true", help="Run deterministic objective selection only")
    args = parser.parse_args(argv)
    if bool(args.candidates) == bool(args.images):
        parser.error("provide exactly one of --candidates (V2) or --images (legacy)")
    if args.candidates and not args.source:
        parser.error("--source is required with production --candidates")

    cfg = load_config(args.config)
    evaluator = TMPGTEvaluator(cfg)
    candidate_root = Path(args.candidates or args.images)
    masks_root = Path(args.masks)
    explicit_source_mask = Path(args.source_mask) if args.source_mask else None
    source_paths = list(_iter_images(Path(args.source))) if args.source else [None]
    output_root, report_path = _output_paths(args.output)
    scenes = []
    for source_path in source_paths:
        if source_path is None:
            # Backward-compatible candidate-only mode cannot run preservation
            # or source-aware V2 Qwen, but can still write the same report tree.
            pool = candidate_root
            candidates = _discover_candidates(pool, strict_manifest=False, expected_source=None)
            proxy_source = Path(candidates[0]["image_path"])
            results = []
            for record in candidates:
                image_path = Path(record["image_path"])
                candidate_mask = _resolve_mask(image_path, masks_root)
                results.append(evaluator.evaluate_one(
                    image_path, candidate_mask, run_vlm=not args.no_vlm,
                    source_path=None, family=str(record["manifest"]["family"]),
                ))
            evaluator.apply_pool_consistency(results)
            semantic = ranking = None
            selection = evaluator.finalize_selection(results, ranking)
            scene_id = pool.name or "legacy"
            source_for_report = proxy_source
        else:
            pool = _pool_for_source(candidate_root, source_path)
            candidates = _discover_candidates(pool, strict_manifest=bool(args.candidates), expected_source=source_path)
            source_mask = _resolve_mask(source_path, masks_root, explicit_source_mask)
            results, semantic, ranking, selection = _run_scene(
                evaluator, source_path=source_path, source_mask=source_mask,
                candidates=candidates, no_vlm=args.no_vlm,
            )
            scene_id = source_path.stem
            source_for_report = source_path
        scenes.append(write_scene_artifacts(
            output_root, scene_id=scene_id, source_path=source_for_report,
            candidates=candidates, results=results, selection=selection,
            semantic=semantic, ranking=ranking,
        ))
    write_batch_report(output_root, scenes, report_path=report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
