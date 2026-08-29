from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluator import TMPGTEvaluator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    p = argparse.ArgumentParser(description="No-reference TM pseudo-GT IQA selector")
    p.add_argument("--images", required=True, help="Directory containing one TM candidate pool")
    p.add_argument("--masks", required=True, help="Directory containing semantic label maps with matching stems")
    p.add_argument("--source", default=None, help="Source image before candidate Tone Mapping; enables semantic Source/Candidate judgment")
    p.add_argument("--source-mask", default=None, help="Explicit source semantic label map; required for source-aware VLM unless masks/<source-stem>.png exists")
    p.add_argument("--config", default=None, help="JSON/YAML config; defaults are used when omitted")
    p.add_argument("--output", required=True, help="Output JSON report")
    p.add_argument("--no-vlm", action="store_true", help="Run deterministic metrics only")
    args = p.parse_args()

    cfg = load_config(args.config)
    evaluator = TMPGTEvaluator(cfg)
    images_dir, masks_dir = Path(args.images), Path(args.masks)
    source_path = Path(args.source) if args.source else None
    source_mask_path = Path(args.source_mask) if args.source_mask else None
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    results = []
    mask_paths: dict[str, str] = {}
    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"missing semantic mask: {mask_path}")
        manifest_path = image_path.with_suffix(".json")
        family = "unknown"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            family = str(manifest.get("family", "unknown"))
        mask_paths[str(image_path)] = str(mask_path)
        results.append(
            evaluator.evaluate_one(
                image_path,
                mask_path,
                # With a source, V2 runs all objective checks first.  The
                # source-aware semantic calls below are then restricted to the
                # deterministic family-balanced Top-K.
                run_vlm=not args.no_vlm and source_path is None,
                source_path=source_path,
                family=family,
            )
        )

    evaluator.apply_pool_consistency(results)
    ranking = None
    semantic = None
    if source_path is not None and not args.no_vlm and evaluator.semantic_judge is not None:
        if source_mask_path is None:
            source_mask_path = masks_dir / f"{source_path.stem}.png"
        if not source_mask_path.exists():
            raise FileNotFoundError(
                "source-aware semantic judging requires --source-mask or "
                f"{source_mask_path}"
            )
        semantic = evaluator.review_semantic_topk(results, source_path, source_mask_path, mask_paths)
        ranking = evaluator.rank_candidates(results, source_path, mask_paths)
    selection = evaluator.finalize_selection(results, ranking)

    report = {
        "count": len(results),
        "certified": sum(r.pgt_class == "CERTIFIED_PGT" for r in results),
        "usable": sum(r.pgt_class == "USABLE_PGT" for r in results),
        "rejected": sum(r.pgt_class == "REJECT" for r in results),
        "semantic_ranking": ranking,
        "semantic_scene": semantic["scene"] if semantic else None,
        "selection": selection,
        "results": [r.to_dict() for r in sorted(results, key=lambda x: x.quality_score, reverse=True)],
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
