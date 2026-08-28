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
    p.add_argument("--config", default=None, help="JSON/YAML config; defaults are used when omitted")
    p.add_argument("--output", required=True, help="Output JSON report")
    p.add_argument("--no-vlm", action="store_true", help="Run deterministic metrics only")
    args = p.parse_args()

    cfg = load_config(args.config)
    evaluator = TMPGTEvaluator(cfg)
    images_dir, masks_dir = Path(args.images), Path(args.masks)
    source_path = Path(args.source) if args.source else None
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    results = []
    mask_paths: dict[str, str] = {}
    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"missing semantic mask: {mask_path}")
        mask_paths[str(image_path)] = str(mask_path)
        results.append(
            evaluator.evaluate_one(
                image_path,
                mask_path,
                run_vlm=not args.no_vlm,
                source_path=source_path,
            )
        )

    evaluator.apply_pool_consistency(results)
    ranking = None
    if source_path is not None and not args.no_vlm and evaluator.semantic_judge is not None:
        ranking = evaluator.rank_candidates(results, source_path, mask_paths)

    report = {
        "count": len(results),
        "certified": sum(r.pgt_class == "CERTIFIED_PGT" for r in results),
        "usable": sum(r.pgt_class == "USABLE_PGT" for r in results),
        "rejected": sum(r.pgt_class == "REJECT" for r in results),
        "semantic_ranking": ranking,
        "results": [r.to_dict() for r in sorted(results, key=lambda x: x.quality_score, reverse=True)],
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
