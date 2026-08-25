from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluator import TMPGTEvaluator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    p = argparse.ArgumentParser(description="No-reference TM pseudo-GT IQA selector")
    p.add_argument("--images", required=True, help="Directory containing TM candidate images")
    p.add_argument("--masks", required=True, help="Directory containing semantic label maps with matching stems")
    p.add_argument("--config", default=None, help="JSON/YAML config; defaults are used when omitted")
    p.add_argument("--output", required=True, help="Output JSON report")
    p.add_argument("--no-vlm", action="store_true", help="Run deterministic metrics only")
    args = p.parse_args()

    cfg = load_config(args.config)
    evaluator = TMPGTEvaluator(cfg)
    images_dir, masks_dir = Path(args.images), Path(args.masks)
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    results = []
    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"missing semantic mask: {mask_path}")
        results.append(evaluator.evaluate_one(image_path, mask_path, run_vlm=not args.no_vlm))
    evaluator.apply_pool_consistency(results)
    report = {
        "count": len(results),
        "certified": sum(r.pgt_class == "CERTIFIED_PGT" for r in results),
        "usable": sum(r.pgt_class == "USABLE_PGT" for r in results),
        "rejected": sum(r.pgt_class == "REJECT" for r in results),
        "results": [r.to_dict() for r in sorted(results, key=lambda x: x.quality_score, reverse=True)],
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
