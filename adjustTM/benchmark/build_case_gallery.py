from __future__ import annotations

import argparse
import json

from adjustTM.constants import LEVELS

from .case_gallery import build_case_gallery, read_jsonl
from .schemas import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic visual evidence galleries for adjustTM benchmark outputs"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-records", required=True)
    parser.add_argument("--dense-records", required=True)
    parser.add_argument("--control-records")
    parser.add_argument("--vlm-records")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", default=[name for name, _ in LEVELS])
    parser.add_argument("--focus-method", required=True)
    parser.add_argument("--comparison-baseline", default="gamma_global")
    parser.add_argument("--representative-count", type=int, default=6)
    parser.add_argument("--best-count", type=int, default=6)
    parser.add_argument("--failure-count", type=int, default=6)
    parser.add_argument("--disagreement-count", type=int, default=6)
    parser.add_argument(
        "--asset-mode", choices=["copy", "hardlink", "symlink"], default="copy"
    )
    parser.add_argument("--crop-fraction", type=float, default=0.25)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_case_gallery(
        manifest=read_json(args.manifest),
        output_root=args.output_root,
        reference_records=read_jsonl(args.reference_records),
        dense_records=read_jsonl(args.dense_records),
        control_records=read_jsonl(args.control_records),
        vlm_records=read_jsonl(args.vlm_records),
        methods=args.methods,
        levels=args.levels,
        focus_method=args.focus_method,
        comparison_baseline=args.comparison_baseline,
        representative_count=args.representative_count,
        best_count=args.best_count,
        failure_count=args.failure_count,
        disagreement_count=args.disagreement_count,
        asset_mode=args.asset_mode,
        crop_fraction=args.crop_fraction,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
