from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from .schemas import build_manifest, write_json_atomic


def _read_tags(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    tags: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            scene = row.get("scene_id") or row.get("scene") or row.get("filename")
            tag = row.get("tag") or row.get("tags")
            if not scene or not tag:
                raise ValueError("scene-tags CSV requires scene_id and tag/tags columns")
            tags[scene].extend(part.strip() for part in tag.split(",") if part.strip())
    return dict(tags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and validate an adjustTM benchmark manifest")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--scene-list")
    parser.add_argument("--scene-tags")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.input_dir, args.gt_root, args.scene_list, scene_tags=_read_tags(args.scene_tags))
    write_json_atomic(args.output, manifest)
    print(f"manifest scenes={manifest['scene_count']} sha256={manifest['content_sha256']}")


if __name__ == "__main__":
    main()
