"""Audit processed DNG sources; export small local-only green patches.

No photos, cache pixels, private absolute paths, EXIF GPS, or expert-GT claims
are written to the public report. The supplied DNGs remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import time
import numpy as np
import rawpy
import tifffile

CAMERAS = ("iphone", "s25")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def filename_from_manifest(value: str) -> str:
    return PureWindowsPath(value.replace("/", "\\")).name


def resolve_moved_file(root: Path, camera: str, folder: str, value: str) -> Path:
    """Resolve stale Windows paths by basename inside the declared camera root."""
    name = filename_from_manifest(value)
    preferred = root / camera / folder / name
    if preferred.is_file():
        return preferred
    candidates = [p for p in (root / camera / folder).rglob("*")
                  if p.is_file() and p.name.casefold() == name.casefold()]
    if not candidates:
        raise FileNotFoundError(f"{camera}/{folder}/{name}")
    hashes = {digest(p) for p in candidates}
    if len(hashes) != 1:
        raise ValueError(f"Ambiguous moved file: {camera}/{folder}/{name}")
    return sorted(candidates, key=lambda p: p.as_posix())[0]


def ifd_metadata(path: Path) -> list[dict]:
    records = []
    names = ("ImageWidth", "ImageLength", "BitsPerSample", "SamplesPerPixel",
             "PhotometricInterpretation", "Compression", "BlackLevel",
             "WhiteLevel", "NoiseProfile", "BaselineExposure", "DNGVersion")
    def simple(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (tuple, bytes)):
            return list(value)
        if isinstance(value, np.generic):
            return value.item()
        return value
    with tifffile.TiffFile(path) as tf:
        def visit(page, label):
            item = {"ifd": label}
            for name in names:
                if name in page.tags:
                    item[name] = simple(page.tags[name].value)
            if "LinearizationTable" in page.tags:
                table = page.tags["LinearizationTable"].value
                item["linearization_entries"] = len(table)
                item["linearization_minmax"] = [int(min(table)), int(max(table))]
            photo = int(item.get("PhotometricInterpretation", -1))
            item["raw_kind"] = ("processed-linear-rgb" if photo == 34892
                                else "cfa-bayer" if photo == 32803 else "non-raw")
            records.append(item)
            if page.pages is not None:
                for index, subpage in enumerate(page.pages):
                    visit(subpage, f"{label}/{index}")
        for index, page in enumerate(tf.pages):
            visit(page, str(index))
    return records


def validate_groups(rows: list[dict], protocol: dict) -> dict[int, tuple[str, str]]:
    """Every source pair maps to one group and one split before cropping."""
    mapping = {}
    group_splits = {}
    for group in protocol["source_groups"]:
        gid, split = group["group_id"], group["split"]
        if gid in group_splits:
            raise ValueError(f"Duplicate group ID {gid}")
        group_splits[gid] = split
        for pair_id in group["pair_ids"]:
            if pair_id in mapping:
                raise ValueError(f"Pair {pair_id} appears in multiple groups")
            mapping[pair_id] = (gid, split)
    ids = {row["pair_id"] for row in rows}
    if set(mapping) != ids or len(ids) != len(rows):
        raise ValueError("Group protocol must cover every manifest pair exactly once")
    if set(group_splits.values()) != {"train", "development", "diagnostic"}:
        raise ValueError("Expected train/development/diagnostic source groups")
    return mapping


def extract_green(path: Path, count: int, size: int, rng) -> tuple[np.ndarray, dict]:
    """One RAW buffer at a time; allocate float32 only for selected patches."""
    begin = time.perf_counter()
    with rawpy.imread(str(path)) as raw:
        visible = raw.raw_image_visible
        if visible.ndim != 3 or visible.shape[2] < 3 or raw.raw_pattern is not None:
            raise ValueError("This diagnostic only accepts processed stacked RGB DNG")
        green = visible[:, :, 1]
        black = float(raw.black_level_per_channel[1])
        white = float(raw.white_level)
        if white <= black:
            raise ValueError("Invalid black/white levels")
        h, w = green.shape
        if min(h, w) < size:
            raise ValueError("Source smaller than requested patch")
        stats_values = green[::16, ::16].astype(np.float32)
        stats_values = np.clip((stats_values-black)/(white-black), 0, 1)
        patches, positions = [], []
        for _ in range(count):
            y, x = int(rng.integers(0, h-size+1)), int(rng.integers(0, w-size+1))
            patch = green[y:y+size, x:x+size].astype(np.float32)
            patch = np.clip((patch-black)/(white-black), 0, 1)
            patches.append(patch)
            positions.append([y, x])
        summary = {"shape": list(visible.shape), "raw_type": str(raw.raw_type),
                   "channel": 1, "black_level": black, "white_level": white,
                   "raw_pattern": None,
                   "strided_source_stats": {
                       "stride": 16, "count": int(stats_values.size),
                       "q01_q50_q99": np.quantile(stats_values, [.01,.5,.99]).tolist(),
                       "black_fraction": float(np.mean(stats_values == 0)),
                       "white_fraction": float(np.mean(stats_values == 1))},
                   "patch_positions": positions,
                   "decode_seconds": time.perf_counter()-begin}
    return np.stack(patches), summary


def run(root: Path, protocol_path: Path, output: Path, cache: Path):
    begin = time.perf_counter()
    manifest = root / "dataset.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    mapping = validate_groups(rows, protocol)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    inventory, sources = [], []
    for camera in CAMERAS:
        # One suffix comparison avoids Windows case-insensitive double glob counts.
        for path in sorted((root/camera/"DNG").rglob("*")):
            if path.is_file() and path.suffix.casefold() == ".dng":
                inventory.append({"camera": camera,
                                  "path": path.relative_to(root).as_posix(),
                                  "bytes": path.stat().st_size,
                                  "sha256": digest(path)})
    inv_by_path = {item["path"]: item for item in inventory}
    for row in rows:
        pair_id = row["pair_id"]
        gid, split = mapping[pair_id]
        for index, camera in enumerate(CAMERAS):
            path = resolve_moved_file(root, camera, "DNG", row[camera]["dng"])
            rel = path.relative_to(root).as_posix()
            metadata = ifd_metadata(path)
            raw_ifds = [item for item in metadata if item["raw_kind"] != "non-raw"]
            if len(raw_ifds) != 1:
                raise ValueError(f"Expected exactly one raw IFD: {rel}")
            if raw_ifds[0]["raw_kind"] != "processed-linear-rgb":
                raise ValueError(f"Unsupported raw domain: {rel}")
            rng = np.random.default_rng(np.random.SeedSequence(
                [protocol["patch_seed"], pair_id, index]))
            patches, summary = extract_green(
                path, protocol["patches_per_source"], protocol["patch_size"], rng)
            source_id = f"{camera}_pair_{pair_id:02d}"
            np.savez_compressed(cache/f"{source_id}.npz", green=patches)
            # Check paired assets' existence; don't treat a file called gt as GT.
            asset_status = {}
            for key, folder in (("input_linear_png", "input"), ("gt_jpg", "gt")):
                try:
                    resolve_moved_file(root, camera, folder, row[camera][key])
                    asset_status[key] = "exists_provenance_unverified"
                except FileNotFoundError:
                    asset_status[key] = "missing"
            item = {"source_id": source_id, "pair_id": pair_id, "camera": camera,
                    "group_id": gid, "split": split, "path": rel,
                    "sha256": inv_by_path[rel]["sha256"],
                    "ifds": metadata, "decoded": summary, "assets": asset_status}
            sources.append(item)
            print(json.dumps({"source":source_id,"group":gid,"split":split,
                              "decode_seconds":summary["decode_seconds"]}), flush=True)
    source_hash_splits = {}
    for item in sources:
        source_hash_splits.setdefault(item["sha256"], set()).add(item["split"])
    if any(len(splits) > 1 for splits in source_hash_splits.values()):
        raise ValueError("Duplicate source content crosses split boundaries")
    hashes = {}
    for item in inventory:
        hashes.setdefault(item["sha256"], []).append(item["path"])
    report = {"protocol_id": protocol["protocol_id"],
              "protocol_sha256": digest(protocol_path),
              "manifest_sha256": digest(manifest),
              "pair_count": len(rows), "source_count": len(sources),
              "source_group_count": len(protocol["source_groups"]),
              "independence_limit": protocol["independence_limit"],
              "inventory_count": len(inventory), "unique_dng_hashes":len(hashes),
              "duplicates": [paths for paths in hashes.values() if len(paths)>1],
              "target_provenance": "fixed analytic transform of bounded green DNG source",
              "original_gt_files": "unused; provenance unverified",
              "source_domain": "processed Linear RAW; already noisy/possibly clipped",
              "recapture_status": "relative synthetic noise/exposure, not physical recapture",
              "sources": sources, "inventory": inventory,
              "wall_seconds": time.perf_counter()-begin,
              "output_pixels_policy": "npz caches stay local and must never be committed"}
    (output/"dng_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (cache/"cache_manifest.json").write_text(json.dumps({
        "protocol_sha256": report["protocol_sha256"],
        "source_domain": report["source_domain"],
        "sources": [{k:item[k] for k in ("source_id","pair_id","camera","group_id","split","sha256")}
                    for item in sources]},indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args=parser.parse_args()
    result=run(args.data_root,args.protocol,args.output,args.cache)
    print(json.dumps({key:result[key] for key in
                      ("pair_count","source_count","source_group_count",
                       "inventory_count","unique_dng_hashes","wall_seconds")}))
