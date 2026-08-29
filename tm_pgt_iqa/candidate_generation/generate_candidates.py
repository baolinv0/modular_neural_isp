from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from ..config import CandidateGenerationConfig, IQAConfig, load_config
from ..segmentation import SemanticMasks, load_label_map, resize_masks
from .gain import generate_gain_max
from .local_face_tm import generate_local_face_tm
from .qwen_edit import QwenEditAdapter, generate_qwen_edits
from .retinex import generate_retinex
from .tone_shape import generate_tone_shape


@dataclass(frozen=True)
class CandidateManifest:
    """Authoritative candidate identity; selectors must not infer this from paths."""

    candidate_id: str
    family: str
    parameters: dict[str, float | str | bool]
    source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "CandidateManifest":
        required = {"candidate_id", "family", "parameters", "source"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"candidate manifest missing keys: {sorted(missing)}")
        return cls(
            candidate_id=str(value["candidate_id"]),
            family=str(value["family"]),
            parameters=dict(value["parameters"]),
            source=str(value["source"]),
        )


@dataclass(frozen=True)
class GeneratedCandidate:
    manifest: CandidateManifest
    rgb: np.ndarray


def _candidate_config(config: IQAConfig | CandidateGenerationConfig) -> CandidateGenerationConfig:
    return config.candidate_generation if isinstance(config, IQAConfig) else config


def _append(
    pool: list[GeneratedCandidate],
    images: dict[str, np.ndarray],
    *,
    family: str,
    source_name: str,
    parameters: Callable[[str], dict[str, float | str | bool]],
) -> None:
    for candidate_id, rgb in images.items():
        pool.append(
            GeneratedCandidate(
                CandidateManifest(candidate_id, family, parameters(candidate_id), source_name),
                np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0),
            )
        )


def generate_pool(
    source_rgb: np.ndarray,
    masks: SemanticMasks,
    config: IQAConfig | CandidateGenerationConfig,
    *,
    source_name: str | None = None,
    edit_adapter: QwenEditAdapter | None = None,
) -> list[GeneratedCandidate]:
    """Build the 14 deterministic candidates plus optional edit-adapter outputs."""
    source_rgb = np.asarray(source_rgb, dtype=np.float32)
    if source_rgb.ndim != 3 or source_rgb.shape[-1] != 3:
        raise ValueError("source_rgb must have HxWx3 shape")
    if source_name is None:
        # Programmatic callers can provide a source name when provenance must
        # be retained.  This stable marker avoids inventing a file identity.
        source_name = "memory://unspecified-source"
    if masks.face.shape != source_rgb.shape[:2]:
        masks = resize_masks(masks, source_rgb.shape[:2])
    cfg = _candidate_config(config)
    pool: list[GeneratedCandidate] = []

    retinex = generate_retinex(source_rgb, cfg.retinex)
    _append(
        pool, retinex, family="retinex", source_name=source_name,
        parameters=lambda candidate_id: {"ev": float(cfg.retinex.levels_ev[candidate_id])},
    )
    local = generate_local_face_tm(source_rgb, masks, cfg.local_face_tm)
    _append(
        pool, local, family="local_face_tm", source_name=source_name,
        parameters=lambda candidate_id: {"lift_ev": float(cfg.local_face_tm.lift_ev[candidate_id.removeprefix("face_lift_")])},
    )
    tone = generate_tone_shape(source_rgb, masks, cfg.tone_shape)
    tone_params = {
        "tone_shadow_preserve": {"gamma": float(cfg.tone_shape.shadow_preserve_gamma)},
        "tone_soft_highlight": {"gamma": float(cfg.tone_shape.soft_highlight_gamma)},
    }
    _append(pool, tone, family="tone_shape", source_name=source_name, parameters=lambda candidate_id: tone_params[candidate_id])

    qwen_edits = generate_qwen_edits(source_rgb, masks, cfg.qwen_edit, edit_adapter)
    _append(
        pool, qwen_edits, family="qwen_edit", source_name=source_name,
        parameters=lambda candidate_id: {"strength": candidate_id.removeprefix("qwen_")},
    )
    _append(
        pool, {"gain_max": generate_gain_max(source_rgb, cfg.gain)}, family="gain", source_name=source_name,
        parameters=lambda _candidate_id: {"max_gain": float(cfg.gain.max_gain), "max_clip_ratio": float(cfg.gain.max_clip_ratio)},
    )
    ids = [candidate.manifest.candidate_id for candidate in pool]
    if len(ids) != len(set(ids)):
        raise RuntimeError("candidate IDs must be unique")
    return pool


def write_pool(pool: Iterable[GeneratedCandidate], output_dir: str | Path) -> list[Path]:
    """Persist PNGs and adjacent authoritative JSON manifests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for candidate in pool:
        image_path = output_dir / f"{candidate.manifest.candidate_id}.png"
        manifest_path = output_dir / f"{candidate.manifest.candidate_id}.json"
        Image.fromarray(np.rint(candidate.rgb * 255.0).astype(np.uint8), mode="RGB").save(image_path)
        manifest_path.write_text(json.dumps(candidate.manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        saved.extend((image_path, manifest_path))
    return saved


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.iterdir()):
        if child.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            yield child


def _mask_for(image: Path, masks_path: Path) -> Path:
    if masks_path.is_file():
        return masks_path
    choices = (masks_path / image.name, masks_path / f"{image.stem}_mask.png", masks_path / f"{image.stem}.png")
    for choice in choices:
        if choice.exists():
            return choice
    raise FileNotFoundError(f"no label map found for {image.name} under {masks_path}")


def _soft_mask_for(image: Path, root: Path | None, kind: str) -> Path | None:
    """Resolve an optional soft map from an explicit file or a per-source directory."""
    if root is None:
        return None
    if root.is_file():
        return root
    choices = (
        root / f"{image.stem}_soft_{kind}.png",
        root / f"{image.stem}_{kind}.png",
        root / image.name,
    )
    return next((choice for choice in choices if choice.exists()), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic TM PGT candidate pools.")
    parser.add_argument("--input", required=True, help="Source image or directory")
    parser.add_argument("--masks", required=True, help="Label map or directory")
    parser.add_argument("--output", required=True, help="Output candidate directory")
    parser.add_argument("--config", default=None, help="Optional JSON/YAML IQA config")
    parser.add_argument("--soft-face", default=None, help="Optional soft face map file or directory (<stem>_soft_face.png)")
    parser.add_argument("--soft-skin", default=None, help="Optional soft skin map file or directory (<stem>_soft_skin.png)")
    parser.add_argument("--soft-human", default=None, help="Optional soft human map file or directory (<stem>_soft_human.png)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    input_path, masks_path, output_path = Path(args.input), Path(args.masks), Path(args.output)
    soft_face_root = Path(args.soft_face) if args.soft_face else None
    soft_skin_root = Path(args.soft_skin) if args.soft_skin else None
    soft_human_root = Path(args.soft_human) if args.soft_human else None
    for source_path in _iter_images(input_path):
        rgb = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32) / 255.0
        masks = load_label_map(
            _mask_for(source_path, masks_path), cfg.labels,
            soft_face_path=_soft_mask_for(source_path, soft_face_root, "face"),
            soft_skin_path=_soft_mask_for(source_path, soft_skin_root, "skin"),
            soft_human_path=_soft_mask_for(source_path, soft_human_root, "human"),
        )
        target = output_path / source_path.stem
        write_pool(generate_pool(rgb, masks, cfg, source_name=source_path.name), target)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
