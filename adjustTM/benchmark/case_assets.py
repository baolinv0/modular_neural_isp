from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _normalise_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    array = image.astype(np.float32)
    maximum = float(array.max()) if array.size else 1.0
    return array / maximum if maximum > 1.0 else array


def _window_origin(
    score: np.ndarray,
    window_width: int,
    window_height: int,
    *,
    maximum: bool,
) -> tuple[int, int]:
    averaged = cv2.boxFilter(
        score.astype(np.float32), -1, (window_width, window_height), normalize=True
    )
    _, _, minimum_location, maximum_location = cv2.minMaxLoc(averaged)
    center_x, center_y = maximum_location if maximum else minimum_location
    x = int(np.clip(center_x - window_width // 2, 0, max(0, score.shape[1] - window_width)))
    y = int(np.clip(center_y - window_height // 2, 0, max(0, score.shape[0] - window_height)))
    return x, y


def automatic_crop_boxes(
    image_rgb: np.ndarray,
    crop_fraction: float = 0.25,
) -> dict[str, tuple[int, int, int, int]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image_rgb.shape}")
    if not 0.05 <= crop_fraction <= 1.0:
        raise ValueError("crop_fraction must be in [0.05, 1.0]")
    height, width = image_rgb.shape[:2]
    crop_width = min(width, max(8, int(round(width * crop_fraction))))
    crop_height = min(height, max(8, int(round(height * crop_fraction))))
    image = _normalise_rgb(image_rgb)
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    highlight_x, highlight_y = _window_origin(
        luminance, crop_width, crop_height, maximum=True
    )
    shadow_x, shadow_y = _window_origin(
        luminance, crop_width, crop_height, maximum=False
    )
    texture_x, texture_y = _window_origin(
        texture, crop_width, crop_height, maximum=True
    )
    return {
        "highlight": (highlight_x, highlight_y, crop_width, crop_height),
        "shadow": (shadow_x, shadow_y, crop_width, crop_height),
        "texture": (texture_x, texture_y, crop_width, crop_height),
    }


def _scale_crop_box(
    box: tuple[int, int, int, int],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    x, y, width, height = box
    scaled_x = int(round(x / max(1, source_width) * target_width))
    scaled_y = int(round(y / max(1, source_height) * target_height))
    scaled_width = max(1, int(round(width / max(1, source_width) * target_width)))
    scaled_height = max(1, int(round(height / max(1, source_height) * target_height)))
    scaled_x = min(max(0, scaled_x), max(0, target_width - 1))
    scaled_y = min(max(0, scaled_y), max(0, target_height - 1))
    scaled_width = min(scaled_width, target_width - scaled_x)
    scaled_height = min(scaled_height, target_height - scaled_y)
    return scaled_x, scaled_y, scaled_width, scaled_height


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected three-channel image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write {path}")


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    if mode == "symlink":
        try:
            destination.symlink_to(source.resolve())
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    raise ValueError("asset_mode must be copy, hardlink, or symlink")


def _scene_slug(scene_id: str) -> str:
    stem = Path(scene_id).stem.replace(" ", "_")[:48]
    digest = hashlib.sha1(scene_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{digest}"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def materialize_gallery_assets(
    *,
    manifest: Mapping[str, Any],
    output_root: str | Path,
    methods: Sequence[str],
    levels: Sequence[str],
    output_dir: str | Path,
    selected_scenes: Sequence[str],
    asset_mode: str = "copy",
    crop_fraction: float = 0.25,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_dir = Path(output_dir)
    assets_root = output_dir / "assets"
    selected = set(selected_scenes)
    index: dict[str, Any] = {
        "scenes": {},
        "methods": list(methods),
        "levels": list(levels),
    }
    for scene in manifest.get("scenes", []):
        scene_id = str(scene["scene_id"])
        slug = _scene_slug(scene_id)
        scene_entry: dict[str, Any] = {
            "scene_id": scene_id,
            "slug": slug,
            "tags": list(scene.get("tags", [])),
            "target": {},
            "methods": {method: {} for method in methods},
            "crops": {},
        }
        for level in levels:
            target_source = Path(scene["gt"][level]["path"])
            target_destination = assets_root / slug / "target" / level / target_source.name
            _materialize(target_source, target_destination, asset_mode)
            scene_entry["target"][level] = _relative(target_destination, output_dir)
            for method in methods:
                method_source = output_root / method / level / scene_id
                method_destination = (
                    assets_root / slug / "methods" / method / level / Path(scene_id).name
                )
                _materialize(method_source, method_destination, asset_mode)
                scene_entry["methods"][method][level] = _relative(
                    method_destination, output_dir
                )

        if scene_id in selected:
            center_source = Path(scene["gt"]["a_000"]["path"])
            center_image = _read_rgb(center_source)
            source_shape = center_image.shape[:2]
            boxes = automatic_crop_boxes(center_image, crop_fraction=crop_fraction)
            scene_entry["crop_boxes"] = {
                key: list(value) for key, value in boxes.items()
            }
            scene_entry["crop_reference_size"] = {
                "height": source_shape[0],
                "width": source_shape[1],
            }
            for crop_name, box in boxes.items():
                crop_entry: dict[str, Any] = {
                    "target": {},
                    "methods": {method: {} for method in methods},
                }
                for level in levels:
                    target_image = _read_rgb(scene["gt"][level]["path"])
                    tx, ty, tw, th = _scale_crop_box(
                        box, source_shape, target_image.shape[:2]
                    )
                    target_crop = target_image[ty : ty + th, tx : tx + tw]
                    target_crop_path = (
                        assets_root
                        / slug
                        / "crops"
                        / crop_name
                        / "target"
                        / f"{level}.png"
                    )
                    _write_rgb(target_crop_path, target_crop)
                    crop_entry["target"][level] = _relative(
                        target_crop_path, output_dir
                    )
                    for method in methods:
                        method_image = _read_rgb(
                            output_root / method / level / scene_id
                        )
                        mx, my, mw, mh = _scale_crop_box(
                            box, source_shape, method_image.shape[:2]
                        )
                        method_crop = method_image[my : my + mh, mx : mx + mw]
                        method_crop_path = (
                            assets_root
                            / slug
                            / "crops"
                            / crop_name
                            / method
                            / f"{level}.png"
                        )
                        _write_rgb(method_crop_path, method_crop)
                        crop_entry["methods"][method][level] = _relative(
                            method_crop_path, output_dir
                        )
                scene_entry["crops"][crop_name] = crop_entry
        index["scenes"][scene_id] = scene_entry
    return index
