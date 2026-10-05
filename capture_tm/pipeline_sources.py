"""Import explicit linear HDR recipes into the existing relative-radiance format.

NPY and NPZ (the ``frames`` key) require floating point storage. EXR and TIFF
are optional: install ``OpenEXR>=3`` for exact float R/G/B EXR decoding, or
imageio plus a float-preserving codec (``imageio[tifffile]`` for TIFF).
Decoded integer images are rejected rather than assigned an inferred scale.
No OETF inversion, per-frame normalization, HDR recovery, or camera calibration
is inferred from an image's appearance.
"""
from collections.abc import Iterator
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .data import _content_provenance, _float_array, _payload_hash, _write_scene
from .sources import _frames, _matrix
from .types import SCENE_DOMAIN, Scene, SensorProfile


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _times(value):
    if not isinstance(value, list) or not value:
        raise ValueError("frame_times_s must be an explicit nonempty list, including for a static singleton")
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in value):
        raise ValueError("frame times must be real numeric values, not bool")
    times = np.asarray(value, dtype=np.float64)
    if not np.isfinite(times).all() or (len(times) > 1 and not (np.diff(times) > 0).all()):
        raise ValueError("frame times must be finite and strictly increasing")
    return times


def _dependencies(record):
    dependencies = []
    for value in (record.get("dependency_ids", []), record["provenance"].get("dependency_ids", [])):
        if not isinstance(value, list):
            raise ValueError("dependency_ids must be a list of nonempty source identities")
        for identity in value:
            _nonempty_string(identity, "dependency_ids entry")
            if identity not in dependencies:
                dependencies.append(identity)
    return dependencies


def _path(root, value, name):
    return (root / Path(_nonempty_string(value, name)).expanduser()).resolve()


def _common(record, scene_ids):
    if not isinstance(record, dict):
        raise ValueError("each scene must be an object")
    result = dict(record)
    for name in ("scene_id", "source_id"):
        _nonempty_string(result.get(name), name)
    if result["scene_id"] in scene_ids:
        raise ValueError("duplicate scene_id")
    scene_ids.add(result["scene_id"])
    if not isinstance(result.get("split"), str) or result["split"] not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    result["provenance"] = dict(result.get("provenance", {})) if isinstance(result.get("provenance", {}), dict) else None
    if result["provenance"] is None:
        raise ValueError("provenance must be an object")
    result["dependency_ids"] = _dependencies(result)
    result["times"] = _times(result.get("frame_times_s"))
    return result


def _check_groups(records, *, original_content=False):
    """Source and dependency identifiers share one namespace before augmentation."""
    groups = {}
    for record in records:
        keys = [("source", identity) for identity in [record["source_id"], *record["dependency_ids"]]]
        keys.extend(("path", str(path)) for path in record["group_paths"])
        if record["provenance"].get("input_sha256"):
            keys.append(("input_sha256", str(record["provenance"]["input_sha256"])))
        if original_content and record["provenance"].get("original_frames_sha256"):
            keys.append(("original_frames_sha256", str(record["provenance"]["original_frames_sha256"])))
        for key in keys:
            if key in groups and groups[key] != record["split"]:
                raise ValueError(f"source identity {key!r} crosses scene splits")
            groups[key] = record["split"]


def _load_header(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or isinstance(payload.get("version"), bool) or payload.get("version") != 1:
        raise ValueError("unsupported source recipe or scene manifest version")
    if "sensor" not in payload or not isinstance(payload.get("scenes"), list) or not payload["scenes"]:
        raise ValueError("source recipe or scene manifest requires sensor and nonempty scenes")
    return SensorProfile.from_dict(payload["sensor"]), payload["scenes"]


def _recipe_records(records, root):
    prepared, scene_ids = [], set()
    for original in records:
        record = _common(original, scene_ids)
        if not isinstance(record.get("input_domain"), str) or record["input_domain"] not in {SCENE_DOMAIN, "linear_xyz"}:
            raise ValueError("input_domain must explicitly declare sensor_linear_relative_radiance or linear_xyz")
        if not isinstance(record.get("input_layout"), str) or record["input_layout"] not in {"CHW", "HWC", "TCHW", "THWC"}:
            raise ValueError("input_layout must explicitly declare CHW, HWC, TCHW, or THWC")
        scale = record.get("radiance_scale", 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("radiance_scale must be a finite positive number common to the whole scene")
        record["radiance_scale"] = float(scale)
        verified = record.get("clean_reference_verified", False)
        if not isinstance(verified, bool):
            raise ValueError("clean_reference_verified must be an explicit boolean")
        record["clean_reference_verified"] = verified
        limitations = record["provenance"].get("limitations", [])
        if not isinstance(limitations, list) or any(not isinstance(value, str) for value in limitations):
            raise ValueError("provenance limitations must be a list of strings")
        if record["input_domain"] == "linear_xyz":
            if record.get("xyz_to_sensor") is None:
                raise ValueError("linear_xyz requires an explicit xyz_to_sensor matrix")
            record["matrix"] = _matrix(record["xyz_to_sensor"])
        elif record.get("xyz_to_sensor") is not None:
            raise ValueError("xyz_to_sensor is only valid for input_domain=linear_xyz")
        if ("input_path" in record) == ("frame_paths" in record):
            raise ValueError("declare exactly one of input_path or frame_paths")
        if "input_path" in record:
            paths = [_path(root, record["input_path"], "input_path")]
        else:
            if not isinstance(record["frame_paths"], list) or not record["frame_paths"]:
                raise ValueError("frame_paths must be a nonempty ordered list")
            paths = [_path(root, value, "frame_paths entry") for value in record["frame_paths"]]
            if len(paths) != len(record["times"]):
                raise ValueError("frame_times_s must match frame_paths count")
        record["paths"] = paths
        record["group_paths"] = paths
        record["mask_path"] = None if record.get("subject_mask_path") is None else _path(root, record["subject_mask_path"], "subject_mask_path")
        prepared.append(record)
    _check_groups(prepared)
    # Complete metadata grouping before checking storage or decoding any scene.
    for record in prepared:
        for path in [*record["paths"], *([] if record["mask_path"] is None else [record["mask_path"]])]:
            if not path.is_file():
                raise FileNotFoundError(f"source file does not exist: {path}")
    return prepared


def _read_float_source(path):
    if path.suffix.lower() in {".npy", ".npz"}:
        return _float_array(path), "float_numpy"
    if path.suffix.lower() not in {".exr", ".tif", ".tiff"}:
        raise ValueError("linear sources require float NPY/NPZ, EXR, or TIFF; encoded PNG/JPEG is not linear HDR")
    if path.suffix.lower() == ".exr":
        try:
            import OpenEXR
        except ImportError:
            pass
        else:
            if not hasattr(OpenEXR, "File"):
                raise ImportError("native float EXR import requires OpenEXR>=3 with the File API")
            try:
                with OpenEXR.File(str(path), separate_channels=True) as source:
                    if len(source.parts) != 1:
                        raise ValueError("EXR import requires a single-part file; multipart sources are ambiguous")
                    if source.parts[0].type() not in {OpenEXR.scanlineimage, OpenEXR.tiledimage}:
                        raise ValueError("deep EXR pixels are not a single relative-radiance image")
                    channels = source.channels()
                    if set(channels) != {"R", "G", "B"}:
                        raise ValueError("EXR must contain exactly the R, G, B channels; extra or missing channels are ambiguous")
                    arrays = []
                    for name in ("R", "G", "B"):
                        channel = channels[name]
                        array = np.asarray(channel.pixels)
                        if channel.xSampling != 1 or channel.ySampling != 1 or array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
                            raise ValueError("EXR R/G/B channels must be full-resolution 2D float samples")
                        arrays.append(array)
                    if any(array.shape != arrays[0].shape for array in arrays):
                        raise ValueError("EXR R/G/B channels must have identical image dimensions")
                    return np.stack(arrays, axis=-1), "float_exr_OpenEXR"
            except ValueError:
                raise
            except Exception as error:
                raise ValueError(f"cannot decode float EXR source {path.name} with OpenEXR; provide a single-part float R/G/B image or export float NPY/NPZ") from error
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise ImportError("float EXR/TIFF import requires optional OpenEXR>=3 for EXR, or imageio and a float-preserving codec; install imageio[tifffile] for TIFF or an EXR-capable plugin") from error
    try:
        array = np.asarray(iio.imread(path))
    except Exception as error:
        raise ValueError(f"cannot decode float {path.suffix.upper()[1:]} source {path.name}; install a compatible imageio codec (imageio[tifffile] for TIFF, an EXR-capable plugin for EXR), or export float NPY/NPZ") from error
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path.name} codec returned {array.dtype}; linear EXR/TIFF requires float decoding, never uint8 normalization")
    return array, "float_" + path.suffix.lower().lstrip(".")


def _recipe_scene(record):
    decoded, storage = [], []
    for path in record["paths"]:
        array, kind = _read_float_source(path)
        layout = record["input_layout"]
        # A frame_paths list explicitly supplies T; its entries may store the
        # declared spatial layout directly or include a singleton T axis.
        if "frame_paths" in record and layout.startswith("T") and array.ndim == 3:
            layout = layout[1:]
        frames = _frames(array, input_layout=layout)
        if "frame_paths" in record and len(frames) != 1:
            raise ValueError("each frame_paths entry must contain exactly one frame in its declared layout")
        decoded.append(frames)
        storage.append(kind)
    if any(value.shape[1:] != decoded[0].shape[1:] for value in decoded):
        raise ValueError("all sequence frames must have identical channel and image dimensions")
    frames = decoded[0] if len(decoded) == 1 else np.concatenate(decoded, axis=0)
    if len(frames) != len(record["times"]):
        raise ValueError("frame_times_s must match decoded frame count")
    provenance = {**record["provenance"], "input_domain": record["input_domain"],
        "original_frames_sha256": _payload_hash(frames),
        "input_layout": record["input_layout"], "input_encoding": "linear_float",
        "input_paths": [str(path) for path in record["paths"]], "storage": storage,
        "radiance_scale": record["radiance_scale"], "radiance_scale_scope": "whole_scene",
        "dependency_ids": record["dependency_ids"],
        "clean_reference_verified": record["clean_reference_verified"],
        "absolute_radiance_calibrated": False, "relative_source": True, "hdr_recovery_claim": False,
        "static_singleton": len(frames) == 1,
        "limitations": [*record["provenance"].get("limitations", []),
            "declared relative linear source; no verified absolute radiometric or sensor calibration",
            "source clipping, denoising, motion sampling and residual noise are not repaired"]}
    if record["input_domain"] == "linear_xyz":
        frames = np.einsum("ij,tjhw->tihw", record["matrix"], frames)
        provenance["xyz_to_sensor"] = record["matrix"].tolist()
        provenance["negative_matrix_output_clip_fraction"] = float(np.mean(frames < 0))
        frames = np.maximum(frames, 0)
    with np.errstate(over="ignore", invalid="ignore"):
        frames = np.asarray(frames * record["radiance_scale"], dtype=np.float32)
    mask = None if record["mask_path"] is None else torch.from_numpy(_float_array(record["mask_path"], "mask"))
    return Scene(record["scene_id"], record["split"], torch.from_numpy(frames),
        torch.from_numpy(record["times"]), mask, source_kind="linear_import",
        source_id=record["source_id"], provenance=provenance)


def import_source_recipe(recipe_path, output_dir) -> Path:
    """Atomically write a new v1 dataset from an explicit version-1 JSON recipe.

    Paths are relative to the recipe. ``input_path`` holds one explicit array;
    ``frame_paths`` is an ordered sequence with one frame per entry. Layout is
    exact; for ``frame_paths``, the list supplies the temporal axis of TCHW/THWC,
    and an optional singleton stored temporal axis is also accepted. Every scene
    supplies ``frame_times_s``, even a singleton ``[0.0]``.
    ``radiance_scale`` defaults to one and multiplies the complete scene once.
    Existing output directories are refused, including empty directories.
    """
    recipe = Path(recipe_path).expanduser().resolve()
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output dataset already exists: {output}")
    sensor, records = _load_header(recipe)
    prepared = _recipe_records(records, recipe.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    original_splits = {}
    try:
        for record in prepared:
            scene = _recipe_scene(record)
            digest = scene.provenance["original_frames_sha256"]
            if digest in original_splits and original_splits[digest] != scene.split:
                raise ValueError("original decoded source content crosses scene splits before augmentation")
            original_splits[digest] = scene.split
            _write_scene(staging, scene, sensor)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output dataset already exists: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "manifest.json"


def iter_source_scenes(manifest_path) -> Iterator[tuple[SensorProfile, Scene]]:
    """Yield existing v1 scenes without retaining previously decoded tensors.

    Source/dependency/path grouping is checked across all rows before the first
    yield. Actual decoded content is checked across splits as scenes are read;
    stored provenance fingerprints are recomputed rather than trusted.
    """
    path = Path(manifest_path).expanduser().resolve()
    sensor, originals = _load_header(path)
    records, scene_ids = [], set()
    for original in originals:
        record = _common(original, scene_ids)
        if record.get("domain") != SCENE_DOMAIN:
            raise ValueError(f"unknown scene domain {record.get('domain')!r}")
        _nonempty_string(record.get("source_kind"), "source_kind")
        record["frames_file"] = _path(path.parent, record.get("frames_path"), "frames_path")
        record["mask_file"] = None if record.get("subject_mask_path") is None else _path(path.parent, record["subject_mask_path"], "subject_mask_path")
        record["group_paths"] = [record["frames_file"]]
        original_paths = record["provenance"].get("input_paths", [])
        if not isinstance(original_paths, list):
            raise ValueError("provenance input_paths must be a list")
        if record["provenance"].get("input_path") is not None:
            original_paths = [*original_paths, record["provenance"]["input_path"]]
        record["group_paths"].extend(_path(path.parent, value, "provenance input path") for value in original_paths)
        records.append(record)
    _check_groups(records, original_content=True)
    content_splits = {}
    for record in records:
        frames = _float_array(record["frames_file"])
        mask = None if record["mask_file"] is None else _float_array(record["mask_file"], "mask")
        provenance = _content_provenance(record["provenance"], frames, mask)
        provenance["dependency_ids"] = record["dependency_ids"]
        digest = provenance["frames_sha256"]
        if digest in content_splits and content_splits[digest] != record["split"]:
            raise ValueError("decoded frame content crosses scene splits")
        content_splits[digest] = record["split"]
        yield sensor, Scene(record["scene_id"], record["split"], torch.from_numpy(frames),
            torch.from_numpy(record["times"]), None if mask is None else torch.from_numpy(mask),
            record["source_kind"], record["domain"], record["source_id"], provenance)
