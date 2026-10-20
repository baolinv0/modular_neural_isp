"""Explicit recipe imports preserve radiance and reject split leakage."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


DOMAIN = "sensor_linear_relative_radiance"


def _api():
    assert importlib.util.find_spec("capture_tm.pipeline_sources") is not None, "recipe importer is missing"
    return importlib.import_module("capture_tm.pipeline_sources")


def _record(**changes):
    return {"scene_id": "scene", "source_id": "original", "split": "train",
        "input_path": "frames.npy", "input_domain": DOMAIN, "input_layout": "HWC",
        "frame_times_s": [0.0], **changes}


def _recipe(tmp_path, records):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"version": 1, "sensor": {"full_well_e": 12345.0}, "scenes": records}))
    return path


def test_recipe_keeps_hdr_and_uses_one_scale_across_every_frame(tmp_path):
    np.save(tmp_path / "frames.npy", np.stack([
        np.full((2, 4, 3), 0.25, np.float32), np.full((2, 4, 3), 2.0, np.float32)]))
    recipe = _recipe(tmp_path, [_record(input_layout="THWC", frame_times_s=[1e9, 1e9 + 0.001],
        radiance_scale=4.0, clean_reference_verified=True, dependency_ids=["original_clip"],
        provenance={"capture": "declared clean HDR", "hdr_recovery_claim": True})])
    manifest = _api().import_source_recipe(recipe, tmp_path / "out")
    from capture_tm.data import load_manifest
    sensor, scenes = load_manifest(manifest)
    scene = scenes[0]
    assert sensor.full_well_e == 12345.0
    assert scene.frames[:, 0, 0, 0].tolist() == [1.0, 8.0]
    assert scene.frame_times_s.dtype == torch.float64
    assert scene.frame_times_s[1] - scene.frame_times_s[0] == pytest.approx(0.001, abs=1e-7)
    assert scene.provenance["radiance_scale"] == 4.0
    assert scene.provenance["clean_reference_verified"] is True
    assert scene.provenance["hdr_recovery_claim"] is False
    assert scene.provenance["dependency_ids"] == ["original_clip"]
    assert scene.provenance["capture"] == "declared clean HDR"
    assert not Path(json.loads(manifest.read_text())["scenes"][0]["frames_path"]).is_absolute()


def test_recipe_linear_xyz_uses_explicit_matrix_without_upper_clip(tmp_path):
    xyz = np.zeros((3, 2, 4), np.float64)
    xyz[0], xyz[1], xyz[2] = 0.5, 1.0, 2.0
    np.savez(tmp_path / "xyz.npz", frames=xyz)
    recipe = _recipe(tmp_path, [_record(input_path="xyz.npz", input_layout="CHW", input_domain="linear_xyz",
        xyz_to_sensor=[[2, 0, 0], [0, 3, 0], [0, 0, 4]])])
    manifest = _api().import_source_recipe(recipe, tmp_path / "out")
    scene = next(_api().iter_source_scenes(manifest))[1]
    assert scene.frames[0, :, 0, 0].tolist() == [1.0, 3.0, 8.0]
    assert scene.provenance["clean_reference_verified"] is False


def test_recipe_imports_explicit_frame_sequence_and_mask_relative_to_recipe(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    np.save(source_dir / "a.npy", np.full((2, 4, 3), 0.4, np.float32))
    np.savez(source_dir / "b.npz", frames=np.full((2, 4, 3), 1.6, np.float32))
    np.savez(source_dir / "mask.npz", mask=np.full((1, 2, 4), 0.75, np.float32))
    record = _record(frame_paths=["sources/a.npy", "sources/b.npz"], frame_times_s=[-0.1, 0.1],
        subject_mask_path="sources/mask.npz")
    record.pop("input_path")
    recipe = _recipe(tmp_path, [record])
    monkeypatch.chdir(tmp_path.parent)
    manifest = _api().import_source_recipe(recipe, tmp_path / "out")
    scene = next(_api().iter_source_scenes(manifest))[1]
    assert scene.frames.shape == (2, 3, 2, 4)
    assert scene.frames[:, 0, 0, 0].tolist() == pytest.approx([0.4, 1.6])
    assert scene.subject_mask.shape == (1, 2, 4)
    assert scene.subject_mask.mean() == 0.75


@pytest.mark.parametrize("change,match", [
    ({"input_domain": "srgb"}, "domain"),
    ({"input_layout": "guess"}, "layout"),
    ({"radiance_scale": 0}, "scale"),
    ({"radiance_scale": True}, "scale"),
    ({"clean_reference_verified": "yes"}, "clean_reference_verified"),
    ({"frame_times_s": [True]}, "time"),
    ({"frame_times_s": []}, "time"),
    ({"source_id": ""}, "source_id"),
    ({"dependency_ids": "original"}, "dependency_ids"),
    ({"provenance": {"limitations": "clipped source"}}, "limitations"),
    ({"input_domain": "linear_xyz"}, "xyz_to_sensor"),
])
def test_recipe_rejects_ambiguous_or_invalid_photometry_before_output(tmp_path, change, match):
    np.save(tmp_path / "frames.npy", np.full((2, 4, 3), 2.0, np.float32))
    recipe = _recipe(tmp_path, [_record(**change)])
    with pytest.raises(ValueError, match=match):
        _api().import_source_recipe(recipe, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_recipe_requires_explicit_singleton_frame_time(tmp_path):
    np.save(tmp_path / "frames.npy", np.full((2, 4, 3), 2.0, np.float32))
    record = _record()
    record.pop("frame_times_s")
    with pytest.raises(ValueError, match="frame_times_s"):
        _api().import_source_recipe(_recipe(tmp_path, [record]), tmp_path / "out")


@pytest.mark.parametrize("times", [[0, 0], [0.1, 0], [0, float("nan")]])
def test_recipe_rejects_dynamic_time_order_and_nonfinite_times(tmp_path, times):
    np.save(tmp_path / "frames.npy", np.ones((2, 3, 2, 4), np.float32))
    with pytest.raises(ValueError, match="time"):
        _api().import_source_recipe(_recipe(tmp_path, [_record(input_layout="TCHW", frame_times_s=times)]), tmp_path / "out")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_recipe_rejects_integer_arrays_claiming_linear_radiance(tmp_path, dtype):
    np.save(tmp_path / "frames.npy", np.ones((2, 4, 3), dtype))
    with pytest.raises(ValueError, match="float"):
        _api().import_source_recipe(_recipe(tmp_path, [_record()]), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("leak", ["source", "dependency", "source_dependency", "path"])
def test_recipe_preflights_all_source_groups_before_decoding_or_writing(tmp_path, leak):
    # Invalid payloads deliberately distinguish preflight from attempted decoding.
    (tmp_path / "frames.npy").write_bytes(b"not numpy")
    (tmp_path / "other.npy").write_bytes(b"not numpy")
    first = _record(dependency_ids=["shared_reference"])
    second = _record(scene_id="other", source_id="other", input_path="other.npy", split="test")
    if leak == "source":
        second["source_id"] = first["source_id"]
    elif leak == "dependency":
        second["dependency_ids"] = ["shared_reference"]
    elif leak == "source_dependency":
        second["dependency_ids"] = [first["source_id"]]
    else:
        second["input_path"] = "./frames.npy"
    with pytest.raises(ValueError, match="split"):
        _api().import_source_recipe(_recipe(tmp_path, [first, second]), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_recipe_failure_on_later_scene_leaves_no_partial_dataset(tmp_path):
    np.save(tmp_path / "frames.npy", np.ones((2, 4, 3), np.float32))
    np.save(tmp_path / "bad.npy", np.ones((2, 4, 3), np.uint8))
    records = [_record(), _record(scene_id="second", source_id="second", input_path="bad.npy")]
    with pytest.raises(ValueError, match="float"):
        _api().import_source_recipe(_recipe(tmp_path, records), tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))


def test_recipe_copied_content_cannot_cross_splits_with_different_ids(tmp_path):
    np.save(tmp_path / "frames.npy", np.ones((2, 4, 3), np.float32))
    np.savez(tmp_path / "copy.npz", frames=np.ones((2, 4, 3), np.float32))
    records = [_record(), _record(scene_id="copy", source_id="copy", input_path="copy.npz", split="test")]
    with pytest.raises(ValueError, match="split"):
        _api().import_source_recipe(_recipe(tmp_path, records), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_original_decoded_source_cannot_cross_splits_after_scale_augmentation(tmp_path):
    array = np.full((2, 4, 3), 2.0, np.float32)
    np.save(tmp_path / "frames.npy", array)
    np.savez(tmp_path / "copy.npz", frames=array)
    records = [_record(), _record(scene_id="copy", source_id="copy", input_path="copy.npz", split="test", radiance_scale=0.5)]
    with pytest.raises(ValueError, match="split"):
        _api().import_source_recipe(_recipe(tmp_path, records), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_iterator_preflights_original_content_identity_before_augmentation(tmp_path):
    array = np.full((2, 4, 3), 2.0, np.float32)
    np.save(tmp_path / "frames.npy", array)
    np.savez(tmp_path / "copy.npz", frames=array)
    records = [_record(), _record(scene_id="copy", source_id="copy", input_path="copy.npz", radiance_scale=0.5,
        provenance={"original_frames_sha256": "caller claim must be replaced"})]
    manifest = _api().import_source_recipe(_recipe(tmp_path, records), tmp_path / "out")
    payload = json.loads(manifest.read_text())
    first, second = payload["scenes"]
    assert len(first["provenance"]["original_frames_sha256"]) == 64
    assert first["provenance"]["original_frames_sha256"] == second["provenance"]["original_frames_sha256"]
    second["split"] = "test"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="split"):
        next(_api().iter_source_scenes(manifest))


def test_recipe_refuses_to_overwrite_existing_output_directory(tmp_path):
    np.save(tmp_path / "frames.npy", np.ones((2, 4, 3), np.float32))
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        _api().import_source_recipe(_recipe(tmp_path, [_record()]), output)
    assert sentinel.read_text() == "keep"


def test_optional_image_decoder_failure_explains_float_codec_requirement(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name in {"imageio.v3", "OpenEXR"}:
            raise ImportError("optional image codec unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    (tmp_path / "frame.exr").write_bytes(b"codec fixture")
    recipe = _recipe(tmp_path, [_record(input_path="frame.exr")])
    with pytest.raises(ImportError, match="imageio.*EXR|EXR.*imageio"):
        _api().import_source_recipe(recipe, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_iterator_checks_dependency_groups_before_yielding_first_scene(tmp_path):
    from capture_tm.data import write_demo_dataset
    manifest = write_demo_dataset(tmp_path / "dataset", size=8, scenes=3)
    payload = json.loads(manifest.read_text())
    payload["scenes"][0]["provenance"]["dependency_ids"] = ["same_reference"]
    payload["scenes"][2]["provenance"]["dependency_ids"] = ["same_reference"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="split"):
        next(_api().iter_source_scenes(manifest))


def test_iterator_loads_only_current_scene_and_recomputes_content_identity(tmp_path):
    from capture_tm.data import write_demo_dataset
    manifest = write_demo_dataset(tmp_path / "dataset", size=8, scenes=3)
    payload = json.loads(manifest.read_text())
    payload["scenes"][0]["provenance"]["frames_sha256"] = "untrusted"
    manifest.write_text(json.dumps(payload))
    iterator = _api().iter_source_scenes(manifest)
    sensor, first = next(iterator)
    assert sensor.full_well_e == 10000.0
    assert len(first.provenance["frames_sha256"]) == 64
    # A genuinely lazy reader sees a payload mutation after its first yield.
    second_path = manifest.parent / payload["scenes"][1]["frames_path"]
    # The generated dynamic scene has 33 timestamps.
    np.save(second_path, np.zeros((33, 3, 8, 8), np.float32))
    _, second = next(iterator)
    assert second.frames.count_nonzero() == 0


def test_iterator_detects_repackaged_cross_split_content_while_streaming(tmp_path):
    from capture_tm.data import write_demo_dataset
    manifest = write_demo_dataset(tmp_path / "dataset", size=8, scenes=3)
    payload = json.loads(manifest.read_text())
    first, second = payload["scenes"][:2]
    np.savez(manifest.parent / "copy.npz", frames=np.load(manifest.parent / first["frames_path"]))
    second["frames_path"] = "copy.npz"
    second["frame_times_s"] = first["frame_times_s"]
    second["provenance"]["frames_sha256"] = "different claimed content"
    manifest.write_text(json.dumps(payload))
    iterator = _api().iter_source_scenes(manifest)
    next(iterator)
    with pytest.raises(ValueError, match="split"):
        next(iterator)


def test_recipe_preserves_caller_declared_source_limitations(tmp_path):
    np.save(tmp_path / "frames.npy", np.ones((2, 4, 3), np.float32))
    recipe = _recipe(tmp_path, [_record(provenance={"limitations": ["cropped to avoid sensor defects"]})])
    scene = next(_api().iter_source_scenes(_api().import_source_recipe(recipe, tmp_path / "out")))[1]
    assert "cropped to avoid sensor defects" in scene.provenance["limitations"]
    assert any("calibration" in limitation for limitation in scene.provenance["limitations"])


@pytest.mark.parametrize("layout", ["CHW", "HWC", "TCHW", "THWC"])
def test_explicit_layouts_keep_rgb_channel_order(tmp_path, layout):
    hwc = np.zeros((2, 4, 3), np.float32)
    hwc[..., 0], hwc[..., 1], hwc[..., 2] = 0.1, 1.0, 4.0
    array = np.moveaxis(hwc, -1, 0) if "CHW" in layout else hwc
    if layout.startswith("T"):
        array = array[None]
    np.save(tmp_path / "frames.npy", array)
    manifest = _api().import_source_recipe(_recipe(tmp_path, [_record(input_layout=layout)]), tmp_path / "out")
    assert next(_api().iter_source_scenes(manifest))[1].frames[0, :, 0, 0].tolist() == pytest.approx([0.1, 1, 4])


def test_same_source_augmentations_are_allowed_within_one_split(tmp_path):
    np.save(tmp_path / "frames.npy", np.ones((2, 4, 3), np.float32))
    records = [_record(), _record(scene_id="crop", dependency_ids=["original"], radiance_scale=0.5)]
    manifest = _api().import_source_recipe(_recipe(tmp_path, records), tmp_path / "out")
    assert [scene.frames.max().item() for _, scene in _api().iter_source_scenes(manifest)] == [1.0, 0.5]


@pytest.mark.parametrize("layout", ["TCHW", "THWC"])
def test_frame_paths_supply_the_temporal_axis_for_explicit_sequence_layouts(tmp_path, layout):
    array = np.zeros((2, 4, 3), np.float32)
    array[..., 0], array[..., 1], array[..., 2] = 0.5, 2, 4
    if layout == "TCHW":
        array = np.moveaxis(array, -1, 0)
    np.save(tmp_path / "a.npy", array)
    np.save(tmp_path / "b.npy", array * 2)
    record = _record(frame_paths=["a.npy", "b.npy"], input_layout=layout, frame_times_s=[-0.1, 0.1])
    record.pop("input_path")
    manifest = _api().import_source_recipe(_recipe(tmp_path, [record]), tmp_path / "out")
    scene = next(_api().iter_source_scenes(manifest))[1]
    assert scene.frames[:, :, 0, 0].tolist() == [[0.5, 2, 4], [1, 4, 8]]


@pytest.mark.parametrize("failure", ["count", "negative", "nonfinite", "mask_geometry", "mask_range"])
def test_invalid_scene_payload_cannot_publish_a_dataset(tmp_path, failure):
    array = np.ones((2, 4, 3), np.float32)
    record = _record()
    if failure == "count":
        record["frame_times_s"] = [0.0, 0.1]
    elif failure == "negative":
        array[0, 0, 0] = -0.1
    elif failure == "nonfinite":
        array[0, 0, 0] = np.inf
    elif failure == "mask_geometry":
        np.save(tmp_path / "mask.npy", np.ones((1, 2, 5), np.float32))
        record["subject_mask_path"] = "mask.npy"
    else:
        np.save(tmp_path / "mask.npy", np.full((1, 2, 4), 1.1, np.float32))
        record["subject_mask_path"] = "mask.npy"
    np.save(tmp_path / "frames.npy", array)
    with pytest.raises(ValueError):
        _api().import_source_recipe(_recipe(tmp_path, [record]), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_recipe_dataset_still_loads_after_directory_move(tmp_path):
    np.save(tmp_path / "frames.npy", np.full((2, 4, 3), 2.0, np.float32))
    manifest = _api().import_source_recipe(_recipe(tmp_path, [_record()]), tmp_path / "out")
    moved = tmp_path / "renamed"
    manifest.parent.rename(moved)
    assert next(_api().iter_source_scenes(moved / "manifest.json"))[1].frames.max() == 2.0


@pytest.mark.parametrize("field,value", [("split", []), ("input_domain", []), ("input_layout", [])])
def test_malformed_metadata_has_actionable_validation_error(tmp_path, field, value):
    np.save(tmp_path / "frames.npy", np.ones((2, 4, 3), np.float32))
    with pytest.raises(ValueError, match=field):
        _api().import_source_recipe(_recipe(tmp_path, [_record(**{field: value})]), tmp_path / "out")


def test_native_float_exr_sequence_preserves_hdr_and_rgb_channel_order(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    for name, multiplier in [("a.exr", 1), ("b.exr", 2)]:
        channels = {channel: np.full((2, 4), level * multiplier, np.float32)
            for channel, level in [("R", 0.5), ("G", 2.0), ("B", 8.0)]}
        OpenEXR.File({"type": OpenEXR.scanlineimage}, channels).write(str(tmp_path / name))
    record = _record(frame_paths=["a.exr", "b.exr"], input_layout="THWC", frame_times_s=[-0.1, 0.1])
    record.pop("input_path")
    manifest = _api().import_source_recipe(_recipe(tmp_path, [record]), tmp_path / "out")
    scene = next(_api().iter_source_scenes(manifest))[1]
    assert scene.frames[:, :, 0, 0].tolist() == [[0.5, 2.0, 8.0], [1.0, 4.0, 16.0]]


@pytest.mark.parametrize("extra_channels", [False, True])
def test_native_exr_rejects_missing_or_ambiguous_extra_channels(tmp_path, extra_channels):
    OpenEXR = pytest.importorskip("OpenEXR")
    channels = {name: np.ones((2, 4), np.float32) for name in (["R", "G", "B", "A"] if extra_channels else ["Y"])}
    OpenEXR.File({"type": OpenEXR.scanlineimage}, channels).write(str(tmp_path / "frame.exr"))
    with pytest.raises(ValueError, match="R.*G.*B|channels"):
        _api().import_source_recipe(_recipe(tmp_path, [_record(input_path="frame.exr")]), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_native_exr_rejects_ambiguous_multipart_file(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    channels = {name: np.ones((2, 4), np.float32) for name in ["R", "G", "B"]}
    parts = [OpenEXR.Part({"type": OpenEXR.scanlineimage}, channels, name) for name in ["one", "two"]]
    OpenEXR.File(parts).write(str(tmp_path / "frame.exr"))
    with pytest.raises(ValueError, match="single.part|multipart"):
        _api().import_source_recipe(_recipe(tmp_path, [_record(input_path="frame.exr")]), tmp_path / "out")


def test_native_exr_rejects_deep_pixels(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    pixels = np.empty((2, 4), dtype=object)
    for index in np.ndindex(pixels.shape):
        pixels[index] = np.array([1.0], np.float32)
    channels = {name: pixels for name in ["R", "G", "B"]}
    OpenEXR.File({"type": OpenEXR.deepscanline, "compression": OpenEXR.ZIPS_COMPRESSION}, channels).write(str(tmp_path / "frame.exr"))
    with pytest.raises(ValueError, match="deep"):
        _api().import_source_recipe(_recipe(tmp_path, [_record(input_path="frame.exr")]), tmp_path / "out")
