import json
import numpy as np
import pytest
import torch


def _api():
    from capture_tm.data import load_manifest, write_demo_dataset
    from capture_tm.sources import import_rawgen_xyz, import_rl3a_proxy, import_linear_scene
    return load_manifest, write_demo_dataset, import_rawgen_xyz, import_rl3a_proxy, import_linear_scene


def test_demo_is_deterministic_hdr_motion_and_split_by_source(tmp_path):
    load, demo, *_ = _api()
    first = demo(tmp_path / "one", seed=10, size=12, scenes=6)
    second = demo(tmp_path / "two", seed=10, size=12, scenes=6)
    sensor, scenes = load(first)
    _, repeated = load(second)
    assert {s.split for s in scenes} == {"train","val","test"}
    assert len({s.source_id for s in scenes}) == 6
    assert any(s.frames.max() > 1. for s in scenes)
    assert any(len(s.frames) > 1 and not torch.equal(s.frames[0], s.frames[-1]) for s in scenes)
    assert all(torch.equal(a.frames,b.frames) for a,b in zip(scenes,repeated))
    assert sensor.reference_exposure_s > 0.


@pytest.mark.parametrize("count,expected",[(3,(1,1,1)),(9,(5,1,3)),(24,(14,4,6))])
def test_demo_allocates_train_val_test_by_integer_sixty_twenty_remainder(tmp_path,count,expected):
    load,demo,*_ = _api()
    _,scenes = load(demo(tmp_path,size=8,scenes=count))
    actual = tuple(sum(scene.split==split for scene in scenes) for split in ("train","val","test"))
    assert actual == expected


def test_manifest_domain_and_source_overlap_are_rejected(tmp_path):
    load, demo, *_ = _api()
    path = demo(tmp_path, size=8, scenes=3)
    payload = json.loads(path.read_text())
    payload["scenes"][0].pop("domain")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="domain"):
        load(path)
    payload["scenes"][0]["domain"] = "sensor_linear_relative_radiance"
    payload["scenes"][1]["source_id"] = payload["scenes"][0]["source_id"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="split"):
        load(path)


def test_rawgen_import_decodes_oetf_once_and_preserves_matrix_scale(tmp_path):
    load, _, rawgen, *_ = _api()
    source = tmp_path / "encoded_xyz.npy"
    np.save(source, np.full((6,6,3), .5, np.float32))
    path = rawgen(source, tmp_path / "converted", scene_id="rawgen", split="train", xyz_to_sensor=np.eye(3)*2)
    _, scenes = load(path)
    assert scenes[0].frames.mean().item() == pytest.approx(.42808228, abs=1e-6)
    assert scenes[0].provenance["input_domain"] == "rawgen_oetf_xyz"
    assert scenes[0].provenance["absolute_radiance_calibrated"] is False
    assert scenes[0].provenance["hdr_recovery_claim"] is False


def test_rawgen_requires_known_xyz_domain_instead_of_camera_preview_guess(tmp_path):
    _, _, rawgen, *_ = _api()
    source = tmp_path / "preview.npy"
    np.save(source, np.ones((4,4,3), np.float32))
    with pytest.raises(ValueError, match="domain"):
        rawgen(source,tmp_path / "out",scene_id="wrong",split="train",xyz_to_sensor=np.eye(3),input_domain="rawgen_camera_preview")


def test_rl3a_reference_exposure_inversion_and_no_clean_hdr_claim(tmp_path):
    load, _, _, proxy, _ = _api()
    source = tmp_path / "linear.npy"
    np.save(source, np.full((4,4,3), .6, np.float32))
    path = proxy(source,tmp_path / "out",scene_id="source",split="val",exposure_s=1/60,analog_gain=3.,digital_gain=1.,reference_exposure_s=1/120)
    _, scenes = load(path)
    assert scenes[0].frames.mean().item() == pytest.approx(.1)
    assert scenes[0].provenance["clean_reference_verified"] is False
    assert scenes[0].provenance["hdr_recovery_claim"] is False


def test_explicit_linear_import_preserves_hdr_and_relative_paths(tmp_path,monkeypatch):
    load, _, _, _, linear = _api()
    source = tmp_path / "hdr.npy"
    np.save(source,np.full((4,4,3),2.5,np.float32))
    path = linear(source,tmp_path / "out",scene_id="hdr",split="test",input_domain="sensor_linear_relative_radiance")
    monkeypatch.chdir(tmp_path.parent)
    _, scenes = load(path)
    assert scenes[0].frames.max() == 2.5
    payload = json.loads(path.read_text())
    assert not __import__("pathlib").Path(payload["scenes"][0]["frames_path"]).is_absolute()


def test_integer_numpy_frames_are_rejected_as_ambiguous_normalization(tmp_path):
    _, _, _, _, linear = _api()
    source = tmp_path / "ambiguous.npy"
    np.save(source,np.ones((4,4,3),np.uint16))
    with pytest.raises(ValueError,match="float"):
        linear(source,tmp_path / "out",scene_id="int",split="test",input_domain="sensor_linear_relative_radiance")


def test_rl3a_png16_import_preserves_every_code_and_rgb_channel(tmp_path):
    import png
    load, _, _, proxy, _ = _api()
    codes = np.array([1,255,256,257,32768,65535],np.uint16)
    rgb = np.stack([codes,codes[::-1],np.full(6,1000,np.uint16)],axis=-1)[None]
    source = tmp_path / "rgb16.png"
    with source.open("wb") as handle:
        png.Writer(6,1,greyscale=False,bitdepth=16).write(handle,rgb.reshape(1,-1).tolist())
    path = proxy(source,tmp_path/"out",scene_id="png",split="test",exposure_s=1/120,analog_gain=1.)
    _, scenes = load(path)
    recovered = scenes[0].frames[0].permute(1,2,0).numpy()
    assert np.allclose(recovered,rgb.astype(np.float32)/65535,atol=1e-8)


def test_import_append_rejects_cross_split_source_without_overwriting(tmp_path):
    load, _, _, _, linear = _api()
    source = tmp_path/"src.npy"
    np.save(source,np.ones((4,4,3),np.float32))
    path = linear(source,tmp_path/"out",scene_id="first",split="train",source_id="same",input_domain="sensor_linear_relative_radiance")
    before = path.read_bytes()
    with pytest.raises(ValueError,match="split"):
        linear(source,tmp_path/"out",scene_id="second",split="test",source_id="same",input_domain="sensor_linear_relative_radiance")
    assert path.read_bytes() == before
    assert len(load(path)[1]) == 1


def test_import_source_hash_groups_same_file_despite_different_scene_names(tmp_path):
    _, _, _, _, linear = _api()
    source = tmp_path/"src.npy"
    np.save(source,np.ones((4,4,3),np.float32))
    linear(source,tmp_path/"out",scene_id="first",split="train",input_domain="sensor_linear_relative_radiance")
    with pytest.raises(ValueError,match="split"):
        linear(source,tmp_path/"out",scene_id="same_file_variant",split="test",input_domain="sensor_linear_relative_radiance")


def test_manifest_npz_float_data_loads_and_integer_data_fails(tmp_path):
    load, demo, *_ = _api()
    path = demo(tmp_path,size=8,scenes=3)
    payload = json.loads(path.read_text())
    filename = tmp_path / "replacement.npz"
    np.savez(filename,frames=np.ones((1,3,8,8),np.float64)*1.5)
    payload["scenes"][0]["frames_path"] = filename.name
    payload["scenes"][0]["frame_times_s"] = [0.]
    path.write_text(json.dumps(payload))
    assert load(path)[1][0].frames.max() == 1.5
    np.savez(filename,frames=np.ones((1,3,8,8),np.uint16))
    with pytest.raises(ValueError,match="float"):
        load(path)


def test_linear_xyz_rejects_camera_dng_before_any_xyz_transform(tmp_path):
    _, _, _, _, linear = _api()
    path = tmp_path/"camera.dng"
    path.write_bytes(b"not a DNG")
    with pytest.raises(ValueError,match="domain"):
        linear(path,tmp_path/"out",scene_id="bad",split="test",input_domain="linear_xyz",xyz_to_sensor=np.eye(3))


def test_rl3a_noise_bridge_names_natural_residual_domain_without_sensor_calibration(tmp_path):
    from capture_tm.sources import load_rl3a_noise_profile
    path = tmp_path/"fit.json"
    payload = {"model":"iso_shot_read_v1","iso_ref":100.,"parameters":{"a":3.904104886817732e-5,"b":6.599337306148046e-26,"c":1.9171109793749755e-6},
        "calibration":{"clean_proxy":"Lightroom AI denoised RAW; not a physical noiseless reference"}}
    path.write_text(json.dumps(payload))
    with pytest.warns(UserWarning,match="not calibrated"):
        bridge = load_rl3a_noise_profile(path)
    assert bridge["variance_domain"] == "normalized_demosaiced_RGB"
    assert bridge["calibrated_sensor"] is False
    assert "full_well_e" not in bridge
    payload["iso_ref"] = "not a numeric ISO"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="malformed"):
        load_rl3a_noise_profile(path)


@pytest.mark.parametrize("identity",["frames_path","input_sha256"])
def test_manifest_cannot_hide_same_real_source_by_renaming_source_id(tmp_path,identity):
    load,demo,*_ = _api()
    path = demo(tmp_path,size=8,scenes=3)
    payload = json.loads(path.read_text())
    first,second = payload["scenes"][:2]
    if identity == "frames_path":
        second["frames_path"] = first["frames_path"]
        second["frame_times_s"] = first["frame_times_s"]
    else:
        first["provenance"]["input_sha256"] = "one_real_source"
        second["provenance"]["input_sha256"] = "one_real_source"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="split"):
        load(path)


def test_explicit_import_source_id_cannot_override_same_file_split(tmp_path):
    _,_,_,_,linear = _api()
    source = tmp_path/"src.npy"
    np.save(source,np.ones((4,4,3),np.float32))
    linear(source,tmp_path/"out",scene_id="first",split="train",source_id="one",input_domain="sensor_linear_relative_radiance")
    with pytest.raises(ValueError,match="split"):
        linear(source,tmp_path/"out",scene_id="second",split="test",source_id="two",input_domain="sensor_linear_relative_radiance")


def test_copied_decoded_frames_cannot_cross_split_under_different_names(tmp_path):
    load,demo,*_ = _api()
    path = demo(tmp_path,size=8,scenes=3)
    payload = json.loads(path.read_text())
    first,second = payload["scenes"][:2]
    original = np.load(tmp_path/first["frames_path"])
    # Different container, filename, source_id and claimed hash; same float data.
    np.savez(tmp_path/"renamed_frames.npz",frames=original)
    second["frames_path"] = "renamed_frames.npz"
    second["frame_times_s"] = first["frame_times_s"]
    first["provenance"] = {"frames_sha256":"invented_first_hash"}
    second["provenance"] = {"frames_sha256":"invented_second_hash"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="split"):
        load(path)


def test_payload_hash_is_computed_from_shape_dtype_bytes_instead_of_metadata(tmp_path):
    load,demo,*_ = _api()
    path = demo(tmp_path,size=8,scenes=3)
    payload = json.loads(path.read_text())
    payload["scenes"][0]["provenance"]["frames_sha256"] = "untrusted"
    payload["scenes"][0]["provenance"]["mask_sha256"] = "untrusted"
    path.write_text(json.dumps(payload))
    _,scenes = load(path)
    original = scenes[0]
    assert len(original.provenance["frames_sha256"]) == 64
    assert len(original.provenance["mask_sha256"]) == 64
    arr = original.frames.numpy()
    np.savez(tmp_path/"another_container.npz",frames=arr.astype(np.float64))
    payload["scenes"][0]["frames_path"] = "another_container.npz"
    path.write_text(json.dumps(payload))
    _,reloaded = load(path)
    assert reloaded[0].provenance["frames_sha256"] == original.provenance["frames_sha256"]
    changed = arr.copy()
    changed.reshape(-1)[0] += .01
    np.savez(tmp_path/"another_container.npz",frames=changed)
    assert load(path)[1][0].provenance["frames_sha256"] != original.provenance["frames_sha256"]


def test_mask_and_frame_geometry_have_independent_actual_payload_fingerprints(tmp_path):
    load,demo,*_ = _api()
    path = demo(tmp_path,size=8,scenes=3)
    payload = json.loads(path.read_text())
    record = payload["scenes"][0]
    original = load(path)[1][0]
    mask_path = tmp_path/record["subject_mask_path"]
    changed_mask = np.load(mask_path).copy()
    changed_mask.reshape(-1)[0] = 1.-changed_mask.reshape(-1)[0]
    np.save(mask_path,changed_mask)
    mask_changed = load(path)[1][0]
    assert mask_changed.provenance["frames_sha256"] == original.provenance["frames_sha256"]
    assert mask_changed.provenance["mask_sha256"] != original.provenance["mask_sha256"]
    frame_path = tmp_path/record["frames_path"]
    np.save(frame_path,np.load(frame_path).reshape(1,3,4,16))
    record.pop("subject_mask_path")
    record["provenance"]["mask_sha256"] = "there is no actual mask"
    path.write_text(json.dumps(payload))
    geometry_changed = load(path)[1][0]
    assert geometry_changed.provenance["frames_sha256"] != original.provenance["frames_sha256"]
    assert "mask_sha256" not in geometry_changed.provenance


def test_import_append_detects_copied_content_after_source_file_hash_changes(tmp_path):
    _,_,_,_,linear = _api()
    source = tmp_path/"source.npy"
    arr = np.full((4,4,3),.3,np.float32)
    np.save(source,arr)
    linear(source,tmp_path/"out",scene_id="first",split="train",source_id="first",input_domain="sensor_linear_relative_radiance")
    copy = tmp_path/"different_container.npz"
    np.savez(copy,frames=arr)
    with pytest.raises(ValueError,match="split"):
        linear(copy,tmp_path/"out",scene_id="second",split="test",source_id="second",input_domain="sensor_linear_relative_radiance")
