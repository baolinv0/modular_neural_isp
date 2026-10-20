"""Train and deploy from reusable physical acquisition records."""
import json
from pathlib import Path
import shutil

import pytest
import torch

from capture_tm.types import SensorProfile


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_archived_candidates_prepare_without_recapture_and_keep_fixed_losses(tmp_path, monkeypatch):
    # Re-capturing or transposing repeat/candidate axes silently changes the
    # physical experiment; compare prepared outputs to direct archived renders.
    from capture_tm import capture_plan
    from capture_tm.joint_experiment import _render, build_archive_cache
    from capture_tm.joint_objective import image_metrics
    from capture_tm.learned_tone import ConditionalToneMapper

    def forbidden(*args, **kwargs):
        raise AssertionError("an acquisition archive must never be recaptured")

    monkeypatch.setattr(capture_plan, "observe_previews", forbidden)
    monkeypatch.setattr(capture_plan, "render_capture", forbidden)
    images = torch.empty(2, 2, 3, 16, 16)
    for repeat, candidate, value in ((0, 0, .04), (0, 1, .2), (1, 0, .08), (1, 1, .3)):
        images[repeat, candidate] = value
    record = {
        "scene_id": "archive_scene", "source_id": "original_source", "split": "train",
        "source_kind": "synthetic", "provenance": {"condition": "static"},
        "previews": torch.full((3, 3, 16, 16), .1), "state": torch.zeros(3, 3),
        "preview_metadata": {"causal": True}, "rule_index": 0,
        "target": torch.full((3, 16, 16), .37), "subject_mask": torch.ones(1, 16, 16),
        "images": images, "capture_ev": torch.tensor([[0., 1.], [-1., 2.]]),
        "reliability": torch.full((2, 2, 1, 16, 16), .8),
        "missing": (torch.arange(4).reshape(2, 2, 1, 1, 1) > 1).expand(2, 2, 3, 16, 16),
        "radiance_mse": torch.tensor([[.11, .12], [.13, .14]]),
        "noise_seeds": [7, 9], "num_plans": 2, "render_ev": .5,
    }
    path = tmp_path / "record.pt"
    torch.save(record, path)
    plans = [{"actions": [{"exposure_s": shutter, "analog_gain": 1., "digital_gain": 1.}],
              "centers_s": [0.], "integration_time_s": shutter} for shutter in (.002, .008)]
    archive = {"schemes": {"apple": {"plans": plans, "records": [
        {key: record[key] for key in ("scene_id", "source_id", "split", "source_kind", "provenance")}
        | {"path": str(path)}]}}}
    info = build_archive_cache(archive, "apple", tmp_path / "prepared",
                               candidate_chunk_size=3, render_ev=.5)
    saved = torch.load(info["records"][0]["path"], weights_only=True)
    for key in ("target", "subject_mask", "missing", "radiance_mse", "previews", "state"):
        torch.testing.assert_close(saved[key], record[key])
    assert saved["noise_seeds"] == [7, 9]
    tm = ConditionalToneMapper("apple", trainable=False).eval()
    with torch.no_grad():
        expected = tm(images.flatten(0, 1), capture_ev=record["capture_ev"].flatten(),
                      reliability=record["reliability"].flatten(0, 1), render_ev=.5)
        actual = tm.render_prepared(saved["prepared"], render_ev=.5)
        for repeat in range(2):
            rendered, metrics = _render(saved, tm, repeat, [0, 1])
            direct = expected[repeat * 2:repeat * 2 + 2]
            direct_metrics = image_metrics(direct, record["target"][None].expand(2, -1, -1, -1),
                record["missing"][repeat], subject_mask=record["subject_mask"][None].expand(2, -1, -1, -1))
            torch.testing.assert_close(rendered, direct)
            for key in direct_metrics:
                torch.testing.assert_close(metrics[key], direct_metrics[key])
    torch.testing.assert_close(actual, expected)
    assert all(not value.requires_grad for value in saved["prepared"].values())


def test_missing_requested_archive_scheme_is_reported_without_creating_output(tmp_path):
    from capture_tm.joint_experiment import run_factorial

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 2, "kind": "capture_tm_acquisition_dataset",
                                    "schemes": {"apple": {"plans": [], "records": []}}}))
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="samsung"):
        run_factorial(manifest, output, scheme="both", epochs=1, warmup=0,
                      noise_seeds=(0,), threads=1)
    assert not output.exists()


@pytest.mark.parametrize("field,value,match", [("render_ev", 1., "render_ev|intent"),
                                              ("noise_seeds", [1], "noise_seeds|noise seeds")])
def test_archive_protocol_requests_reject_different_intent_or_noise_before_output(tmp_path, field, value, match):
    from capture_tm.joint_experiment import run_factorial

    payload = {"version": 2, "kind": "capture_tm_acquisition_dataset", "render_ev": .5,
               "noise_seeds": [0], "schemes": {"apple": {"plans": [], "records": []}}}
    payload[field] = value
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    output = tmp_path / "training"
    with pytest.raises(ValueError, match=match):
        run_factorial(manifest, output, scheme="apple", epochs=1, warmup=0,
                      noise_seeds=(0,), threads=1, render_ev=.5)
    assert not output.exists()


@pytest.fixture(scope="module")
def generated_archive(tmp_path_factory):
    from capture_tm.data import write_demo_dataset
    from capture_tm.pipeline import generate_acquisition_dataset

    directory = tmp_path_factory.mktemp("training_archive")
    scenes = write_demo_dataset(directory / "scenes", size=16, scenes=3, seed=11)
    archive = generate_acquisition_dataset(scenes, directory / "archive", scheme="both",
                                            noise_seeds=(0,), threads=1, render_ev=.5)
    return archive


def test_development_archive_loader_does_not_open_official_test(generated_archive, tmp_path):
    # Missing a pre-decode split filter would attempt to open this absent file.
    from capture_tm.pipeline import load_acquisition_manifest

    shutil.copytree(generated_archive.parent, tmp_path / 'archive')
    manifest = tmp_path / 'archive' / 'manifest.json'
    payload = json.loads(manifest.read_text())
    for value in payload['schemes'].values():
        for row in value['records']:
            if row['split'] == 'test':
                row['path'] = 'OFFICIAL_TEST_MUST_NOT_BE_OPENED.pt'
    manifest.write_text(json.dumps(payload))
    loaded = load_acquisition_manifest(manifest, splits={'train', 'val'})
    for value in loaded['schemes'].values():
        assert {row['split'] for row in value['records']} == {'train', 'val'}


def test_relocated_archive_trains_all_eight_groups_and_restores_raw_acquisition(
        generated_archive, tmp_path, monkeypatch):
    # Archive paths must remain relative; moving the complete acquisition must
    # preserve all eight updates without access to latent source frames.
    from capture_tm import acquisition, capture_plan, joint_experiment, pipeline
    from capture_tm.joint_algorithm import JointCaptureAlgorithm
    from capture_tm.joint_experiment import run_factorial

    moved = tmp_path / "relocated"
    shutil.copytree(generated_archive.parent, moved)
    manifest = moved / "manifest.json"
    payload = json.loads(manifest.read_text())
    for value in payload["schemes"].values():
        assert all(not Path(row["path"]).is_absolute() for row in value["records"])

    def forbidden(*args, **kwargs):
        raise AssertionError("archive training must not call the RGB camera simulator")

    monkeypatch.setattr(capture_plan, "observe_previews", forbidden)
    monkeypatch.setattr(capture_plan, "render_capture", forbidden)
    monkeypatch.setattr(acquisition, "capture_raw", forbidden)
    monkeypatch.setattr(pipeline, "observe_raw_previews", forbidden)
    monkeypatch.setattr(pipeline, "compose_captures", forbidden)
    monkeypatch.setattr(joint_experiment, "load_manifest", forbidden)
    output = tmp_path / "training"
    report = run_factorial(manifest, output, scheme="both", epochs=1, warmup=1,
                           seeds=(0,), noise_seeds=(0,), threads=1,
                           candidate_chunk_size=4, render_ev=.5)
    assert report["manifest_kind"] == "capture_tm_acquisition_dataset"
    assert set(report["schemes"]) == {"apple", "samsung"}
    for scheme, prefix in (("apple", "A"), ("samsung", "S")):
        assert set(report["schemes"][scheme]["groups"]) == {prefix + x for x in ("00", "10", "01", "11")}
        initial = torch.load(output / scheme / "seed_0" / "initial.pt", weights_only=True)
        for suffix in ("00", "10", "01", "11"):
            group_dir = output / scheme / "seed_0" / (prefix + suffix)
            checkpoint = torch.load(group_dir / "last.pt", weights_only=True)
            assert checkpoint["acquisition"] == payload["acquisition"]
            assert checkpoint["manifest_kind"] == "capture_tm_acquisition_dataset"
            assert checkpoint["plans"] == payload["schemes"][scheme]["plans"]
            assert checkpoint["sensor"] == payload["sensor"]
            for module, learned in (("policy_state", suffix[0] == "1"), ("tm_state", suffix[1] == "1")):
                changed = any(not torch.equal(value, initial[module][key])
                              for key, value in checkpoint[module].items())
                assert changed == learned
            evaluation = json.loads((group_dir / "evaluation.json").read_text())
            assert evaluation["test_scenes"] == 1
            assert evaluation["per_scene"][0]["scene_id"] == "demo_002"
            assert checkpoint["optimizer_steps"] == {"ae": int(suffix[0]), "tm": int(suffix[1])}
        restored = JointCaptureAlgorithm.from_checkpoint(output / scheme / "seed_0" / (prefix + "11") / "selected.pt")
        assert restored.acquisition.to_dict() == payload["acquisition"]
        assert restored.checkpoint_manifest_kind == "capture_tm_acquisition_dataset"
        assert restored.checkpoint_render_ev == .5


@pytest.mark.parametrize("mutation,match", [
    ("missing_scheme", "scheme|samsung"),
    ("missing_split", "train, val and test"),
    ("invalid_profile", "readout"),
    ("manifest_intent", "render_ev|intent"),
    ("record_intent", "render_ev|intent"),
    ("noise_seeds", "noise_seeds|noise seeds"),
])
def test_archive_rejections_happen_before_creating_results(generated_archive, tmp_path, mutation, match):
    from capture_tm.joint_experiment import run_factorial

    directory = tmp_path / "invalid"
    shutil.copytree(generated_archive.parent, directory)
    manifest = directory / "manifest.json"
    payload = json.loads(manifest.read_text())
    if mutation == "missing_scheme":
        del payload["schemes"]["samsung"]
    elif mutation == "missing_split":
        payload["schemes"]["apple"]["records"] = [
            row for row in payload["schemes"]["apple"]["records"] if row["split"] != "val"]
    elif mutation == "invalid_profile":
        payload["acquisition"]["readout_s"] = -1
    elif mutation == "manifest_intent":
        payload["render_ev"] = 1.
    elif mutation == "record_intent":
        row = payload["schemes"]["apple"]["records"][0]
        path = directory / row["path"]
        record = torch.load(path, weights_only=True)
        record["render_ev"] = 1.
        torch.save(record, path)
    elif mutation == "noise_seeds":
        payload["noise_seeds"] = [1]
    manifest.write_text(json.dumps(payload))
    output = tmp_path / "training"
    with pytest.raises(ValueError, match=match):
        run_factorial(manifest, output, scheme="both", epochs=1, warmup=0,
                      noise_seeds=(0,), threads=1, render_ev=.5)
    assert not output.exists()


def test_acquisition_checkpoint_cannot_silently_restore_legacy_rgb_without_profile(tmp_path):
    from capture_tm.joint_algorithm import JointCaptureAlgorithm

    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"version": 1, "scheme": "apple", "sensor": SensorProfile().to_dict(),
                "plans": [], "tm_state": {}, "learned_ae": False,
                "manifest_kind": "capture_tm_acquisition_dataset"}, checkpoint)
    with pytest.raises(ValueError, match="acquisition"):
        JointCaptureAlgorithm.from_checkpoint(checkpoint)


@pytest.mark.parametrize("scheme,frames", [("apple", 1), ("samsung", 3)])
def test_raw_deployment_uses_bound_acquisition_for_historical_and_final_frames(scheme, frames):
    # The optional profile must change acquisition itself, including previews;
    # merely attaching metadata to legacy RGB results would lose Bayer RAW.
    from capture_tm.acquisition import AcquisitionProfile
    from capture_tm.joint_algorithm import JointCaptureAlgorithm
    from capture_tm.types import Scene

    scene = Scene("raw_deployment", "test", torch.full((1, 3, 32, 32), .1),
                  torch.tensor([0.], dtype=torch.float64))
    profile = AcquisitionProfile(spatial_downsample=2)
    algorithm = JointCaptureAlgorithm(scheme, SensorProfile(), acquisition=profile)
    result = algorithm.run_simulated(scene, seed=3, noisy=False)
    assert result["observations"]["previews"].shape == (3, 3, 16, 16)
    assert result["output"].shape == (3, 16, 16)
    assert len(result["captures"]) == frames
    assert all(capture.raw_dn.shape == (1, 16, 16) for capture in result["captures"])
    assert all(capture.action.digital_gain == 1. for capture in result["captures"])
    assert max(result["observations"]["metadata"]["readout_ends_s"]) < min(
        capture.metadata["shutter_interval_s"][0] for capture in result["captures"])


def test_physical_request_declares_the_bound_native_acquisition_profile():
    from capture_tm.acquisition import AcquisitionProfile
    from capture_tm.joint_algorithm import JointCaptureAlgorithm

    profile = AcquisitionProfile(cfa_pattern="BGGR", readout_s=.002)
    algorithm = JointCaptureAlgorithm("apple", SensorProfile(), acquisition=profile)
    request = algorithm.select(torch.full((3, 3, 16, 16), .1), torch.zeros(3, 3))
    json.dumps(request, allow_nan=False)
    assert request["acquisition"] == profile.to_dict()


def camera_readback(center, *, profile=None):
    from capture_tm.types import CaptureAction, CaptureResult

    rgb = torch.full((3, 16, 16), .1)
    metadata = {"center_s": center}
    if profile is not None:
        metadata["acquisition"] = profile.to_dict()
    return CaptureResult(rgb, torch.zeros_like(rgb), torch.zeros_like(rgb, dtype=torch.bool),
                         CaptureAction(.006), metadata)


@pytest.mark.parametrize("rolling,readout,spacing", [(.008, 0., .013), (0., .01, .015)])
def test_bound_finish_rejects_effective_rolling_or_readout_overlap(rolling, readout, spacing):
    # Nominal shutters are disjoint; only full row exposure plus readout makes
    # these actual camera readbacks physically incompatible.
    from capture_tm.acquisition import AcquisitionProfile
    from capture_tm.joint_algorithm import JointCaptureAlgorithm

    profile = AcquisitionProfile(rolling_shutter_s=rolling, readout_s=readout)
    algorithm = JointCaptureAlgorithm("samsung", SensorProfile(), acquisition=profile)
    captures = [camera_readback(1000. + index * spacing) for index in range(3)]
    with pytest.raises(ValueError, match="overlap|readout|rolling"):
        algorithm.finish(captures)


def test_bound_finish_accepts_absolute_effective_clocks_and_matching_profile():
    from capture_tm.acquisition import AcquisitionProfile
    from capture_tm.joint_algorithm import JointCaptureAlgorithm

    profile = AcquisitionProfile(rolling_shutter_s=.008, readout_s=.002)
    algorithm = JointCaptureAlgorithm("samsung", SensorProfile(), acquisition=profile)
    captures = [camera_readback(1000. + index * .021, profile=profile) for index in range(3)]
    result = algorithm.finish(captures)
    assert torch.isfinite(result["output"]).all()
    assert result["metadata"]["effective_centers_s"] == [1000., 1000.021, 1000.042]


def test_bound_finish_rejects_measurements_from_a_different_acquisition_profile():
    from capture_tm.acquisition import AcquisitionProfile
    from capture_tm.joint_algorithm import JointCaptureAlgorithm

    algorithm = JointCaptureAlgorithm("apple", SensorProfile(), acquisition=AcquisitionProfile())
    capture = camera_readback(1000., profile=AcquisitionProfile(cfa_pattern="BGGR"))
    with pytest.raises(ValueError, match="acquisition.*profile"):
        algorithm.finish([capture])
