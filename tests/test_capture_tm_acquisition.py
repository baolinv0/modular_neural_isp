"""Physics checks for the native Bayer acquisition path."""
import json

import pytest
import torch

from capture_tm.types import CaptureAction, CaptureResult, Scene, SensorProfile


def _api():
    from capture_tm.acquisition import (
        AcquisitionProfile, capture_raw, demosaic_bayer, mosaic_bayer, prepare_scene,
    )
    return AcquisitionProfile, capture_raw, demosaic_bayer, mosaic_bayer, prepare_scene


def _scene(value=.2, height=8, width=8):
    return Scene("static", "train", torch.full((1, 3, height, width), value), torch.tensor([0.]))


def _sensor(**kwargs):
    return SensorProfile(read_noise_e=0., adc_noise_dn=0., bit_depth=20, **kwargs)


@pytest.mark.parametrize("pattern,tile", [
    ("RGGB", [[1., 2.], [2., 3.]]),
    ("BGGR", [[3., 2.], [2., 1.]]),
    ("GRBG", [[2., 1.], [3., 2.]]),
    ("GBRG", [[2., 3.], [1., 2.]]),
])
def test_cfa_channel_phase_and_constant_color_demosaic_at_edges(pattern, tile):
    _, _, demosaic, mosaic, _ = _api()
    rgb = torch.tensor([1., 2., 3.])[:, None, None].expand(3, 4, 6)
    raw = mosaic(rgb, pattern)
    assert raw.shape == (1, 4, 6)
    assert torch.equal(raw[0, :2, :2], torch.tensor(tile))
    assert torch.equal(demosaic(raw, pattern), rgb)
    assert torch.equal(demosaic(torch.zeros_like(raw), pattern), torch.zeros_like(rgb))


def test_demosaic_bilinear_interpolation_and_sample_preservation():
    _, _, demosaic, _, _ = _api()
    raw = torch.zeros((1, 6, 6))
    raw[0, 2, 2] = 1.  # an RGGB red sample
    rgb = demosaic(raw, "RGGB")
    assert rgb[0, 2, 2] == 1.
    assert rgb[0, 2, 3] == .5
    assert rgb[0, 3, 3] == .25
    assert rgb[1:].count_nonzero() == 0


def test_raw_exposure_collects_photons_while_analog_gain_does_not():
    _, capture, _, _, _ = _api()
    sensor = _sensor(full_well_e=10000.)
    base = capture(_scene(.1), CaptureAction(sensor.reference_exposure_s), sensor, noisy=False)
    gain = capture(_scene(.1), CaptureAction(sensor.reference_exposure_s, 2.), sensor, noisy=False)
    long = capture(_scene(.1), CaptureAction(sensor.reference_exposure_s * 2), sensor, noisy=False)
    assert isinstance(base, CaptureResult)
    assert base.raw_dn.shape == base.raw_noise_variance.shape == base.raw_saturation_mask.shape == (1, 8, 8)
    assert base.metadata["expected_photons_mean_e"] == pytest.approx(1000.)
    assert gain.metadata["expected_photons_mean_e"] == pytest.approx(1000.)
    assert long.metadata["expected_photons_mean_e"] == pytest.approx(2000.)
    assert base.rgb.mean().item() == pytest.approx(.1, abs=2e-6)
    assert gain.rgb.mean().item() == pytest.approx(.2, abs=2e-6)
    assert long.raw_noise_variance.mean() < gain.raw_noise_variance.mean()
    assert base.metadata["raw_domain"] == "bayer_adc_dn"
    json.dumps(base.metadata)  # tensor data belong to fields, never metadata


def test_full_well_clips_after_linear_integration_and_before_analog_gain():
    _, capture, _, _, _ = _api()
    frames = torch.zeros((3, 3, 4, 4))
    frames[1] = 4.
    scene = Scene("pulse", "test", frames, torch.tensor([-1., 0., 1.]))
    sensor = _sensor(reference_exposure_s=2.)
    result = capture(scene, CaptureAction(2., .25), sensor, noisy=False)
    assert result.metadata["integrated_radiance_mean"] == pytest.approx(2.)
    assert result.metadata["expected_photons_mean_e"] == pytest.approx(20000.)
    assert result.rgb.mean().item() == pytest.approx(.25, abs=2e-6)
    assert result.raw_saturation_mask.all() and result.saturation_mask.all()
    assert result.metadata["adc_saturation_fraction"] == 0.


def test_exact_temporal_triangle_is_not_clipped_to_unit_radiance():
    _, capture, _, _, _ = _api()
    frames = torch.zeros((3, 3, 4, 4))
    frames[1] = 2.
    scene = Scene("triangle", "test", frames, torch.tensor([-1., 0., 1.]))
    result = capture(scene, CaptureAction(2.), _sensor(reference_exposure_s=4.), noisy=False)
    assert result.metadata["integrated_radiance_mean"] == pytest.approx(1.)
    assert result.rgb.mean().item() == pytest.approx(.5, abs=2e-6)
    assert not result.raw_saturation_mask.any()


def test_rolling_shutter_centers_rows_and_reports_full_interval_plus_readout():
    Profile, capture, _, _, _ = _api()
    frames = torch.zeros((2, 3, 4, 4))
    frames[1] = .4
    scene = Scene("ramp", "test", frames, torch.tensor([-1., 1.]))
    sensor = _sensor(reference_exposure_s=.2)
    result = capture(scene, CaptureAction(.2), sensor, acquisition=Profile(rolling_shutter_s=1., readout_s=.03), noisy=False)
    normalized_raw = (result.raw_dn - sensor.black_level_dn) / (2**sensor.bit_depth - 1 - sensor.black_level_dn)
    assert torch.allclose(normalized_raw[:, :, 0], torch.tensor([[.1, 1/6, 7/30, .3]]), atol=2e-6)
    assert result.metadata["shutter_interval_s"] == pytest.approx([-.6, .6])
    assert result.metadata["readout_end_s"] == pytest.approx(.63)
    with pytest.raises(ValueError, match="support"):
        capture(scene, CaptureAction(.2), sensor, acquisition=Profile(rolling_shutter_s=2.), noisy=False)


def test_temporal_support_does_not_expand_with_large_absolute_timestamps():
    Profile, capture, _, _, _ = _api()
    frames = torch.full((2, 3, 4, 4), .2)
    scene = Scene("epoch", "test", frames, torch.tensor([1e8, 1e8+1], dtype=torch.float64))
    result = capture(scene, CaptureAction(.01), _sensor(), acquisition=Profile(), center_s=1e8+.5, noisy=False)
    assert result.metadata["integrated_radiance_mean"] == pytest.approx(.2, abs=1e-7)
    with pytest.raises(ValueError, match="support"):
        capture(scene, CaptureAction(.02), _sensor(), center_s=1e8+.001, noisy=False)


def test_prepare_scene_averages_pixel_area_without_changing_intensity_or_reference_time():
    Profile, capture, _, _, prepare = _api()
    frames = torch.zeros((2, 3, 8, 8))
    frames[0, :, :2, :2] = 4.
    frames[1] = frames[0] * .5
    scene = Scene("area", "test", frames, torch.tensor([-1., 1.]), subject_mask=torch.ones((1, 8, 8)))
    profile = Profile(spatial_downsample=2)
    native = prepare(scene, profile)
    assert native.frames.shape == (2, 3, 4, 4)
    assert native.frames[0, :, 0, 0].tolist() == [4., 4., 4.]
    assert native.frames.mean() == scene.frames.mean()
    assert native.subject_mask.shape == (1, 4, 4)
    assert torch.equal(native.frame_times_s, scene.frame_times_s)
    again = prepare(native, profile)
    assert torch.equal(again.frames, native.frames)
    result = capture(native, CaptureAction(.1), _sensor(), acquisition=profile, noisy=False)
    assert result.raw_dn.shape == (1, 4, 4)
    assert result.metadata["source_resolution"] == [8, 8]
    assert result.metadata["native_resolution"] == [4, 4]


def test_psf_runs_only_on_pre_optics_and_precedes_area_reduction():
    Profile, _, _, _, prepare = _api()
    scene = _scene(0., 8, 8)
    scene.frames[:, :, 3, 3] = 1.
    kernels = [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
    before = prepare(scene, Profile(input_stage="pre_optics", spatial_downsample=2, psf_kernels=kernels))
    after = prepare(scene, Profile(input_stage="post_optics", spatial_downsample=2))
    assert torch.allclose(before.frames[0, 0, 1:3, 1:3], torch.tensor([[1/9, 1/18], [1/18, 1/36]]))
    assert after.frames[0, 0, 1, 1] == .25
    assert after.frames[0, 0].count_nonzero() == 1
    assert before.frames.sum().item() == pytest.approx(after.frames.sum().item())


def test_post_optics_configuration_rejects_an_additional_psf_instead_of_silently_ignoring_it():
    Profile, _, _, _, _ = _api()
    with pytest.raises(ValueError, match="post_optics"):
        Profile(input_stage="post_optics", psf_kernels=[[1.]])


def test_asymmetric_psf_is_a_point_response_convolution_not_reversed_correlation():
    Profile, _, _, _, prepare = _api()
    scene = _scene(0., 8, 8)
    scene.frames[:, :, 3, 3] = 1.
    profile = Profile(input_stage="pre_optics", psf_kernels=[[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]])
    native = prepare(scene, profile)
    assert native.frames[0, :, 3, 4].tolist() == [1., 1., 1.]
    assert native.frames[0, :, 3, 2].tolist() == [0., 0., 0.]
    assert native.frames.count_nonzero() == 3
    assert native.provenance["physical_acquisition"]["psf_convention"] == "point_response_convolution"


def test_seeded_cfa_shot_noise_matches_photons_before_demosaic():
    _, capture, _, _, _ = _api()
    sensor = _sensor(full_well_e=1000.)
    action = CaptureAction(sensor.reference_exposure_s)
    a = capture(_scene(.2, 128, 128), action, sensor, seed=12)
    b = capture(_scene(.2, 128, 128), action, sensor, seed=12)
    c = capture(_scene(.2, 128, 128), action, sensor, seed=13)
    assert torch.equal(a.raw_dn, b.raw_dn) and torch.equal(a.rgb, b.rgb)
    assert not torch.equal(a.raw_dn, c.raw_dn)
    raw = (a.raw_dn - sensor.black_level_dn) / (2**sensor.bit_depth - 1 - sensor.black_level_dn)
    assert raw.mean().item() == pytest.approx(.2, abs=.0005)
    assert raw.var().item() == pytest.approx(.0002, rel=.04)
    assert a.rgb[0, 2, 2] == raw[0, 2, 2]


def test_variance_uses_squared_bilinear_weights_and_saturation_tracks_dependencies():
    _, capture, _, _, _ = _api()
    scene = _scene(.2)
    sensor = _sensor(full_well_e=1000.)
    result = capture(scene, CaptureAction(sensor.reference_exposure_s), sensor, noisy=False)
    variance = result.raw_noise_variance[0, 2, 2]
    assert result.noise_variance[0, 2, 2] == variance
    assert result.noise_variance[0, 2, 3] == variance / 2
    assert result.noise_variance[0, 3, 3] == variance / 4
    assert result.noise_variance[1, 2, 2] == variance / 4
    scene.frames[:, 0, 2, 2] = 2.
    clipped = capture(scene, CaptureAction(sensor.reference_exposure_s), sensor, noisy=False)
    assert clipped.raw_saturation_mask.sum() == 1
    assert clipped.saturation_mask[0, 3, 3]
    assert not clipped.saturation_mask[1:].any()


def test_black_subtracted_read_noise_is_not_clamped_before_demosaic():
    _, capture, _, _, _ = _api()
    sensor = SensorProfile(full_well_e=1000., read_noise_e=5., adc_noise_dn=0., bit_depth=16, black_level_dn=5000.)
    result = capture(_scene(0., 128, 128), CaptureAction(sensor.reference_exposure_s), sensor, seed=7)
    assert (result.rgb < 0).any()
    assert abs(result.rgb.mean().item()) < .00015


@pytest.mark.parametrize("kwargs", [
    {"cfa_pattern": "RGBG"}, {"spatial_downsample": 0}, {"spatial_downsample": True},
    {"spatial_downsample": 1.5}, {"readout_s": -1}, {"rolling_shutter_s": float("nan")},
    {"input_stage": "srgb"}, {"psf_kernels": [[0., 0.], [0., 0.]]},
])
def test_profile_rejects_unsupported_physics(kwargs):
    Profile, _, _, _, _ = _api()
    with pytest.raises(ValueError):
        Profile(**kwargs)


@pytest.mark.parametrize("shape,factor", [((7, 8), 1), ((8, 9), 2), ((6, 8), 2)])
def test_prepare_rejects_nondivisible_or_odd_native_bayer_geometry(shape, factor):
    Profile, _, _, _, prepare = _api()
    with pytest.raises(ValueError):
        prepare(_scene(.2, *shape), Profile(spatial_downsample=factor))


def test_serialized_profile_recreates_the_same_physical_capture_and_rejects_digital_gain():
    Profile, capture, _, _, _ = _api()
    profile = Profile(cfa_pattern="GBRG", spatial_downsample=2, readout_s=.002)
    restored = Profile.from_dict(json.loads(json.dumps(profile.to_dict())))
    a = capture(_scene(), CaptureAction(1/120), _sensor(), acquisition=profile, seed=5)
    b = capture(_scene(), CaptureAction(1/120), _sensor(), acquisition=restored, seed=5)
    assert torch.equal(a.raw_dn, b.raw_dn)
    with pytest.raises(ValueError, match="digital_gain"):
        capture(_scene(), CaptureAction(1/120, digital_gain=2.), _sensor())
