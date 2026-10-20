import pytest
import torch


def _api():
    from capture_tm.simulation import capture, action_grid
    from capture_tm.types import CaptureAction, SensorProfile, Scene
    return capture, action_grid, CaptureAction, SensorProfile, Scene


def _static(value=0.2, size=16):
    _, _, _, _, Scene = _api()
    return Scene("static", "train", torch.full((1, 3, size, size), value), torch.tensor([0.]))


def test_exposure_changes_photons_but_both_gains_only_scale_signal():
    capture, _, Action, Sensor, _ = _api()
    sensor = Sensor(full_well_e=10000., read_noise_e=0., adc_noise_dn=0., bit_depth=20)
    a = capture(_static(.1), Action(sensor.reference_exposure_s), sensor, noisy=False)
    analog = capture(_static(.1), Action(sensor.reference_exposure_s, 2.), sensor, noisy=False)
    digital = capture(_static(.1), Action(sensor.reference_exposure_s, 1., 2.), sensor, noisy=False)
    longer = capture(_static(.1), Action(sensor.reference_exposure_s * 2.), sensor, noisy=False)
    assert a.metadata["expected_photons_mean_e"] == pytest.approx(1000.)
    assert analog.metadata["expected_photons_mean_e"] == pytest.approx(1000.)
    assert digital.metadata["expected_photons_mean_e"] == pytest.approx(1000.)
    assert longer.metadata["expected_photons_mean_e"] == pytest.approx(2000.)
    assert torch.mean(a.rgb).item() == pytest.approx(.1, abs=2e-6)
    assert torch.mean(analog.rgb).item() == pytest.approx(.2, abs=2e-6)
    assert torch.mean(digital.rgb).item() == pytest.approx(.2, abs=2e-6)
    assert longer.noise_variance.mean().item() < analog.noise_variance.mean().item()


def test_full_well_saturation_happens_before_low_analog_gain():
    capture, _, Action, Sensor, _ = _api()
    sensor = Sensor(read_noise_e=0., adc_noise_dn=0., bit_depth=20)
    result = capture(_static(2.), Action(sensor.reference_exposure_s, .25), sensor, noisy=False)
    assert torch.mean(result.rgb).item() == pytest.approx(.25, abs=2e-6)
    assert result.saturation_mask.all()
    assert result.metadata["full_well_saturation_fraction"] == 1.
    assert result.metadata["adc_saturation_fraction"] == 0.


def test_adc_saturation_distinguished_from_full_well_and_floating_digital_gain():
    capture, _, Action, Sensor, _ = _api()
    sensor = Sensor(read_noise_e=0., adc_noise_dn=0.)
    adc = capture(_static(.75), Action(sensor.reference_exposure_s, 2.), sensor, noisy=False)
    digital = capture(_static(.75), Action(sensor.reference_exposure_s, 1., 2.), sensor, noisy=False)
    assert adc.metadata["full_well_saturation_fraction"] == 0.
    assert adc.metadata["adc_saturation_fraction"] == 1.
    assert digital.metadata["adc_saturation_fraction"] == 0.
    assert digital.metadata["digital_overrange_fraction"] == 1.
    assert digital.rgb.mean().item() == pytest.approx(1.5, abs=.001)
    assert adc.saturation_mask.all() and not digital.saturation_mask.any()


def test_temporal_integration_uses_common_midpoint_and_rejects_unsupported_duration():
    capture, _, Action, Sensor, Scene = _api()
    frames = torch.zeros((3, 3, 4, 4))
    frames[1] = .5
    scene = Scene("pulse", "test", frames, torch.tensor([-1., 0., 1.]))
    sensor = Sensor(reference_exposure_s=1., read_noise_e=0., adc_noise_dn=0., bit_depth=20)
    short = capture(scene, Action(.5), sensor, center_s=0., noisy=False)
    long = capture(scene, Action(2.), sensor, center_s=0., noisy=False)
    assert short.metadata["integrated_radiance_mean"] == pytest.approx(.4375)
    assert long.metadata["integrated_radiance_mean"] == pytest.approx(.25)
    assert long.rgb.mean().item() == pytest.approx(.5, abs=2e-6)
    with pytest.raises(ValueError, match="support"):
        capture(scene, Action(2.1), sensor, noisy=False)


def test_seed_reproducible_and_measured_shot_noise_matches_electron_scale():
    capture, _, Action, Sensor, _ = _api()
    sensor = Sensor(full_well_e=1000., read_noise_e=0., adc_noise_dn=0., bit_depth=20)
    action = Action(sensor.reference_exposure_s)
    first = capture(_static(.2, 128), action, sensor, seed=12)
    again = capture(_static(.2, 128), action, sensor, seed=12)
    other = capture(_static(.2, 128), action, sensor, seed=13)
    assert torch.equal(first.rgb, again.rgb)
    assert not torch.equal(first.rgb, other.rgb)
    assert first.rgb.mean().item() == pytest.approx(.2, abs=.0005)
    assert first.rgb.var().item() == pytest.approx(.0002, rel=.04)


def test_black_subtracted_read_noise_remains_unbiased_and_can_be_negative():
    capture, _, Action, Sensor, _ = _api()
    sensor = Sensor(full_well_e=1000.,read_noise_e=5.,adc_noise_dn=0.,bit_depth=16,black_level_dn=5000.)
    result = capture(_static(0.,128),Action(sensor.reference_exposure_s),sensor,seed=7)
    assert (result.rgb<0).any()
    assert abs(result.rgb.mean().item()) < .0001


@pytest.mark.parametrize("kwargs", [{"exposure_s":0}, {"exposure_s":float("nan")}, {"exposure_s":.1,"analog_gain":-1}])
def test_actions_reject_nonphysical_values(kwargs):
    _, _, Action, _, _ = _api()
    with pytest.raises(ValueError):
        Action(**kwargs)


def test_scene_rejects_duplicate_times_negative_radiance_and_bad_mask():
    _, _, _, _, Scene = _api()
    with pytest.raises(ValueError):
        Scene("bad", "train", torch.zeros((2,3,4,4)), torch.tensor([0.,0.]))
    with pytest.raises(ValueError):
        Scene("bad", "train", -torch.ones((1,3,4,4)), torch.tensor([0.]))
    with pytest.raises(ValueError):
        Scene("bad", "train", torch.ones((1,3,4,4)), torch.tensor([0.]), torch.ones((1,3,3)))


def test_profile_roundtrip_and_grid_preserve_physical_actions():
    _, grid, Action, Sensor, _ = _api()
    profile = Sensor()
    assert Sensor.from_dict(profile.to_dict()).to_dict() == profile.to_dict()
    actions = grid([.01,.02], [1.,2.], digital_gain=1.5)
    assert len(actions) == 4
    assert actions[-1].scale(Action(.01)) == 6.


def test_digital_gain_scales_same_measurement_and_variance_without_reacquiring_photons():
    capture, _, Action, Sensor, _ = _api()
    sensor = Sensor()
    first = capture(_static(.2),Action(sensor.reference_exposure_s),sensor,seed=51)
    double = capture(_static(.2),Action(sensor.reference_exposure_s,1.,2.),sensor,seed=51)
    assert torch.equal(double.rgb,first.rgb*2)
    assert torch.equal(double.noise_variance,first.noise_variance*4)
    assert torch.equal(double.saturation_mask,first.saturation_mask)


@pytest.mark.parametrize("kwargs", [{"full_well_e":0.}, {"read_noise_e":-1.}, {"bit_depth":12.5}, {"black_level_dn":4095}, {"wb":(1,0,1)}, {"calibration_status":"measured_sensor"}])
def test_sensor_profile_rejects_invalid_physics_or_unproven_calibration(kwargs):
    _, _, _, Sensor, _ = _api()
    with pytest.raises(ValueError):
        Sensor(**kwargs)


def test_motion_integration_is_invariant_to_large_absolute_time_origin():
    capture, _, Action, Sensor, Scene = _api()
    frames = torch.full((2,3,4,4),.2)
    sensor = Sensor(read_noise_e=0.,adc_noise_dn=0.,bit_depth=20)
    origin = Scene("origin","test",frames,torch.tensor([0.,1.],dtype=torch.float64))
    shifted = Scene("shifted","test",frames,torch.tensor([1e8,1e8+1],dtype=torch.float64))
    a = capture(origin,Action(.01),sensor,center_s=.5,noisy=False)
    b = capture(shifted,Action(.01),sensor,center_s=1e8+.5,noisy=False)
    assert b.metadata["integrated_radiance_mean"] == pytest.approx(.2,abs=1e-7)
    assert torch.equal(a.rgb,b.rgb)


def test_time_support_tolerance_does_not_expand_with_absolute_time_origin():
    capture, _, Action, Sensor, Scene = _api()
    scene = Scene("shifted","test",torch.ones((2,3,4,4)),torch.tensor([1e6,1e6+1],dtype=torch.float64))
    with pytest.raises(ValueError,match="support"):
        capture(scene,Action(.02),Sensor(),center_s=1e6+.001,noisy=False)


@pytest.mark.parametrize("times", [
    torch.tensor([255,0],dtype=torch.uint8),
    torch.tensor([True,False],dtype=torch.bool),
    torch.tensor([False,True],dtype=torch.bool),
    torch.tensor([0+0j,1+0j],dtype=torch.complex64),
])
def test_scene_rejects_wrapping_or_nonreal_time_types(times):
    _,_,_,_,Scene = _api()
    with pytest.raises(ValueError):
        Scene("invalid_time","train",torch.ones((2,3,4,4)),times)


@pytest.mark.parametrize("provenance", ["calibration notes",["measured"],None])
def test_measured_sensor_provenance_must_be_a_dictionary(provenance):
    _,_,_,Sensor,_ = _api()
    with pytest.raises(ValueError,match="provenance"):
        Sensor(calibration_status="measured_sensor",provenance=provenance)
