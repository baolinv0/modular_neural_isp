"""Physical-unit, causal-observation and fixed-fusion regression tests."""
from dataclasses import replace

import pytest
import torch

from capture_tm.simulation import capture
from capture_tm.types import CaptureAction, Scene, SensorProfile


def _api():
    from capture_tm import capture_plan
    return capture_plan


def _scene(image, *, frames=None, times=None):
    return Scene("fixture", "test", image[None] if frames is None else frames,
                 torch.tensor([0.], dtype=torch.float64) if times is None else times)


def _sensor():
    return SensorProfile(read_noise_e=0., adc_noise_dn=0., bit_depth=20)


def test_bank_has_one_common_reference_time_and_nonoverlapping_legal_shutters():
    api, sensor = _api(), _sensor()
    for scheme, count in (("apple", 1), ("samsung", 3)):
        plans = api.build_plan_bank(scheme, sensor)
        features = api.plan_features(plans, sensor)
        assert features.shape == (len(plans), count, 3)
        assert torch.isfinite(features).all()
        assert all(len(p.actions) == count for p in plans)
        assert all(p.centers_s[count // 2] == 0 for p in plans)
        for p in plans:
            assert all(a.digital_gain == 1 for a in p.actions)
            assert all(a.exposure_s <= 1 / 30 for a in p.actions)
            assert sum(a.exposure_s for a in p.actions) <= .1
            assert all(p.centers_s[i] + p.actions[i].exposure_s / 2 <=
                       p.centers_s[i + 1] - p.actions[i + 1].exposure_s / 2
                       for i in range(count - 1))
        # Removing equal-EV time/gain alternatives would hide the actual AE problem.
        pairs = [(p.actions[count // 2].exposure_s, p.actions[count // 2].analog_gain) for p in plans]
        assert any(abs(t * g - t2 * g2) < 1e-10 and t != t2
                   for t, g in pairs for t2, g2 in pairs)
        if scheme == "apple":
            # The single-frame study needs both a highlight-safe option near
            # eight reference-white units and the full 1/30 s dark-scene slot.
            assert min(t for t, _ in pairs) <= sensor.reference_exposure_s / 8
            assert max(t for t, _ in pairs) == pytest.approx(1 / 30)
            near_eight = _scene(torch.full((3, 8, 8), 7.9))
            assert any(not capture(near_eight, p.actions[0], sensor, noisy=False).saturation_mask.any()
                       for p in plans)


def test_previews_do_not_observe_changed_future_frames():
    api, sensor = _api(), _sensor()
    times = torch.linspace(-.2, .1, 61, dtype=torch.float64)
    frames = torch.full((61, 3, 16, 16), .1)
    changed = frames.clone()
    changed[times >= -.05] = 7.
    before = api.observe_previews(_scene(frames[0], frames=frames, times=times), sensor, seed=4)
    after = api.observe_previews(_scene(frames[0], frames=changed, times=times), sensor, seed=4)
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])
    assert before[0].shape == (3, 3, 16, 16)
    assert before[0].min() >= 0 and before[0].max() <= 1
    first_start = min(p.centers_s[0] - p.actions[0].exposure_s / 2
                      for p in api.build_plan_bank("samsung", sensor))
    assert max(before[2]["shutter_ends_s"]) < first_start


def test_single_exposure_stays_in_capture_units_but_hdr_is_normalized_once():
    api, sensor = _api(), _sensor()
    scene = _scene(torch.full((3, 16, 16), .1))
    ref = sensor.reference_exposure_s
    single = api.CapturePlan((CaptureAction(ref * 2),), (0.,))
    actual = api.render_capture(scene, single, sensor, noisy=False)
    assert actual["image"].mean().item() == pytest.approx(.2, abs=2e-6)
    assert actual["capture_ev"] == pytest.approx(1.)
    hdr = api.CapturePlan(tuple(CaptureAction(ref * f) for f in (.25, 1., 4.)), (-1/30, 0., 1/30))
    fused = api.render_capture(scene, hdr, sensor, noisy=False)
    assert fused["image"].mean().item() == pytest.approx(.1, abs=4e-6)
    assert fused["capture_ev"] == 0.


def test_static_hdr_recovers_clipped_reference_highlight_and_marks_only_all_missing():
    api, sensor = _api(), _sensor()
    image = torch.full((3, 16, 16), .1)
    image[:, :8] = 2.
    image[:, :2] = 20.
    ref = sensor.reference_exposure_s
    plan = api.CapturePlan(tuple(CaptureAction(ref * f) for f in (.125, .5, 2.)), (-1/30, 0., 1/30))
    result = api.render_capture(_scene(image), plan, sensor, noisy=False)
    assert torch.allclose(result["image"][:, 2:8], image[:, 2:8], atol=1e-5)
    assert not result["missing"][:, 2:8].any()
    assert result["missing"][:, :2].all()
    assert result["reliability"].shape == (1, 16, 16)


def test_fusion_uses_observed_noise_estimates_not_simulator_oracle_fields():
    api, sensor = _api(), _sensor()
    scene = _scene(torch.full((3, 16, 16), .2))
    caps = [capture(scene, CaptureAction(sensor.reference_exposure_s * f), sensor, seed=i)
            for i, f in enumerate((.25, 1., 4.))]
    first = api.fixed_fuse(caps, sensor)
    corrupted = [replace(c, noise_variance=torch.full_like(c.noise_variance, 1e10),
                         saturation_mask=torch.ones_like(c.saturation_mask)) for c in caps]
    second = api.fixed_fuse(corrupted, sensor)
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["reliability"], second["reliability"])


def test_motion_rejection_keeps_reference_object_instead_of_ghost_average():
    api, sensor = _api(), _sensor()
    base = torch.full((3, 24, 24), .1)
    reference, other = base.clone(), base.clone()
    reference[:, 8:13, 7:12] = .7
    other[:, 8:13, 15:20] = .7
    images = (other, reference, other)
    caps = [capture(_scene(im), CaptureAction(sensor.reference_exposure_s), sensor, noisy=False)
            for im in images]
    result = api.fixed_fuse(caps, sensor, max_translation=0)
    assert torch.allclose(result["image"], reference, atol=2e-6)
    assert result["metadata"]["motion_rejection_fraction"] > 0


def test_rule_ae_changes_capture_for_dark_and_bright_observations():
    api, sensor = _api(), _sensor()
    dark = api.observe_previews(_scene(torch.full((3, 16, 16), .01)), sensor, seed=1)
    bright = api.observe_previews(_scene(torch.full((3, 16, 16), .8)), sensor, seed=1)
    for scheme in ("apple", "samsung"):
        plans = api.build_plan_bank(scheme, sensor)
        di = api.rule_plan_index(dark[0], dark[1], plans, sensor)
        bi = api.rule_plan_index(bright[0], bright[1], plans, sensor)
        d = plans[di].actions[len(plans[di].actions)//2]
        b = plans[bi].actions[len(plans[bi].actions)//2]
        assert d.exposure_s * d.analog_gain > b.exposure_s * b.analog_gain


def test_actual_measurement_composition_matches_simulator_path_with_one_frontend():
    api = _api()
    sensor = SensorProfile(wb=(2., 1., .5), read_noise_e=0., adc_noise_dn=0., bit_depth=20)
    scene = _scene(torch.full((3, 16, 16), .1))
    for scheme in ("apple", "samsung"):
        plan = api.build_plan_bank(scheme, sensor)[5]
        observations = [capture(scene, a, sensor, center_s=t, noisy=False)
                        for a, t in zip(plan.actions, plan.centers_s)]
        real = api.compose_captures(observations, sensor)
        simulated = api.render_capture(scene, plan, sensor, noisy=False)
        assert torch.equal(real["image"], simulated["image"])
        ratio = real["image"][0].mean() / real["image"][2].mean()
        assert ratio.item() == pytest.approx(4., abs=1e-4)
        assert real["metadata"]["frame_count"] == len(plan.actions)


def test_global_camera_translation_is_aligned_to_middle_frame():
    api, sensor = _api(), _sensor()
    generator = torch.Generator().manual_seed(18)
    reference = .05 + .3 * torch.rand((3, 24, 24), generator=generator)
    shifted = torch.roll(reference, (2, -3), (-2, -1))
    caps = [capture(_scene(im), CaptureAction(sensor.reference_exposure_s), sensor, noisy=False)
            for im in (shifted, reference, shifted)]
    result = api.fixed_fuse(caps, sensor)
    assert result["metadata"]["alignment_shifts_yx"] == [[-2, 3], [0, 0], [-2, 3]]
    assert torch.allclose(result["image"], reference, atol=2e-6)


def test_actual_capture_composition_rejects_nonfinite_measurements():
    api, sensor = _api(), _sensor()
    c = capture(_scene(torch.full((3, 8, 8), .1)), CaptureAction(1/120), sensor)
    c.rgb[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        api.compose_captures([c], sensor)


def test_hdr_plan_bank_optimizes_bracket_shape_while_rule_keeps_fixed_bracket():
    api, sensor = _api(), _sensor()
    plans = api.build_plan_bank("samsung", sensor)
    shapes = {(round(p.actions[0].scale(p.actions[1]), 5),
               round(p.actions[2].scale(p.actions[1]), 5)) for p in plans}
    assert len(shapes) >= 3
    assert (.25, 4.) in shapes
    for level in (.01, .1, .8):
        previews, state, _ = api.observe_previews(_scene(torch.full((3, 16, 16), level)), sensor, seed=2)
        chosen = plans[api.rule_plan_index(previews, state, plans, sensor)]
        assert chosen.actions[0].scale(chosen.actions[1]) == pytest.approx(.25)
        assert chosen.actions[2].scale(chosen.actions[1]) == pytest.approx(4.)
