"""Real delayed simulation tests catch early action application and mislabeled capture state."""
import importlib

import pytest
import torch


def module(name="control"):
    try:
        return importlib.import_module(f"capture_tm.{name}")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Required capture control implementation is absent: {exc}")


def action(exposure_s=1 / 120, analog_gain=1., digital_gain=1.):
    return module("types").CaptureAction(exposure_s, analog_gain, digital_gain)


def test_constraints_mask_time_gain_and_total_capture_budget():
    constraints = module().ExposureConstraints(min_exposure_s=.001, max_exposure_s=.03, min_analog_gain=1., max_analog_gain=8., max_total_capture_s=.021, readout_s=.002)
    actions = [action(.01, 2), action(.02, 2), action(.0005, 2), action(.01, 9)]
    assert constraints.feasible_mask(actions).tolist() == [True, False, False, False]
    assert constraints.feasible_mask(actions, remaining_capture_s=.005).tolist() == [False, False, False, False]


def test_anti_flicker_mask_uses_integer_half_cycles():
    constraints = module().ExposureConstraints(min_exposure_s=.001, max_exposure_s=.04, max_total_capture_s=.05, anti_flicker_hz=50.)
    assert constraints.feasible_mask([action(.005), action(.01), action(.02), action(.011)]).tolist() == [False, True, True, False]


@pytest.mark.parametrize("settings", [{"min_exposure_s": 0}, {"max_exposure_s": float("inf")}, {"control_delay_frames": -1}, {"control_delay_frames": 1.5}, {"anti_flicker_hz": 0}, {"readout_s": -1}])
def test_constraints_reject_invalid_physical_bounds(settings):
    with pytest.raises(ValueError):
        module().ExposureConstraints(**settings)


def test_requested_action_only_becomes_effective_after_delay():
    initial, requested = action(), action(1 / 60, 2.)
    constraints = module().ExposureConstraints(control_delay_frames=2)
    controller = module().DelayedExposureController(initial, initial, constraints)
    request = controller.request(requested)
    assert (request.requested_frame, request.effective_frame) == (0, 2)
    frames = [controller.advance(), controller.advance(), controller.advance()]
    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [frame.effective_action for frame in frames] == [initial, initial, requested]
    assert [frame.capture_bias_ev for frame in frames] == [0., 0., 2.]
    assert [frame.render_compensation_ev for frame in frames] == [0., 0., -2.]
    assert frames[0].requested_action == requested
    assert frames[1].requested_action is None
    assert frames[2].requested_action is None
    assert controller.ledger == frames


def test_delay_zero_and_next_frame_requests_do_not_have_off_by_one():
    initial = action()
    first, second = action(1 / 60), action(1 / 240)
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints(control_delay_frames=0))
    controller.request(first)
    assert controller.advance().effective_action == first
    request = controller.request(second)
    assert (request.requested_frame, request.effective_frame) == (1, 1)
    assert controller.advance().effective_action == second


def test_invalid_request_does_not_enter_pending_queue_or_ledger():
    initial = action()
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints(max_analog_gain=2.))
    with pytest.raises(ValueError, match="feasible"):
        controller.request(action(analog_gain=4.))
    assert controller.advance().effective_action == initial


def test_duplicate_same_frame_request_is_rejected_explicitly():
    initial = action()
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints())
    controller.request(action(1 / 60))
    with pytest.raises(ValueError, match="already"):
        controller.request(action(1 / 240))


def test_capture_frame_records_effective_metadata_and_real_simulator_output():
    types = module("types")
    initial, requested = action(), action(1 / 60, 2.)
    scene = types.Scene("delay-test", "test", torch.full((1, 3, 4, 4), .05), torch.tensor([0.]))
    sensor = types.SensorProfile(read_noise_e=0., adc_noise_dn=0.)
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints(control_delay_frames=1))
    controller.request(requested)
    frame0 = controller.capture_frame(scene, sensor, noisy=False)
    frame1 = controller.capture_frame(scene, sensor, noisy=False)
    assert frame0.result.action == initial
    assert frame1.result.action == requested
    assert frame1.result.rgb.mean() > frame0.result.rgb.mean() * 3.5
    assert frame0.result.metadata["control"]["effective_frame"] == 0
    assert frame0.result.metadata["control"]["requested_frame"] == 0
    assert frame0.result.metadata["control"]["requested_effective_frame"] == 1
    assert frame0.result.metadata["control"]["effective_action"]["exposure_s"] == initial.exposure_s
    assert frame0.control.capture_bias_ev == 0.
    assert frame1.control.capture_bias_ev == 2.


def test_delayed_requests_keep_their_order_and_causal_source():
    initial, first, second = action(), action(1 / 60), action(1 / 240)
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints(control_delay_frames=2))
    controller.request(first)
    frame0 = controller.advance()
    controller.request(second, source_observation_frame=0)
    frames = [frame0, controller.advance(), controller.advance(), controller.advance()]
    assert [frame.effective_action for frame in frames] == [initial, initial, first, second]
    assert frames[2].effective_request_frame == 0
    assert frames[3].effective_request_frame == 1
    assert frames[3].source_observation_frame == 0


def test_pending_request_and_effective_action_keep_separate_observation_sources():
    initial = action()
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints(control_delay_frames=2))
    controller.advance()
    controller.request(action(1 / 60), source_observation_frame=0)
    pending = controller.advance().to_dict()
    assert pending["requested_source_observation_frame"] == 0
    assert pending["source_observation_frame"] is None
    controller.advance()
    effective = controller.advance().to_dict()
    assert effective["source_observation_frame"] == 0
    assert effective["effective_due_frame"] == 3
    assert effective["requested_source_observation_frame"] is None


def test_request_rejects_future_observation_source():
    initial = action()
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints())
    with pytest.raises(ValueError, match="observ"):
        controller.request(action(1 / 60), source_observation_frame=0)


def test_budget_charges_effective_exposure_and_one_readout_per_frame():
    initial = action(.01)
    constraints = module().ExposureConstraints(max_total_capture_s=.025, readout_s=.002, control_delay_frames=2)
    controller = module().DelayedExposureController(initial, initial, constraints)
    controller.request(action(.02))
    first, second = controller.advance(), controller.advance()
    assert first.total_capture_s == pytest.approx(.012)
    assert second.total_capture_s == pytest.approx(.024)
    with pytest.raises(ValueError, match="budget"):
        controller.advance()
    assert controller.next_frame == 2
    assert len(controller.ledger) == 2


def test_failed_capture_does_not_commit_effective_action_or_cursor():
    types = module("types")
    initial = action()
    controller = module().DelayedExposureController(initial, initial, module().ExposureConstraints())
    controller.request(action(1 / 60))
    scene = types.Scene("too-short", "test", torch.ones(2, 3, 4, 4), torch.tensor([-.001, .001]))
    with pytest.raises(ValueError):
        controller.capture_frame(scene, types.SensorProfile(), noisy=False)
    assert controller.next_frame == 0
    assert controller.effective_action == initial
    assert controller.ledger == []
