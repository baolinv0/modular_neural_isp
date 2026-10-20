"""Exercise deployed AE-to-capture-to-TM paths, independently of training."""
import importlib
import json

import pytest
import torch

from capture_tm.capture_plan import CapturePlan
from capture_tm.learned_policy import TemporalExposurePolicy
from capture_tm.types import CaptureAction, CaptureResult, Scene, SensorProfile


def implementation():
    try:
        return importlib.import_module("capture_tm.joint_algorithm")
    except ModuleNotFoundError as error:
        pytest.fail(f"The joint deployment algorithm is absent: {error}")


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def scene():
    y = torch.linspace(.025, .25, 16).reshape(1, 1, 16, 1)
    radiance = y.expand(1, 3, 16, 16).clone()
    radiance[:, :, 3:6, 11:14] = 2.
    return Scene("deployment", "test", radiance, torch.tensor([0.], dtype=torch.float64))


@pytest.mark.parametrize("scheme,frames", [("apple", 1), ("samsung", 3)])
def test_deployment_captures_only_selected_physical_plan(scheme, frames, monkeypatch):
    """Evaluating every candidate at inference would violate the capture budget."""
    module = implementation()
    algorithm = module.JointCaptureAlgorithm(scheme, SensorProfile())
    actual_capture = module.capture
    observed_calls = []

    def observed_capture(*args, **kwargs):
        result = actual_capture(*args, **kwargs)
        observed_calls.append(result.action)
        return result

    monkeypatch.setattr(module, "capture", observed_capture)
    result = algorithm.run_simulated(scene(), seed=11, noisy=False)
    request = result["request"]
    json.dumps(request, allow_nan=False)
    selected_plan = algorithm.plans[request["selected_index"]]
    assert len(observed_calls) == frames
    assert observed_calls == list(selected_plan.actions)
    assert result["output"].shape == (3, 16, 16)
    assert torch.isfinite(result["output"]).all()
    assert (result["output"] >= 0).all() and (result["output"] <= 1).all()
    assert result["fusion"]["metadata"]["frame_count"] == frames
    assert result["fusion"]["metadata"]["normalization_count"] == (scheme == "samsung")
    assert result["metadata"]["effective_actions"] == request["plan"]["actions"]


@pytest.mark.parametrize("learned", [False, True])
def test_select_honors_legality_for_rule_and_learned_policies(learned):
    module = implementation()
    sensor = SensorProfile()
    policy = TemporalExposurePolicy(width=8) if learned else None
    algorithm = module.JointCaptureAlgorithm("apple", sensor, policy=policy)
    mask = torch.zeros(len(algorithm.plans), dtype=torch.bool)
    mask[2] = True
    request = algorithm.select(torch.full((3, 3, 16, 16), .1), torch.zeros(3, 3), mask)
    assert request["selected_index"] == 2
    assert request["rule_vs_learned"] == ("learned" if learned else "rule")
    assert request["plan"]["actions"] == [algorithm.plans[2].actions[0].to_dict()]


def measurement(rgb, action, center=0.):
    return CaptureResult(rgb, torch.zeros_like(rgb), torch.zeros_like(rgb, dtype=torch.bool),
                         action, {"center_s": center})


def test_single_frame_finish_compensates_actual_gain_even_if_request_differs():
    """Remembering the requested action would render actual effective exposure wrong."""
    module = implementation()
    sensor = SensorProfile()
    plan = CapturePlan((CaptureAction(sensor.reference_exposure_s),), (0.,))
    algorithm = module.JointCaptureAlgorithm("apple", sensor, plans=[plan])
    algorithm.select(torch.full((3, 3, 16, 16), .1), torch.zeros(3, 3))
    x = torch.linspace(.03, .2, 16).reshape(1, 16, 1).expand(3, 16, 16).clone()
    ordinary = algorithm.finish([measurement(x, CaptureAction(sensor.reference_exposure_s))])
    effective = CaptureAction(sensor.reference_exposure_s, analog_gain=2.)
    gained = algorithm.finish([measurement(x * 2., effective)])
    torch.testing.assert_close(gained["output"], ordinary["output"], atol=1e-6, rtol=1e-5)
    assert gained["fusion"]["capture_ev"] == 1.
    assert gained["metadata"]["effective_actions"] == [effective.to_dict()]


def test_hdr_finish_normalizes_effective_exposures_once():
    module = implementation()
    sensor = SensorProfile()
    algorithm = module.JointCaptureAlgorithm("samsung", sensor)
    x = torch.full((3, 16, 16), .12)
    captures = [measurement(x * scale, CaptureAction(sensor.reference_exposure_s * scale), center)
                for scale, center in zip((.25, 1., 2.), (-1 / 30, 0., 1 / 30))]
    result = algorithm.finish(captures)
    torch.testing.assert_close(result["fusion"]["image"], x, atol=1e-6, rtol=1e-5)
    assert result["fusion"]["capture_ev"] == 0.
    assert result["fusion"]["metadata"]["normalization_count"] == 1


@pytest.mark.parametrize("scheme,frames", [("apple", 3), ("samsung", 1)])
def test_finish_rejects_wrong_capture_frame_count(scheme, frames):
    algorithm = implementation().JointCaptureAlgorithm(scheme, SensorProfile())
    capture = measurement(torch.full((3, 16, 16), .1), CaptureAction(1 / 120))
    with pytest.raises(ValueError, match="frame"):
        algorithm.finish([capture] * frames)


def test_select_rejects_empty_legal_set():
    algorithm = implementation().JointCaptureAlgorithm("apple", SensorProfile())
    with pytest.raises(ValueError, match="feasible"):
        algorithm.select(torch.full((3, 3, 16, 16), .1), torch.zeros(3, 3),
                         torch.zeros(len(algorithm.plans), dtype=torch.bool))


@pytest.mark.parametrize("learned", [False, True])
def test_runner_checkpoint_restores_selected_policy_and_trained_tm(tmp_path, learned):
    """Loading only an AE checkpoint or retaining an untrained TM would fail."""
    module = implementation()
    sensor = SensorProfile(reference_exposure_s=1 / 100)
    policy = TemporalExposurePolicy(width=8)
    algorithm = module.JointCaptureAlgorithm("apple", sensor, policy=policy if learned else None)
    with torch.no_grad():
        algorithm.tm.conditioner[-1].bias[0] = .2
    data = {"version": 1, "group": "A11" if learned else "A01", "scheme": "apple",
            "policy_kwargs": {"width": 8, "history_length": 3}, "policy_state": policy.state_dict(),
            "tm_state": algorithm.tm.state_dict(), "learned_ae": learned, "learned_tm": True,
            "sensor": sensor.to_dict(), "plans": [
                {"actions": [action.to_dict() for action in plan.actions], "centers_s": list(plan.centers_s),
                 "integration_time_s": sum(action.exposure_s for action in plan.actions)}
                for plan in algorithm.plans], "tm_weights": None, "render_ev": .5}
    checkpoint = tmp_path / "selected.pt"
    torch.save(data, checkpoint)
    restored = module.JointCaptureAlgorithm.from_checkpoint(checkpoint)
    previews = torch.full((3, 3, 16, 16), .2)
    state = torch.zeros(3, 3)
    assert restored.select(previews, state) == algorithm.select(previews, state)
    assert restored.sensor == sensor
    assert restored.plans == algorithm.plans
    assert restored.checkpoint_render_ev == .5
    capture = measurement(torch.full((3, 16, 16), .1), CaptureAction(1 / 100))
    # A deployed checkpoint must retain the rendering intent used in training,
    # while still accepting an explicit application-level appearance override.
    intended = algorithm.finish([capture], render_ev=.5)["output"]
    unshifted = algorithm.finish([capture], render_ev=0.)["output"]
    torch.testing.assert_close(restored.finish([capture])["output"], intended)
    torch.testing.assert_close(restored.finish([capture], render_ev=0.)["output"], unshifted)
    assert not torch.allclose(intended, unshifted)
    assert not any(parameter.requires_grad for parameter in restored.tm.parameters())
