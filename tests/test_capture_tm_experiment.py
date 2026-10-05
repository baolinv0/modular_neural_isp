import importlib.util
from pathlib import Path

import pytest
import torch


def test_c_workflow_api_is_available():
    assert importlib.util.find_spec('capture_tm.experiment') is not None, 'C workflow is not implemented'


def test_fixed_appearance_penalizes_darkening_and_clipped_highlights():
    from capture_tm.objective import quality_cost
    target = torch.full((1, 3, 12, 12), .5)
    safe = torch.zeros_like(target, dtype=torch.bool)
    exact = quality_cost(target, target, safe)
    darker = quality_cost(target * .5, target, safe)
    concealed_clip = quality_cost(target, target, torch.ones_like(safe))
    assert exact['cost'].item() < darker['cost'].item()
    assert exact['cost'].item() < concealed_clip['cost'].item()


def test_dataset_oracle_train_reload_and_evaluate(tmp_path):
    from capture_tm.data import write_demo_dataset
    from capture_tm.experiment import build_oracle, train_policy, evaluate_policy, infer_request
    manifest = write_demo_dataset(tmp_path / 'data', size=24, scenes=9)
    cache = build_oracle(manifest, tmp_path / 'oracle', backend='analytic', tm_mode='apple',
                         exposures_s=[1/240, 1/120, 1/60], analog_gains=[1., 2.], repeats=2)
    checkpoint = train_policy(cache, tmp_path / 'train', epochs=2, width=8, seed=7)
    report = evaluate_policy(cache, checkpoint, tmp_path / 'eval')
    assert report['test_scenes'] >= 1
    assert set(report['methods']) >= {'baseline', 'physics', 'oracle', 'policy'}
    assert report['methods']['policy']['mean_regret'] >= -1e-6
    req = infer_request(cache, checkpoint, scene_id=None)
    assert req['requested_action']['exposure_s'] > 0
    assert req['execution'] == 'request_only'
    assert req['effective_frame_id'] > req['observation_frame_id']
    assert Path(checkpoint).is_file()


def test_training_does_not_accept_test_only_cache(tmp_path):
    from capture_tm.experiment import train_policy
    payload = {'version': 1, 'records': [], 'actions': [], 'renderer': {}}
    path = tmp_path / 'empty.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='train|schema'):
        train_policy(path, tmp_path / 'train', epochs=1)


@pytest.fixture(scope='module')
def trained_case(tmp_path_factory):
    from capture_tm.data import write_demo_dataset
    from capture_tm.experiment import build_oracle, train_policy
    root = tmp_path_factory.mktemp('causal_case')
    manifest = write_demo_dataset(root/'data', size=16, scenes=9)
    cache = build_oracle(manifest, root/'oracle', backend='analytic',
        exposures_s=[1/120, 1/20], analog_gains=[1.], repeats=1,
        constraints={'control_delay_frames': 2})
    checkpoint = train_policy(cache, root/'train', epochs=1, width=8)
    return manifest, cache, checkpoint


def test_candidate_shutter_cannot_start_before_observation_readout(trained_case):
    _, cache, _ = trained_case
    payload = torch.load(cache, weights_only=True)
    assert payload['records'][0]['feasible_mask'].tolist() == [True, False]
    assert payload['renderer']['renderer_identity']
    assert len(payload['renderer']['implementation_sha256']) == 64


def test_preview_times_must_be_distinct(trained_case, tmp_path):
    from capture_tm.experiment import build_oracle
    manifest, _, _ = trained_case
    with pytest.raises(ValueError, match='chronological'):
        build_oracle(manifest, tmp_path/'duplicate', backend='analytic',
                     preview_centers_s=(0., 0., 0.))


def test_replay_rejects_incompatible_cache(trained_case, tmp_path):
    from capture_tm.experiment import infer_request
    _, cache, checkpoint = trained_case
    payload = torch.load(cache, weights_only=True)
    payload['renderer']['render_intent_ev'] = 1.
    other = tmp_path/'other.pt'
    torch.save(payload, other)
    with pytest.raises(ValueError, match='renderer mismatch'):
        infer_request(other, checkpoint)


def test_request_uses_next_submission_tick_and_remaining_budget(trained_case):
    from capture_tm.experiment import request_from_observations
    _, cache, checkpoint = trained_case
    row = torch.load(cache, weights_only=True)['records'][0]
    req = request_from_observations(row['previews'], row['capture_state'], checkpoint,
                                   observation_frame_id=7)
    assert req['requested_frame_id'] == 8
    assert req['effective_frame_id'] == 10
    with pytest.raises(ValueError, match='feasible'):
        request_from_observations(row['previews'], row['capture_state'], checkpoint,
                                 remaining_capture_s=.001)


def test_physics_motion_estimate_normalizes_observed_exposure_history():
    from capture_tm.experiment import _physics_estimate
    from capture_tm.types import CaptureAction, SensorProfile
    sensor = SensorProfile()
    constant = torch.full((3, 3, 16, 16), .04)
    observed = [CaptureAction(sensor.reference_exposure_s, g) for g in (1., 2., 4.)]
    actions = [CaptureAction(1/240), CaptureAction(1/120), CaptureAction(1/60)]
    changed = constant * torch.tensor([1., 2., 4.])[:, None, None, None]
    neutral_cost = _physics_estimate(constant, [CaptureAction(sensor.reference_exposure_s)]*3, actions, sensor)
    changed_cost = _physics_estimate(changed, observed, actions, sensor)
    assert torch.allclose(neutral_cost, changed_cost, atol=1e-8, rtol=1e-5)
