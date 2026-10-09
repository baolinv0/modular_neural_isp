"""Guard must reject clipping without consulting candidate RAW or targets."""
import pytest
import torch


def test_preview_guard_rejects_high_gain_but_retains_rule_and_hdr_coverage():
    from capture_tm.learned_policy import preview_clipping_guard
    previews = torch.full((1, 3, 3, 16, 16), .2)
    state = torch.zeros(1, 3, 3)
    single = torch.zeros(1, 3, 1, 3)
    single[0, 1, 0, 0] = 4.  # 16x exposure destroys every measurement
    mask, risk = preview_clipping_guard(previews, state, single, torch.tensor([0]), tolerance=.01)
    assert mask.tolist() == [[True, False, True]]
    assert risk.tolist() == [[0., 1., 0.]]
    hdr = single.expand(-1, -1, 3, -1).clone()
    hdr[0, 1, 0, 0] = -2.  # short frame retains highlights
    mask, risk = preview_clipping_guard(previews, state, hdr, torch.tensor([0]), tolerance=.01)
    assert mask.all() and risk.max() == 0


def test_guard_uses_effective_preview_exposure_and_always_keeps_reference():
    from capture_tm.learned_policy import preview_clipping_guard
    previews = torch.full((1, 3, 3, 16, 16), .6)
    state = torch.zeros(1, 3, 3)
    state[:, -1, 0] = 2.
    features = torch.zeros(1, 2, 1, 3)
    features[:, 1, 0, 0] = 4.
    mask, risk = preview_clipping_guard(previews, state, features, torch.tensor([0]), tolerance=0)
    assert mask.tolist() == [[True, False]]
    assert risk.tolist() == [[0., 1.]]
    mask, _ = preview_clipping_guard(previews, state, features, torch.tensor([1]), tolerance=0)
    assert mask.all()  # no fake guarantee when the rule itself clips


def test_disabled_guard_preserves_original_input_mask_and_scores():
    from capture_tm.joint_experiment import _inputs
    from capture_tm.learned_policy import TemporalExposurePolicy
    torch.manual_seed(17)
    record = {'previews': torch.full((3, 3, 16, 16), .2),
              'state': torch.zeros(3, 3), 'rule_index': 0}
    features = torch.zeros(2, 1, 3)
    features[1, 0, 0] = 4
    policy = TemporalExposurePolicy(width=4).eval()
    old = policy(*_inputs(record, features))
    disabled = policy(*_inputs(record, features, clip_risk_tolerance=None))
    torch.testing.assert_close(old, disabled, atol=0, rtol=0)
    guarded = policy(*_inputs(record, features, clip_risk_tolerance=.01))
    assert guarded.argmax(-1).item() == 0 and guarded[0, 1] == -torch.inf


def test_guarded_checkpoint_reloads_same_causal_selection(tmp_path):
    from capture_tm.data import write_demo_dataset
    from capture_tm.joint_experiment import run_factorial, _inputs, _load
    from capture_tm.joint_algorithm import JointCaptureAlgorithm
    import json
    manifest = write_demo_dataset(tmp_path / 'data', size=16, scenes=3, seed=19)
    report = run_factorial(manifest, tmp_path / 'study', scheme='apple', epochs=1,
                           warmup=1, noise_seeds=(0,), threads=1,
                           evaluation_split='val', clip_risk_tolerance=.01)
    cache = json.loads((tmp_path / 'study/apple/cache/cache.json').read_text())
    record = _load(next(r for r in cache['records'] if r['split'] == 'val'))
    for group in ('A10', 'A11'):
        model = JointCaptureAlgorithm.from_checkpoint(tmp_path / f'study/apple/seed_0/{group}/selected.pt')
        assert model.clip_risk_tolerance == .01
        selected = model.select(record['previews'], record['state'])['selected_index']
        assert selected == report['schemes']['apple']['groups'][group]['per_scene'][0]['action_index']
        # No target/candidate capture arguments enter the deployed API.
        from capture_tm.capture_plan import plan_features
        features = plan_features(model.plans, model.sensor)
        assert _inputs(record, features, .01)[-1][0, selected]


def test_guarded_learned_hdr_accepts_external_asymmetric_only_mask():
    from capture_tm.joint_algorithm import JointCaptureAlgorithm
    from capture_tm.learned_policy import TemporalExposurePolicy
    from capture_tm.types import SensorProfile
    algorithm = JointCaptureAlgorithm('samsung', SensorProfile(), policy=TemporalExposurePolicy(width=4))
    algorithm.clip_risk_tolerance = .01
    feasible = torch.zeros(len(algorithm.plans), dtype=torch.bool)
    feasible[14] = True  # no fixed +/-2 bracket in this driver-feasible set
    result = algorithm.select(torch.full((3, 3, 16, 16), .2), torch.zeros(3, 3), feasible)
    assert result['selected_index'] == 14


@pytest.mark.parametrize('value', [-.01, 1.1, float('nan')])
def test_guard_rejects_invalid_tolerance(value):
    from capture_tm.learned_policy import preview_clipping_guard
    with pytest.raises(ValueError, match='tolerance'):
        preview_clipping_guard(torch.ones(1, 3, 3, 2, 2), torch.zeros(1, 3, 3),
                               torch.zeros(1, 1, 1, 3), torch.tensor([0]), tolerance=value)
