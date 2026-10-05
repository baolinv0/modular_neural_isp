"""Factorial controls and exact split gradients for the joint experiments."""
import json

import pytest
import torch


def test_chunked_expected_gradient_matches_direct_joint_gradient():
    # Catch a detached AE path, omitted probability weights, or TM gradient
    # being counted a second time when the policy backward pass runs.
    from capture_tm.joint_experiment import backward_expectation

    a = torch.tensor(.3, requires_grad=True)
    b = torch.tensor(.2, requires_grad=True)
    scores = torch.stack((a, -a))[None]
    costs = torch.stack(((b - 1).square(), (b + 2).square()))[None]
    direct = (scores.softmax(-1) * costs).sum()
    reference = torch.autograd.grad(direct, (a, b))
    scores = torch.stack((a, -a))[None]
    chunks = iter(((0, (b - 1).square().reshape(1, 1)),
                   (1, (b + 2).square().reshape(1, 1))))
    value, detached = backward_expectation(scores, chunks, torch.ones(1, 2, dtype=torch.bool),
                                          learn_ae=True, learn_tm=True)
    torch.testing.assert_close(a.grad, reference[0])
    torch.testing.assert_close(b.grad, reference[1])
    assert value == pytest.approx(float(direct.detach()))
    assert not detached.requires_grad


def test_scene_bootstrap_keeps_paired_factorial_contrast():
    from capture_tm.joint_experiment import paired_contrasts

    groups = {
        '00': [{'scene_id': 'a', 'cost': 4.}, {'scene_id': 'b', 'cost': 4.}],
        '10': [{'scene_id': 'a', 'cost': 3.}, {'scene_id': 'b', 'cost': 3.}],
        '01': [{'scene_id': 'a', 'cost': 2.}, {'scene_id': 'b', 'cost': 2.}],
        '11': [{'scene_id': 'a', 'cost': .5}, {'scene_id': 'b', 'cost': .5}],
    }
    contrast = paired_contrasts(groups, seed=1, samples=100)
    assert contrast['interaction']['mean'] == pytest.approx(.5)
    assert contrast['interaction']['ci95'] == pytest.approx([.5, .5])
    assert contrast['joint_minus_ae']['mean'] == pytest.approx(-2.5)
    assert contrast['joint_minus_tm']['mean'] == pytest.approx(-1.5)


def test_all_eight_groups_train_only_assigned_modules(tmp_path):
    # Real shipped TM + actual camera simulator: catches swapped 01/10 flags,
    # frozen-weight updates, skipped joint optimization and missing test outputs.
    from capture_tm.data import write_demo_dataset
    from capture_tm.joint_experiment import run_factorial

    manifest = write_demo_dataset(tmp_path / 'data', size=16, scenes=3, seed=11)
    output = tmp_path / 'experiment'
    report = run_factorial(manifest, output, scheme='both', epochs=1, warmup=1,
                           seeds=(0,), noise_seeds=(0,), threads=1,
                           candidate_chunk_size=4)
    assert set(report['schemes']) == {'apple', 'samsung'}
    for scheme, prefix in [('apple', 'A'), ('samsung', 'S')]:
        assert set(report['schemes'][scheme]['groups']) == {prefix + x for x in ('00', '10', '01', '11')}
        initial = torch.load(output / scheme / 'seed_0' / 'initial.pt', weights_only=True)
        for suffix in ('00', '10', '01', '11'):
            group_dir = output / scheme / 'seed_0' / (prefix + suffix)
            state = torch.load(group_dir / 'last.pt', weights_only=True)
            changes = {
                name: any(not torch.equal(v, initial[name][key])
                          for key, v in state[name].items())
                for name in ('policy_state', 'tm_state')
            }
            assert changes['policy_state'] == (suffix[0] == '1')
            assert changes['tm_state'] == (suffix[1] == '1')
            result = json.loads((group_dir / 'evaluation.json').read_text())
            assert result['test_scenes'] == 1
            assert len(result['per_scene']) == 1
            assert result['per_scene'][0]['scene_id'] == 'demo_002'
            assert result['per_scene'][0]['cost'] >= 0
            assert state['optimizer_steps']['ae'] == int(suffix[0])
            assert state['optimizer_steps']['tm'] == int(suffix[1])


def test_invalid_splits_fail_before_creating_results(tmp_path):
    from capture_tm.data import write_demo_dataset
    from capture_tm.joint_experiment import run_factorial

    manifest = write_demo_dataset(tmp_path / 'data', size=16, scenes=3)
    payload = json.loads(manifest.read_text())
    payload['scenes'][1]['split'] = 'train'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='train, val and test'):
        run_factorial(manifest, tmp_path / 'out', scheme='apple', epochs=1, warmup=0)
