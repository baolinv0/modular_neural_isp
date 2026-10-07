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


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
@pytest.mark.parametrize('scheme,prefix', [('apple', 'A'), ('samsung', 'S')])
def test_development_training_never_opens_test_and_reloads_checkpoints(tmp_path, device, scheme, prefix):
    # Catches eager test decoding, host-only inputs, missing GPU updates, and
    # checkpoint/PNG evaluation paths accidentally keeping CUDA tensors.
    from capture_tm.data import write_demo_dataset
    from capture_tm.joint_experiment import run_factorial

    manifest = write_demo_dataset(tmp_path / 'data', size=16, scenes=3, seed=27)
    payload = json.loads(manifest.read_text())
    payload['scenes'][2]['frames_path'] = 'OFFICIAL_TEST_MUST_NOT_BE_OPENED.npy'
    manifest.write_text(json.dumps(payload))
    output = tmp_path / 'development'
    report = run_factorial(manifest, output, scheme=scheme, epochs=1, warmup=1,
                           noise_seeds=(0,), threads=1, candidate_chunk_size=4,
                           device=device, evaluation_split='val')
    assert report['scene_counts'] == {'train': 1, 'val': 1, 'test': 0}
    assert report['config']['evaluation_split'] == 'val'
    cached = json.loads((output / scheme / 'cache' / 'cache.json').read_text())
    assert {row['split'] for row in cached['records']} == {'train', 'val'}
    initial = torch.load(output / scheme / 'seed_0' / 'initial.pt', weights_only=True)
    for suffix in ('00', '10', '01', '11'):
        group_dir = output / scheme / 'seed_0' / (prefix + suffix)
        state = torch.load(group_dir / 'last.pt', map_location='cpu', weights_only=True)
        result = json.loads((group_dir / 'evaluation.json').read_text())
        assert result['evaluation_split'] == 'val'
        assert result['per_scene'][0]['scene_id'] == 'demo_001'
        assert (group_dir / 'val_0000_target_output.png').is_file()
        assert result['training']['runtime']['parameter_device'].startswith(device)
        assert result['training']['runtime']['checkpoint_reload_device'].startswith(device)
        assert result['training']['runtime']['output_device'].startswith(device)
        assert all(value.startswith(device) for value in result['training']['runtime']['input_devices'].values())
        if device == 'cuda':
            assert result['training']['runtime']['peak_allocated_bytes'] > 0
        for module, bit in [('policy_state', suffix[0]), ('tm_state', suffix[1])]:
            changed = any(not torch.equal(v, initial[module][key]) for key, v in state[module].items())
            assert changed == (bit == '1')
        for module, bit in [('ae', suffix[0]), ('tm', suffix[1])]:
            if bit == '1':
                assert result['training']['runtime']['gradient_norm_max'][module] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('scheme', ['apple', 'samsung'])
def test_nested_cache_cuda_render_matches_cpu_fp32(scheme, monkeypatch):
    # Missing recursive transfer or an incorrect device conversion affects
    # either indexing/rendering or the same frozen-coefficient output.
    import copy
    from capture_tm.joint_experiment import _to_device, _render, _inputs, _cost_chunks, backward_expectation
    from capture_tm.learned_tone import ConditionalToneMapper
    from capture_tm.learned_policy import TemporalExposurePolicy

    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)

    torch.manual_seed(17)
    tm = ConditionalToneMapper(scheme, trainable=True).eval()
    policy = TemporalExposurePolicy(width=4).eval()
    gpu_tm, gpu_policy = copy.deepcopy(tm).to('cuda'), copy.deepcopy(policy).to('cuda')
    x = torch.rand(2, 3, 16, 16) * .6
    with torch.no_grad():
        prepared = tm.prepare(x)
    record = {'prepared': prepared, 'num_plans': 2, 'render_ev': 0.,
              'target': torch.full((3, 16, 16), .4), 'subject_mask': None,
              'missing': torch.zeros(1, 2, 3, 16, 16, dtype=torch.bool),
              'previews': torch.rand(3, 3, 16, 16), 'state': torch.zeros(3, 3),
              'nested': [{'value': torch.tensor([3.])}, (torch.tensor([4.]), 'unchanged')]}
    moved = _to_device(record, torch.device('cuda'))
    assert moved['nested'][0]['value'].is_cuda
    assert moved['nested'][1][0].is_cuda
    assert record['nested'][0]['value'].device.type == 'cpu'
    cpu, cpu_metrics = _render(record, tm, 0, [0, 1])
    gpu, gpu_metrics = _render(moved, gpu_tm, 0, [0, 1])
    torch.testing.assert_close(cpu, gpu.cpu(), atol=1e-4, rtol=1e-3)
    for key in cpu_metrics:
        torch.testing.assert_close(cpu_metrics[key], gpu_metrics[key].cpu(), atol=1e-4, rtol=1e-3)
    measurements = {}

    def compare(name, reference, actual):
        actual = actual.detach().cpu()
        reference = reference.detach()
        delta = (actual - reference).abs()
        measurements[name] = {'max_abs': float(delta.max()),
                              'max_rel_denominator_floor_1e-8': float((delta / reference.abs().clamp_min(1e-8)).max())}
        torch.testing.assert_close(reference, actual, atol=1e-4, rtol=1e-3)

    compare('render', cpu, gpu)
    features = torch.randn(2, 1 if scheme == 'apple' else 3, 3)
    scores = policy(*_inputs(record, features))
    gpu_scores = gpu_policy(*_inputs(moved, features.to('cuda')))
    compare('policy_scores', scores, gpu_scores)
    value, _ = backward_expectation(scores, _cost_chunks(record, tm, 0, 1),
                                    _inputs(record, features)[-1], learn_ae=True, learn_tm=True)
    gpu_value, _ = backward_expectation(gpu_scores, _cost_chunks(moved, gpu_tm, 0, 1),
                                      _inputs(moved, features.to('cuda'))[-1], learn_ae=True, learn_tm=True)
    compare('expected_cost', torch.tensor(value), torch.tensor(gpu_value))
    for name, cpu_model, gpu_model in [('ae', policy, gpu_policy), ('tm', tm, gpu_tm)]:
        reference, actual = [], []
        for (_, a), (_, b) in zip(cpu_model.named_parameters(), gpu_model.named_parameters()):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
                reference.append(a.grad.flatten())
                actual.append(b.grad.flatten())
        reference, actual = torch.cat(reference), torch.cat(actual)
        assert float(reference.norm()) > 0 and float(actual.norm()) > 0
        compare(name + '_gradients', reference, actual)
    print('CUDA_PARITY ' + json.dumps({'scheme': scheme, 'measurements': measurements}, allow_nan=False))


def test_nested_transfer_preserves_metadata_and_moves_lists_and_tuples():
    from capture_tm.joint_experiment import _to_device
    record = {'a': [torch.tensor([3.]), {'b': (torch.tensor([4.]), None, 'metadata')}]}
    moved = _to_device(record, 'meta')
    assert moved['a'][0].is_meta
    assert moved['a'][1]['b'][0].is_meta
    assert moved['a'][1]['b'][1:] == (None, 'metadata')
    assert record['a'][0].item() == 3.
    assert record['a'][1]['b'][0].item() == 4.


@pytest.mark.parametrize('field,value', [('source_id', 'demo_000'), ('domain', 'invalid-domain')])
def test_development_keeps_full_manifest_metadata_validation(tmp_path, field, value):
    # Excluding pixels must not hide a declared cross-split source or bad schema.
    from capture_tm.data import write_demo_dataset, load_manifest
    manifest = write_demo_dataset(tmp_path / 'data', size=16, scenes=3)
    payload = json.loads(manifest.read_text())
    payload['scenes'][2]['frames_path'] = 'DO_NOT_OPEN_TEST.npy'
    payload['scenes'][2][field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='crosses|domain'):
        load_manifest(manifest, splits={'train', 'val'})


def test_development_cache_keeps_seed_when_test_is_first(tmp_path):
    # Filtering must not reseed a scene according to its new compacted index.
    from capture_tm.data import write_demo_dataset
    from capture_tm.joint_experiment import run_factorial
    manifest = write_demo_dataset(tmp_path / 'data', size=16, scenes=3, seed=19)
    payload = json.loads(manifest.read_text())
    payload['scenes'] = [payload['scenes'][2], *payload['scenes'][:2]]
    manifest.write_text(json.dumps(payload))
    for mode in ('test', 'val'):
        run_factorial(manifest, tmp_path / mode, scheme='apple', epochs=1, warmup=0,
                      noise_seeds=(0,), threads=1, evaluation_split=mode)
    caches = {}
    for mode in ('test', 'val'):
        info = json.loads((tmp_path / mode / 'apple/cache/cache.json').read_text())
        caches[mode] = {row['scene_id']: torch.load(row['path'], weights_only=True)
                        for row in info['records'] if row['split'] != 'test'}
    for identity, expected in caches['test'].items():
        actual = caches['val'][identity]
        for key in ('previews', 'state', 'target', 'missing', 'radiance_mse'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        for key in expected['prepared']:
            torch.testing.assert_close(actual['prepared'][key], expected['prepared'][key], atol=0, rtol=0)
