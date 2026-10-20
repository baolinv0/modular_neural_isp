"""Behavioral checks of shared targets, real RAW export and capture timing."""
import json
import shutil

import pytest
import torch


def _dataset(tmp_path, schemes='both'):
    from capture_tm.data import write_demo_dataset
    from capture_tm.pipeline import generate_acquisition_dataset
    source = write_demo_dataset(tmp_path / 'source', size=16, scenes=3, seed=7)
    manifest = generate_acquisition_dataset(source, tmp_path / 'archive',
        scheme=schemes, noise_seeds=(0,), seed=11, threads=1)
    return source, manifest


def test_export_preserves_native_bayer_and_fixed_targets_for_both_schemes(tmp_path):
    from capture_tm.pipeline import load_acquisition_manifest
    source, manifest = _dataset(tmp_path)
    payload = load_acquisition_manifest(manifest)
    assert set(payload['schemes']) == {'apple', 'samsung'}
    reference_targets = {}
    for scheme, count in [('apple', 1), ('samsung', 3)]:
        rows = payload['schemes'][scheme]['records']
        assert {r['split'] for r in rows} == {'train', 'val', 'test'}
        for row in rows:
            sample = torch.load(row['path'], weights_only=True)
            assert sample['raw_dn'].shape[2:] == (count, 1, 16, 16)
            assert sample['images'].shape == (1, sample['num_plans'], 3, 16, 16)
            assert sample['raw_saturation_mask'].dtype == torch.bool
            assert sample['preview_metadata']['causal']
            assert sample['state'].shape == (3, 3)
            for plan_meta in sample['capture_metadata'][0]:
                first_start = min(m['shutter_interval_s'][0] for m in plan_meta)
                assert max(sample['preview_metadata']['readout_ends_s']) < first_start
                assert all(m['action']['digital_gain'] == 1. for m in plan_meta)
            if row['scene_id'] in reference_targets:
                torch.testing.assert_close(sample['target'], reference_targets[row['scene_id']])
            reference_targets[row['scene_id']] = sample['target']
    encoded = json.loads(manifest.read_text())
    assert all(not __import__('pathlib').Path(row['path']).is_absolute()
               for info in encoded['schemes'].values() for row in info['records'])


def test_archive_relocation_and_deterministic_seed_do_not_depend_on_output_path(tmp_path):
    from capture_tm.pipeline import generate_acquisition_dataset, load_acquisition_manifest
    source, manifest = _dataset(tmp_path, 'apple')
    repeated = generate_acquisition_dataset(source, tmp_path / 'again', scheme='apple',
                                            noise_seeds=(0,), seed=11, threads=1)
    a = load_acquisition_manifest(manifest)['schemes']['apple']['records']
    b = load_acquisition_manifest(repeated)['schemes']['apple']['records']
    for ra, rb in zip(a, b):
        torch.testing.assert_close(torch.load(ra['path'], weights_only=True)['raw_dn'],
                                  torch.load(rb['path'], weights_only=True)['raw_dn'])
    shutil.copytree(manifest.parent, tmp_path / 'relocated')
    moved = load_acquisition_manifest(tmp_path / 'relocated' / 'manifest.json')
    assert all('relocated' in r['path'] for r in moved['schemes']['apple']['records'])


def test_pipeline_refuses_overwrite_and_invalid_profile_before_output(tmp_path):
    from capture_tm.pipeline import generate_acquisition_dataset
    from capture_tm.acquisition import AcquisitionProfile
    source, manifest = _dataset(tmp_path, 'apple')
    with pytest.raises(FileExistsError):
        generate_acquisition_dataset(source, manifest.parent, scheme='apple')
    with pytest.raises(ValueError, match='readout|slot|window|preview'):
        generate_acquisition_dataset(source, tmp_path / 'bad',
            acquisition=AcquisitionProfile(readout_s=.2), scheme='samsung')
    assert not (tmp_path / 'bad').exists()


def test_dynamic_missing_time_support_never_extrapolates_or_leaves_manifest(tmp_path):
    from capture_tm.data import write_demo_dataset
    from capture_tm.pipeline import generate_acquisition_dataset
    source = write_demo_dataset(tmp_path / 'source', size=16, scenes=3)
    payload = json.loads(source.read_text())
    row = payload['scenes'][1]
    row['frame_times_s'] = [v / 10 for v in row['frame_times_s']]
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='support|shutter'):
        generate_acquisition_dataset(source, tmp_path / 'out', scheme='apple', threads=1)
    assert not (tmp_path / 'out').exists()


def test_validator_rejects_cross_split_dependency_and_changed_target(tmp_path):
    from capture_tm.pipeline import load_acquisition_manifest
    _, manifest = _dataset(tmp_path, 'apple')
    payload = json.loads(manifest.read_text())
    rows = payload['schemes']['apple']['records']
    rows[0]['provenance']['dependency_ids'] = ['same_original']
    rows[1]['provenance']['dependency_ids'] = ['same_original']
    for row in rows[:2]:
        record_path = manifest.parent / row['path']
        record = torch.load(record_path, weights_only=True)
        record['provenance'] = row['provenance']
        torch.save(record, record_path)
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='split'):
        load_acquisition_manifest(manifest)


def test_validator_rejects_candidate_dependent_target(tmp_path):
    from capture_tm.pipeline import load_acquisition_manifest
    _, manifest = _dataset(tmp_path, 'apple')
    payload = json.loads(manifest.read_text())
    record_path = manifest.parent / payload['schemes']['apple']['records'][0]['path']
    record = torch.load(record_path, weights_only=True)
    record['target'] = record['target'] * .8
    torch.save(record, record_path)
    with pytest.raises(ValueError, match='target'):
        load_acquisition_manifest(manifest)


def test_cli_validate_returns_report(tmp_path):
    from capture_tm.pipeline_cli import main
    _, manifest = _dataset(tmp_path, 'apple')
    report_path = tmp_path / 'validation.json'
    report = main(['validate', '--manifest', str(manifest), '--output', str(report_path)])
    assert report['valid']
    assert json.loads(report_path.read_text())['schemes']['apple']['scenes'] == 3


def test_readout_and_rolling_shutter_fit_full_budget():
    from capture_tm.pipeline import build_acquisition_plans, validate_capture_protocol
    from capture_tm.types import SensorProfile
    from capture_tm.acquisition import AcquisitionProfile
    profile = AcquisitionProfile(readout_s=.001, rolling_shutter_s=.003)
    plans = build_acquisition_plans('samsung', SensorProfile(), profile)
    validate_capture_protocol(plans, profile)
    for plan in plans:
        starts = [t - (a.exposure_s + profile.rolling_shutter_s) / 2
                  for t, a in zip(plan.centers_s, plan.actions)]
        ends = [t + (a.exposure_s + profile.rolling_shutter_s) / 2 + profile.readout_s
                for t, a in zip(plan.centers_s, plan.actions)]
        assert starts[0] >= -.05 - 1e-10 and ends[-1] <= .05 + 1e-10
        assert all(end <= start + 1e-10 for end, start in zip(ends[:-1], starts[1:]))


@pytest.mark.parametrize('change', ['dtype', 'ev', 'profile'])
def test_archive_rejects_incompatible_dtype_ev_or_mislabelled_cfa(tmp_path, change):
    from capture_tm.pipeline import load_acquisition_manifest
    _, manifest = _dataset(tmp_path, 'apple')
    payload = json.loads(manifest.read_text())
    row = payload['schemes']['apple']['records'][0]
    record_path = manifest.parent / row['path']
    if change == 'profile':
        payload['acquisition']['cfa_pattern'] = 'BGGR'
        manifest.write_text(json.dumps(payload))
    else:
        record = torch.load(record_path, weights_only=True)
        if change == 'dtype':
            record['previews'] = record['previews'].double()
        else:
            record['capture_ev'][0, 0] = 99.
        torch.save(record, record_path)
    with pytest.raises(ValueError, match='float32|EV|acquisition|profile'):
        load_acquisition_manifest(manifest)


@pytest.mark.parametrize('change', ['preview_clock', 'preview_state', 'rule_index'])
def test_archive_rejects_inconsistent_observation_metadata(tmp_path, change):
    from capture_tm.pipeline import load_acquisition_manifest
    _, manifest = _dataset(tmp_path, 'apple')
    payload = json.loads(manifest.read_text())
    path = manifest.parent / payload['schemes']['apple']['records'][0]['path']
    record = torch.load(path, weights_only=True)
    if change == 'preview_clock':
        record['preview_metadata']['captures'][0]['center_s'] = 1.
    elif change == 'preview_state':
        record['state'].fill_(99.)
    else:
        record['rule_index'] = float(record['rule_index'])
    torch.save(record, path)
    with pytest.raises(ValueError, match='preview|state|rule index'):
        load_acquisition_manifest(manifest)


def test_standalone_validator_checks_raw_to_composed_measurements(tmp_path):
    from capture_tm.pipeline import validate_acquisition_dataset
    _, manifest = _dataset(tmp_path, 'apple')
    payload = json.loads(manifest.read_text())
    path = manifest.parent / payload['schemes']['apple']['records'][0]['path']
    record = torch.load(path, weights_only=True)
    record['raw_dn'].zero_()
    torch.save(record, path)
    with pytest.raises(ValueError, match='RAW|composition'):
        validate_acquisition_dataset(manifest)


def test_archive_rejects_nonfinite_final_capture_clock(tmp_path):
    from capture_tm.pipeline import load_acquisition_manifest
    _, manifest = _dataset(tmp_path, 'apple')
    payload = json.loads(manifest.read_text())
    path = manifest.parent / payload['schemes']['apple']['records'][0]['path']
    record = torch.load(path, weights_only=True)
    record['capture_metadata'][0][0][0]['center_s'] = float('nan')
    torch.save(record, path)
    with pytest.raises(ValueError, match='clock|timing|finite'):
        load_acquisition_manifest(manifest)
