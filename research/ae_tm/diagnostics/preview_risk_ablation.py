"""Small validation-only ablation using the existing cache/trainer (not a framework)."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from capture_tm.capture_plan import CapturePlan, plan_features
from capture_tm.types import CaptureAction, SensorProfile
from capture_tm.pipeline import load_acquisition_manifest
from capture_tm.joint_experiment import (build_archive_cache, _seed, _state, _warmstart,
    _train_group, _load, _inputs, _render, _evaluate, _mean_metrics)
from capture_tm.joint_objective import fixed_target
from capture_tm.learned_policy import TemporalExposurePolicy, preview_clipping_guard
from capture_tm.learned_tone import ConditionalToneMapper
from capture_tm.joint_algorithm import JointCaptureAlgorithm


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def interval(rows_a, rows_b, metric):
    # Average noise first (already in rows), then training seeds within scene.
    def values(rows):
        out = {}
        for r in rows:
            out.setdefault(r['scene_id'], []).append(r[metric])
        return {k: np.mean(v) for k, v in out.items()}
    a, b = values(rows_a), values(rows_b)
    assert a.keys() == b.keys()
    d = np.array([a[k] - b[k] for k in sorted(a)])
    ids = np.random.default_rng(0).integers(len(d), size=(2000, len(d)))
    return {'mean': float(d.mean()), 'ci95_descriptive': np.quantile(d[ids].mean(1), [.025, .975]).tolist(),
            'unit': 'validation scene; noise and training seeds averaged; NOT confirmatory'}


def enrich(result, cache_rows, raw_rows, model, features, tolerance):
    cache = {r['scene_id']: r for r in cache_rows}
    raw = {r['scene_id']: r for r in raw_rows}
    for row in result['per_scene']:
        record = _load(cache[row['scene_id']])
        physical = _load(raw[row['scene_id']])
        idx = row['action_index']
        mask, risk = preview_clipping_guard(record['previews'][None], record['state'][None], features[None],
            torch.tensor([record['rule_index']]), tolerance=.01)
        row['predicted_all_frame_clip'] = float(risk[0, idx])
        row['guard_candidates'] = int(mask.sum())
        row['native_all_frame_saturation'] = float(physical['raw_saturation_mask'][:, idx].all(1).float().mean())
        normalized = physical['images'][:, idx] * torch.exp2(-physical['capture_ev'][:, idx, None, None, None])
        delta = normalized - physical['sharp_reference'][None]
        bright = (physical['target'] * physical['target'].new_tensor([.2126, .7152, .0722])[:, None, None]).sum(0) > .8
        row['bright_pixel_count'] = int(bright.sum())
        row['bright_radiance_mse'] = float(delta[..., bright].square().mean()) if bright.any() else 0.
        row['analytic_capture_display_mse'] = float((fixed_target(normalized) - physical['target'][None]).square().mean())
        with torch.no_grad():
            if model.policy is not None:
                probabilities = model.policy(*_inputs(record, features, tolerance)).softmax(-1)[0]
                row['policy_soft_entropy'] = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
                row['policy_max_probability'] = float(probabilities.max())
            predictions = torch.cat([_render(record, model.tm, n, [idx])[0] for n in range(len(record['noise_seeds']))])
            d = predictions - record['target'][None]
            gradient_errors = []
            for axis in (-1, -2):
                adjacent = bright[..., 1:] & bright[..., :-1] if axis == -1 else bright[1:] & bright[:-1]
                if adjacent.any():
                    gradient_errors.append(torch.diff(d, dim=axis)[..., adjacent].abs().mean())
            row['bright_detail_mae'] = float(torch.stack(gradient_errors).mean()) if gradient_errors else 0.
        row['psnr'] = -10 * math.log10(max(row['mse'], 1e-12))
    keys = ['native_all_frame_saturation', 'bright_radiance_mse', 'analytic_capture_display_mse',
            'bright_detail_mae', 'psnr', 'predicted_all_frame_clip', 'guard_candidates',
            'policy_soft_entropy', 'policy_max_probability']
    result['means'].update({k: float(np.mean([r[k] for r in result['per_scene']])) for k in keys
                            if all(k in r for r in result['per_scene'])})
    actions = Counter(r['action_index'] for r in result['per_scene'])
    p = np.array(list(actions.values())) / len(result['per_scene'])
    result['hard_action_modal_share'] = float(p.max())
    result['hard_action_entropy'] = float(-(p * np.log(p)).sum())
    result['action_count'] = len(actions)
    return result


def candidate_audit(cache_rows, raw_rows, scheme):
    tm = ConditionalToneMapper(scheme, trainable=False).eval()
    out = []
    raw = {r['scene_id']: r for r in raw_rows}
    with torch.no_grad():
        for row in cache_rows:
            record, physical = _load(row), _load(raw[row['scene_id']])
            costs = torch.stack([_render(record, tm, n, range(record['num_plans']))[1]['cost']
                                 for n in range(len(record['noise_seeds']))]).mean(0)
            normalized = physical['images'] * torch.exp2(-physical['capture_ev'][..., None, None, None])
            analytic = (fixed_target(normalized.flatten(0, 1)) - physical['target'][None]).square().flatten(1).mean(1)
            analytic = analytic.reshape(len(record['noise_seeds']), record['num_plans']).mean(0)
            missing = record['missing'].float().mean((0, 2, 3, 4))
            raw_sat = physical['raw_saturation_mask'].all(2).float().mean((0, 2, 3, 4))
            out.append({'scene_id': record['scene_id'], 'split': record['split'], 'rule': record['rule_index'],
                'frozen_tm_best': int(costs.argmin()), 'analytic_capture_best': int(analytic.argmin()),
                'frozen_costs': costs.tolist(), 'analytic_capture_mse': analytic.tolist(),
                'observed_missing': missing.tolist(), 'native_all_frame_saturation': raw_sat.tolist()})
    val = [r for r in out if r['split'] == 'val']
    costs = np.array([r['frozen_costs'] for r in val])
    return {'per_scene': out, 'val_frozen_oracle_actions': dict(Counter(str(r['frozen_tm_best']) for r in val)),
            'val_analytic_oracle_actions': dict(Counter(str(r['analytic_capture_best']) for r in val)),
            'val_best_constant_action': int(costs.mean(0).argmin()),
            'val_best_constant_cost': float(costs.mean(0).min()),
            'val_context_oracle_cost': float(costs.min(1).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--prepare-threads', type=int, default=4)
    args = p.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError('choose a new output directory')
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    archive = load_acquisition_manifest(args.manifest, splits={'train', 'val'})
    sensor = SensorProfile.from_dict(archive['sensor'])
    seeds = tuple(int(x) for x in args.seeds.split(','))
    config = {**vars(args), 'seeds': list(seeds), 'clip_tolerance': .01, 'threshold': .98,
              'evaluation_split': 'val', 'data_seed': 2026, 'noise_seeds': archive['noise_seeds'],
              'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
              'ae_lr': .001, 'tm_lr': .0003, 'candidate_chunk_size': 4}
    save(root / 'config.json', config)
    summary = {'config': config, 'schemes': {}, 'evidence': 'development validation; analytic simulated Bayer only'}
    for scheme, prefix in [('apple', 'A'), ('samsung', 'S')]:
        torch.set_num_threads(args.prepare_threads)
        cache = build_archive_cache(archive, scheme, root / scheme / 'cache', candidate_chunk_size=4)
        torch.set_num_threads(1)
        records = cache['records']
        val = [r for r in records if r['split'] == 'val']
        raw = archive['schemes'][scheme]['records']
        plans = [CapturePlan(tuple(CaptureAction(**a) for a in r['actions']), tuple(r['centers_s'])) for r in cache['plans']]
        features = plan_features(plans, sensor)
        audit = candidate_audit(records, raw, scheme)
        save(root / scheme / 'candidate_audit.json', audit)
        collected = {}
        for variant, tolerance, warmup in [('original', None, args.warmup), ('guarded', .01, args.warmup),
                                           ('guarded_no_warmup', .01, 0)]:
            results = {prefix + x: [] for x in ('00', '10', '01', '11')}
            for seed in seeds:
                _seed(seed)
                kwargs = {'width': 16, 'history_length': 3}
                policy = TemporalExposurePolicy(**kwargs)
                tm = ConditionalToneMapper(scheme, trainable=False).eval()
                warm = _warmstart(policy, tm, [r for r in records if r['split'] == 'train'], val, features,
                    epochs=warmup, lr=.001, seed=seed, chunk_size=4, clip_risk_tolerance=tolerance)
                initial = {'policy_kwargs': kwargs, 'policy_state': _state(policy), 'tm_state': _state(tm),
                    'warmstart': warm, 'sensor': archive['sensor'], 'plans': cache['plans'], 'render_ev': 0.,
                    'manifest_kind': 'capture_tm_acquisition_dataset', 'acquisition': archive['acquisition'],
                    'clip_risk_tolerance': tolerance}
                seed_dir = root / scheme / variant / f'seed_{seed}'
                seed_dir.mkdir(parents=True)
                save(seed_dir / 'warmstart.json', warm)
                for suffix in ('00', '10', '01', '11'):
                    group = prefix + suffix
                    directory = seed_dir / group
                    result = _train_group(group, scheme, initial, records, features, directory,
                        epochs=args.epochs, seed=seed, ae_lr=.001, tm_lr=.0003, chunk_size=4,
                        weights=None, evaluation_split='val')
                    model = JointCaptureAlgorithm.from_checkpoint(directory / 'selected.pt')
                    result = enrich(result, val, raw, model, features, tolerance)
                    # Actual deployment/cache selection must agree for every validation scene.
                    for row in result['per_scene']:
                        rec = _load(next(r for r in val if r['scene_id'] == row['scene_id']))
                        assert model.select(rec['previews'], rec['state'])['selected_index'] == row['action_index']
                    save(directory / 'diagnostic_evaluation.json', result)
                    results[group].append(result)
                    print(scheme, variant, seed, group, result['means'], flush=True)
                    if variant == 'original' and suffix[0] == '1':
                        post = _evaluate(val, model.policy, model.tm, features, learned_ae=True,
                                         clip_risk_tolerance=.01, evaluation_split='val')
                        model.clip_risk_tolerance = .01
                        post = enrich(post, val, raw, model, features, .01)
                        post['seed'] = seed
                        save(directory / 'inference_guard_only.json', post)
                        collected.setdefault('inference_guard_only', {}).setdefault(group, []).append(post)
            collected[variant] = results
        groups = {}
        for variant, result in collected.items():
            groups[variant] = {}
            for group, runs in result.items():
                rows = [dict(row, training_seed=run['seed']) for run in runs for row in run['per_scene']]
                means = {k: float(np.mean([run['means'][k] for run in runs])) for k in runs[0]['means']}
                groups[variant][group] = {'means': means, 'per_scene': rows,
                    'seed_actions': [{k: run[k] for k in ('seed', 'actions', 'action_count', 'hard_action_modal_share', 'hard_action_entropy')} for run in runs]}
        for variant in ('guarded', 'guarded_no_warmup'):
            for suffix in ('00', '01'):
                assert groups[variant][prefix + suffix]['means'] == groups['original'][prefix + suffix]['means']
        contrasts = {variant: {metric: interval(groups[variant][prefix+'11']['per_scene'], groups['original'][prefix+'01']['per_scene'], metric)
            for metric in ('mse', 'cost', 'detail_mae', 'highlight_mse', 'missing_fraction', 'radiance_mse', 'native_all_frame_saturation')}
            for variant in groups}
        summary['schemes'][scheme] = {'groups': groups, 'joint_minus_tm': contrasts,
                                      'candidate_audit_summary': {k: v for k, v in audit.items() if k != 'per_scene'}}
        save(root / 'summary.json', summary)
    summary['seconds'] = time.perf_counter() - started
    save(root / 'summary.json', summary)
    print(json.dumps({'seconds': summary['seconds'], 'output': str(root)}, indent=2))


if __name__ == '__main__':
    main()
