"""Paired 2x2 AE/TM experiments with exact discrete-expectation training.

Camera samples and the frozen ISP's coefficients are cached independently of
the learned parameters. Each physical candidate is rendered separately. The
policy and TM backward passes are split only to bound activation memory; their
sum is exactly the gradient of the same expected final-image loss.
"""
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import load_manifest


def _json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def backward_expectation(scores, chunks, mask, *, learn_ae, learn_tm, temperature=1.):
    """Accumulate exact joint gradients, retaining at most one TM chunk graph.

    ``chunks`` yields (first_candidate, costs[B,C]) in candidate order. AE and
    TM parameters must be disjoint: the scores depend on AE, costs on TM. The
    final AE backward uses detached costs so it cannot double-count TM grads.
    """
    from .learned_policy import expected_quality_loss

    if scores.shape != mask.shape or temperature <= 0 or not mask.any(-1).all():
        raise ValueError('scores/mask require a legal candidate per observation')
    probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf) / temperature, -1)
    detached = torch.zeros_like(scores)
    cursor = 0
    for start, costs in chunks:
        if start != cursor or costs.ndim != 2 or costs.shape[0] != scores.shape[0]:
            raise ValueError('cost chunks must cover the candidate axis in order')
        end = start + costs.shape[1]
        if end > scores.shape[1] or not torch.isfinite(costs).all():
            raise ValueError('candidate costs must be finite and match scores')
        detached[:, start:end] = costs.detach()
        if learn_tm:
            weighted = (probabilities[:, start:end].detach() * costs).sum(-1).mean()
            weighted.backward()
        cursor = end
    if cursor != scores.shape[1]:
        raise ValueError('cost chunks must cover all candidates')
    expected = expected_quality_loss(scores, detached, mask, temperature=temperature)
    if learn_ae:
        expected.backward()
    return float(expected.detach()), detached


def paired_contrasts(groups, *, seed=0, samples=2000):
    """Average repeats within scene, then paired-bootstrap independent scenes."""
    by_group = {}
    for group in ('00', '10', '01', '11'):
        rows = {}
        for row in groups[group]:
            rows.setdefault(row['scene_id'], []).append(float(row['cost']))
        by_group[group] = {sid: float(np.mean(values)) for sid, values in rows.items()}
    identities = set(by_group['00'])
    if not identities or any(set(rows) != identities for rows in by_group.values()):
        raise ValueError('paired contrasts require identical nonempty scene sets')
    ids = sorted(identities)
    v = {key: np.array([rows[sid] for sid in ids]) for key, rows in by_group.items()}
    values = {'interaction': v['10'] + v['01'] - v['00'] - v['11'],
              'joint_minus_ae': v['11'] - v['10'],
              'joint_minus_tm': v['11'] - v['01'],
              'joint_minus_baseline': v['11'] - v['00']}
    indices = np.random.default_rng(seed).integers(0, len(ids), (samples, len(ids)))
    result = {name: {'mean': float(value.mean()),
                     'ci95': np.quantile(value[indices].mean(1), [.025, .975]).tolist(),
                     'per_scene': {sid: float(x) for sid, x in zip(ids, value)}}
              for name, value in values.items()}
    result['independent_scenes'] = len(ids)
    result['bootstrap_unit'] = 'scene; training seeds and noise repeats averaged within scene'
    result['interpretation'] = ('Positive interaction supports super-additivity only in the declared cost; '
                                'negative joint-minus-* favors joint. CI is conditional on these training runs.')
    return result


def _sharp_reference(scene, sensor):
    if len(scene.frames) == 1:
        sharp = scene.frames[0]
    else:
        times = scene.frame_times_s
        if float(times[0]) > 0 or float(times[-1]) < 0:
            raise ValueError('scene must cover the fixed reference time t=0')
        right = int(torch.searchsorted(times, times.new_tensor(0.)))
        if right == 0:
            sharp = scene.frames[0]
        else:
            alpha = -float(times[right - 1]) / float(times[right] - times[right - 1])
            sharp = scene.frames[right - 1] * (1 - alpha) + scene.frames[right] * alpha
    wb = sharp.new_tensor(sensor.wb).reshape(3, 1, 1)
    return torch.einsum('ij,jhw->ihw', sharp.new_tensor(sensor.ccm), sharp * wb).clamp_min(0)


def _plan_dict(plan):
    return {'actions': [a.to_dict() for a in plan.actions], 'centers_s': list(plan.centers_s),
            'integration_time_s': float(sum(a.exposure_s for a in plan.actions))}


def build_capture_cache(scenes, sensor, plans, scheme, output, *, noise_seeds, weights=None,
                        candidate_chunk_size=4, render_ev=0., scene_indices=None):
    """Prepare one scene at a time; all four groups consume the same cache."""
    from .capture_plan import observe_previews, render_capture, rule_plan_index
    from .joint_objective import fixed_target
    from .learned_tone import ConditionalToneMapper

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tm = ConditionalToneMapper(scheme, weights=weights, trainable=False).eval()
    metadata = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, scene in enumerate(scenes):
            seed_index = index if scene_indices is None else scene_indices[scene.scene_id]
            previews, state, preview_meta = observe_previews(scene, sensor, seed=100000 + seed_index)
            rule_index = int(rule_plan_index(previews, state, plans, sensor))
            sharp = _sharp_reference(scene, sensor)
            target = fixed_target(sharp[None], render_ev=render_ev)[0]
            prepared_parts, missing_parts, fusion_errors = [], [], []
            effective_noise_seeds = [int(n) + {'train': 0, 'val': 5000, 'test': 10000}[scene.split]
                                     for n in noise_seeds]
            for repeat in effective_noise_seeds:
                for start in range(0, len(plans), candidate_chunk_size):
                    subset = plans[start:start + candidate_chunk_size]
                    captures = [render_capture(scene, plan, sensor,
                                seed=int(repeat) + 100003 * (seed_index + 1), noisy=True) for plan in subset]
                    images = torch.stack([cap['image'] for cap in captures])
                    reliability = torch.stack([cap['reliability'] for cap in captures])
                    ev = images.new_tensor([cap['capture_ev'] for cap in captures])
                    prepared = tm.prepare(images, capture_ev=ev, reliability=reliability)
                    prepared_parts.append({k: v.detach().cpu() for k, v in prepared.items()})
                    missing_parts.extend(cap['missing'].detach().cpu() for cap in captures)
                    normalized = images * torch.exp2(-ev[:, None, None, None])
                    fusion_errors.extend((normalized - sharp[None]).square().flatten(1).mean(1).tolist())
            prepared = {key: torch.cat([part[key] for part in prepared_parts])
                        for key in prepared_parts[0]}
            record = {'scene_id': scene.scene_id, 'source_id': scene.source_id, 'split': scene.split,
                      'source_kind': scene.source_kind, 'provenance': scene.provenance,
                      'previews': previews.cpu(), 'state': state.cpu(), 'preview_metadata': preview_meta,
                      'rule_index': rule_index, 'target': target.cpu(),
                      'subject_mask': None if scene.subject_mask is None else scene.subject_mask.cpu(),
                      'prepared': prepared,
                      'missing': torch.stack(missing_parts).reshape(len(noise_seeds), len(plans), *missing_parts[0].shape),
                      'radiance_mse': torch.tensor(fusion_errors).reshape(len(noise_seeds), len(plans)),
                      'num_plans': len(plans), 'noise_seeds': effective_noise_seeds, 'render_ev': render_ev}
            path = output / f'scene_{index:05d}.pt'
            torch.save(record, path)
            metadata.append({'scene_id': scene.scene_id, 'split': scene.split, 'path': str(path),
                             'rule_index': rule_index, 'source_kind': scene.source_kind})
            print(f'{scheme} cache {index + 1}/{len(scenes)}: {scene.scene_id} '
                  f'({scene.split}), {len(plans)} plans x {len(noise_seeds)} noise repeats', flush=True)
    info = {'records': metadata, 'seconds': time.perf_counter() - started,
            'prepare_threads': torch.get_num_threads(),
            'target': 'fixed_target(fixed WB/CCM sharp latent at t=0); independent of action and learned TM',
            'prepared_cache': 'detached fixed ISP coefficient predictions; learned adapters excluded',
            'noise_seeds': list(noise_seeds), 'plans': [_plan_dict(p) for p in plans],
            'noise_split_offsets': {'train': 0, 'val': 5000, 'test': 10000},
            'shared_across_training_seeds_and_groups': True}
    _json(output / 'cache.json', info)
    return info


def build_archive_cache(archive, scheme, output, *, weights=None,
                        candidate_chunk_size=4, render_ev=0.):
    """Prepare frozen ISP coefficients directly from archived measurements.

    The archive's images have already passed through fixed capture composition
    and WB/CCM. Reusing them preserves its exact noise, fusion, and fixed target
    across every training seed and factorial group. Native RAW remains in the
    acquisition archive rather than being duplicated in this training cache.
    ``archive`` is validated loader output with absolute record paths.
    """
    from .learned_tone import ConditionalToneMapper

    if candidate_chunk_size < 1:
        raise ValueError('candidate_chunk_size must be positive')
    source = archive['schemes'][scheme]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tm = ConditionalToneMapper(scheme, weights=weights, trainable=False).eval()
    metadata = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, row in enumerate(source['records']):
            record = _load(row)
            if float(record['render_ev']) != float(render_ev):
                raise ValueError('archive record target intent does not match requested render_ev')
            images = record['images'].flatten(0, 1)
            reliability = record['reliability'].flatten(0, 1)
            capture_ev = record['capture_ev'].flatten()
            parts = [tm.prepare(images[start:start + candidate_chunk_size],
                                capture_ev=capture_ev[start:start + candidate_chunk_size],
                                reliability=reliability[start:start + candidate_chunk_size])
                     for start in range(0, len(images), candidate_chunk_size)]
            prepared = {key: torch.cat([part[key].detach().cpu() for part in parts])
                        for key in parts[0]}
            saved = {key: record[key] for key in (
                'scene_id', 'source_id', 'split', 'source_kind', 'provenance',
                'previews', 'state', 'preview_metadata', 'rule_index', 'target',
                'subject_mask', 'missing', 'radiance_mse', 'num_plans',
                'noise_seeds', 'render_ev')}
            saved['prepared'] = prepared
            path = output / f'scene_{index:05d}.pt'
            torch.save(saved, path)
            metadata.append({**row, 'path': str(path), 'rule_index': record['rule_index']})
            print(f'{scheme} archive cache {index + 1}/{len(source["records"])}: '
                  f'{record["scene_id"]} ({record["split"]}), {record["num_plans"]} plans '
                  f'x {len(record["noise_seeds"])} archived noise repeats', flush=True)
    info = {'records': metadata, 'seconds': time.perf_counter() - started,
            'prepare_threads': torch.get_num_threads(), 'plans': source['plans'],
            'target': 'unchanged fixed acquisition-archive target',
            'prepared_cache': 'detached fixed ISP coefficient predictions from archived composed images',
            'capture_source': 'acquisition archive; no capture or fusion is repeated',
            'shared_across_training_seeds_and_groups': True}
    _json(output / 'cache.json', info)
    return info


def _to_device(value, device):
    """Move one cache record recursively, preserving its non-tensor metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _load(record, device='cpu'):
    return _to_device(torch.load(record['path'], map_location='cpu', weights_only=True), device)


def _inputs(record, features, clip_risk_tolerance=None):
    count = features.shape[0]
    mask = torch.ones(1, count, dtype=torch.bool, device=features.device)
    if clip_risk_tolerance is not None:
        from .learned_policy import preview_clipping_guard
        mask, _ = preview_clipping_guard(record['previews'][None], record['state'][None],
            features[None], torch.tensor([record['rule_index']], device=features.device),
            tolerance=clip_risk_tolerance)
    return (record['previews'][None], record['state'][None], features[None],
            mask)


def _render(record, tm, noise_index, candidate_indices):
    from .joint_objective import image_metrics

    ids = torch.as_tensor(candidate_indices, dtype=torch.long, device=record['target'].device)
    flat = ids + noise_index * record['num_plans']
    prepared = {key: value[flat] for key, value in record['prepared'].items()}
    output = tm.render_prepared(prepared, render_ev=record['render_ev'])
    target = record['target'][None].expand(len(ids), -1, -1, -1)
    subject = record['subject_mask']
    if subject is not None:
        subject = subject[None].expand(len(ids), -1, -1, -1)
    metrics = image_metrics(output, target, record['missing'][noise_index, ids], subject_mask=subject)
    return output, metrics


def _cost_chunks(record, tm, repeat, chunk_size):
    for start in range(0, record['num_plans'], chunk_size):
        ids = range(start, min(start + chunk_size, record['num_plans']))
        _, metrics = _render(record, tm, repeat, ids)
        yield start, metrics['cost'][None]


def _teacher_costs(metadata, tm, chunk_size):
    costs = {}
    with torch.no_grad():
        for row in metadata:
            record = _load(row, next(tm.parameters()).device)
            repeats = [torch.cat([value[0] for _, value in _cost_chunks(record, tm, n, chunk_size)])
                       for n in range(len(record['noise_seeds']))]
            costs[record['scene_id']] = torch.stack(repeats).mean(0)[None]
    return costs


def _warmstart(policy, tm, train, val, features, *, epochs, lr, seed, chunk_size, clip_risk_tolerance=None):
    from .learned_policy import warmstart_loss

    if epochs == 0:
        return {'history': [], 'selected_epoch': 0, 'optimizer_steps': 0, 'seconds': 0.}
    started = time.perf_counter()
    teacher = _teacher_costs(train + val, tm, chunk_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history, best, selected, steps = [], float('inf'), 0, 0
    best_state = _state(policy)
    for epoch in range(epochs):
        policy.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        losses = []
        for index in order:
            record = _load(train[int(index)], features.device)
            args = _inputs(record, features, clip_risk_tolerance)
            scores = policy(*args)
            loss = warmstart_loss(scores, teacher[record['scene_id']], args[-1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        with torch.no_grad():
            policy.eval()
            val_cost = float(np.mean([float(teacher[row['scene_id']][0,
                int(policy(*_inputs(_load(row, features.device), features, clip_risk_tolerance)).argmax(-1))]) for row in val]))
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'val_cost': val_cost})
        if val_cost < best:
            best, selected, best_state = val_cost, epoch + 1, _state(policy)
    policy.load_state_dict(best_state)
    return {'history': history, 'selected_epoch': selected, 'optimizer_steps': steps,
            'seconds': time.perf_counter() - started,
            'train_oracle_actions': dict(Counter(str(int(teacher[row['scene_id']].masked_fill(
                ~_inputs(_load(row, features.device), features, clip_risk_tolerance)[-1], torch.inf).argmin())) for row in train)),
            'val_oracle_actions': dict(Counter(str(int(teacher[row['scene_id']].masked_fill(
                ~_inputs(_load(row, features.device), features, clip_risk_tolerance)[-1], torch.inf).argmin())) for row in val)),
            'teacher': 'frozen initial TM costs; train scenes optimize, validation scenes select'}


def _mean_metrics(rows):
    keys = ('cost', 'mse', 'subject_luma_mae', 'detail_mae', 'missing_fraction',
            'radiance_mse', 'dark_mse', 'highlight_mse', 'rendered_repeat_variance')
    return {key: float(np.mean([row[key] for row in rows])) for key in keys if all(key in row for row in rows)}


def _evaluate(metadata, policy, tm, features, *, learned_ae, output=None, evaluation_split='test',
              clip_risk_tolerance=None):
    policy.eval()
    tm.eval()
    rows = []
    runtime = {}
    with torch.no_grad():
        for scene_index, row in enumerate(metadata):
            record = _load(row, features.device)
            args = _inputs(record, features, clip_risk_tolerance)
            idx = int(policy(*args).argmax(-1)) if learned_ae else record['rule_index']
            repeats, predictions = [], []
            for n, noise_seed in enumerate(record['noise_seeds']):
                prediction, metrics = _render(record, tm, n, [idx])
                runtime = {'output_device': str(prediction.device), 'input_devices': {
                    key: str(record[key].device) for key in ('previews', 'state', 'target')}}
                runtime['input_devices']['plan_features'] = str(features.device)
                metrics = {key: float(value[0]) for key, value in metrics.items()}
                metrics['radiance_mse'] = float(record['radiance_mse'][n, idx])
                repeats.append({'noise_seed': noise_seed, **metrics})
                predictions.append(prediction[0])
            result = {'scene_id': record['scene_id'], 'source_id': record['source_id'],
                      'source_kind': record['source_kind'], 'action_index': idx,
                      'condition': record['provenance'].get('condition', 'unspecified'),
                      'rule_index': record['rule_index'], **_mean_metrics(repeats), 'per_noise': repeats}
            if len(predictions) > 1:
                result['rendered_repeat_variance'] = float(torch.stack(predictions).var(0, unbiased=True).mean())
            rows.append(result)
            if output is not None:
                from PIL import Image
                strip = torch.cat((record['target'], predictions[0]), -1)
                array = (strip.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(array).save(Path(output) / f'{evaluation_split}_{scene_index:04d}_target_output.png')
    return {'means': _mean_metrics(rows), 'per_scene': rows,
            'actions': dict(Counter(str(row['action_index']) for row in rows)), 'runtime': runtime}


def _train_group(group, scheme, initial, metadata, features, output, *, epochs, seed, ae_lr, tm_lr,
                 chunk_size, weights, evaluation_split='test'):
    from .learned_policy import TemporalExposurePolicy
    from .learned_tone import ConditionalToneMapper

    learned_ae, learned_tm = group[-2] == '1', group[-1] == '1'
    clip_risk_tolerance = initial.get('clip_risk_tolerance')
    _seed(seed)
    policy = TemporalExposurePolicy(**initial['policy_kwargs']).to(features.device)
    policy.load_state_dict(initial['policy_state'])
    tm = ConditionalToneMapper(scheme, weights=weights, trainable=learned_tm).to(features.device)
    tm.load_state_dict(initial['tm_state'])
    policy.requires_grad_(learned_ae)
    parameters = []
    if learned_ae:
        parameters.append({'params': list(policy.parameters()), 'lr': ae_lr})
    if learned_tm:
        parameters.append({'params': [p for p in tm.parameters() if p.requires_grad], 'lr': tm_lr})
    optimizer = torch.optim.Adam(parameters) if parameters else None
    splits = {split: [row for row in metadata if row['split'] == split] for split in ('train', 'val', 'test')}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    steps = {'ae': 0, 'tm': 0}
    history = []
    runtime = {'parameter_device': str(next(tm.parameters()).device),
               'gradient_norm_max': {'ae': 0., 'tm': 0.}, 'input_devices': {}, 'output_device': None}
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
        torch.cuda.reset_peak_memory_stats(features.device)
    started = time.perf_counter()
    baseline_val = _evaluate(splits['val'], policy, tm, features, learned_ae=learned_ae,
                             clip_risk_tolerance=clip_risk_tolerance)['means']['cost']
    best, selected_epoch = baseline_val, 0

    def checkpoint(epoch):
        return {'version': 1, 'group': group, 'scheme': scheme, 'seed': seed, 'epoch': epoch,
                'policy_kwargs': initial['policy_kwargs'], 'policy_state': _state(policy), 'tm_state': _state(tm),
                'optimizer_steps': dict(steps), 'learned_ae': learned_ae, 'learned_tm': learned_tm,
                'selected_on': 'validation cost; test scenes excluded',
                'plan_features': features.detach().cpu().clone(), 'sensor': initial['sensor'], 'plans': initial['plans'],
                'tm_weights': str(weights) if weights else None, 'render_ev': initial['render_ev'],
                'acquisition': initial.get('acquisition'), 'manifest_kind': initial.get('manifest_kind'),
                'clip_risk_tolerance': clip_risk_tolerance}

    torch.save(checkpoint(0), output / 'selected.pt')
    for epoch in range(epochs if optimizer is not None else 0):
        policy.train(learned_ae)
        tm.train(learned_tm)
        order = np.random.default_rng(seed + 1009 + epoch).permutation(len(splits['train']))
        losses = []
        for index in order:
            record = _load(splits['train'][int(index)], features.device)
            runtime['input_devices'] = {key: str(record[key].device) for key in ('previews', 'state', 'target')}
            runtime['input_devices']['plan_features'] = str(features.device)
            for repeat in range(len(record['noise_seeds'])):
                optimizer.zero_grad(set_to_none=True)
                if learned_ae:
                    args = _inputs(record, features, clip_risk_tolerance)
                    scores = policy(*args)
                    # Lazy rendering retains at most one chunk graph. Frozen
                    # weights and cached inputs already have no TM gradients.
                    chunks = _cost_chunks(record, tm, repeat, chunk_size)
                    value, _ = backward_expectation(scores, chunks, args[-1],
                                                    learn_ae=True, learn_tm=learned_tm)
                else:
                    prediction, metrics = _render(record, tm, repeat, [record['rule_index']])
                    runtime['output_device'] = str(prediction.device)
                    loss = metrics['cost'].mean()
                    loss.backward()
                    value = float(loss.detach())
                for name, module in (('ae', policy), ('tm', tm)):
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    if grads:
                        norm = float(torch.stack([g.detach().square().sum() for g in grads]).sum().sqrt())
                        if not np.isfinite(norm):
                            raise RuntimeError(f'nonfinite {name} gradient')
                        runtime['gradient_norm_max'][name] = max(runtime['gradient_norm_max'][name], norm)
                torch.nn.utils.clip_grad_norm_([p for item in parameters for p in item['params']], 5.)
                optimizer.step()
                steps['ae'] += int(learned_ae)
                steps['tm'] += int(learned_tm)
                losses.append(value)
        validation = _evaluate(splits['val'], policy, tm, features, learned_ae=learned_ae,
                               clip_risk_tolerance=clip_risk_tolerance)['means']['cost']
        history.append({'epoch': epoch + 1, 'train_expected_cost': float(np.mean(losses)), 'val_cost': validation})
        if validation < best:
            best, selected_epoch = validation, epoch + 1
            torch.save(checkpoint(epoch + 1), output / 'selected.pt')
    torch.save(checkpoint(epochs if optimizer is not None else 0), output / 'last.pt')
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
        runtime['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(features.device)
    training = {'group': group, 'history': history, 'initial_val_cost': baseline_val,
                'selected_epoch': selected_epoch, 'selected_val_cost': best, 'optimizer_steps': steps,
                'ae_warmup_optimizer_steps': initial['warmstart']['optimizer_steps'] if learned_ae else 0,
                'shared_ae_warmup_seconds': initial['warmstart']['seconds'] if learned_ae else 0.,
                'seconds': time.perf_counter() - started,
                'train_scenes': len(splits['train']), 'val_scenes': len(splits['val']),
                'tm_trainable_parameters': sum(p.numel() for p in tm.parameters() if p.requires_grad),
                'ae_trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
                'update_budget': 'one optimizer step per train scene per noise repeat per epoch',
                'train_candidate_renders_per_step': int(features.shape[0]) if learned_ae else int(learned_tm),
                'joint_gradient': 'exact separate-candidate expectation; split disjoint-parameter backward'}
    selected = torch.load(output / 'selected.pt', map_location='cpu', weights_only=True)
    policy.load_state_dict(selected['policy_state'])
    tm.load_state_dict(selected['tm_state'])
    runtime['checkpoint_reload_device'] = str(next(tm.parameters()).device)
    evaluation = _evaluate(splits[evaluation_split], policy, tm, features, learned_ae=learned_ae,
                           output=output, evaluation_split=evaluation_split, clip_risk_tolerance=clip_risk_tolerance)
    runtime.update(evaluation['runtime'])
    runtime['policy_parameter_device'] = str(next(policy.parameters()).device)
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
        runtime['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(features.device)
    training['runtime'] = runtime
    _json(output / 'training.json', training)
    evaluation.update({'group': group, 'seed': seed, 'test_scenes': len(splits['test']),
                       'evaluation_split': evaluation_split,
                       'selected_epoch': selected_epoch, 'training': training,
                       'evidence': ('development validation; reused for checkpoint selection' if evaluation_split == 'val'
                                    else 'held-out simulator results; not real-camera quality validation')})
    _json(output / 'evaluation.json', evaluation)
    return evaluation


def run_factorial(manifest, output, *, scheme='both', epochs=10, warmup=5, seeds=(0,),
                  noise_seeds=(0, 1), threads=2, prepare_threads=None, candidate_chunk_size=4, ae_lr=1e-3,
                  tm_lr=3e-4, weights=None, render_ev=0., device='cpu', evaluation_split='test',
                  clip_risk_tolerance=None):
    """Execute four controlled groups from v1 scenes or a v2 RAW archive.

    Archive noise repeats, target intent, and physical plans are fixed inputs;
    the requested noise seeds and render EV must match their recorded protocol.
    """
    if scheme not in ('apple', 'samsung', 'both'):
        raise ValueError('scheme must be apple, samsung or both')
    if evaluation_split not in ('val', 'test'):
        raise ValueError('evaluation_split must be val or test')
    if clip_risk_tolerance is not None and (not np.isfinite(clip_risk_tolerance) or not 0 <= clip_risk_tolerance <= 1):
        raise ValueError('clip risk tolerance must be finite in [0,1]')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    required_splits = {'train', 'val'} if evaluation_split == 'val' else {'train', 'val', 'test'}
    load_splits = required_splits if evaluation_split == 'val' else None
    if epochs < 1 or warmup < 0 or threads < 1 or candidate_chunk_size < 1 or min(ae_lr, tm_lr) <= 0:
        raise ValueError('invalid epoch, warmup, thread, chunk or learning-rate setting')
    prepare_threads = threads if prepare_threads is None else prepare_threads
    if isinstance(prepare_threads, bool) or not isinstance(prepare_threads, int) or prepare_threads < 1:
        raise ValueError('prepare_threads must be a positive integer or None')
    if not seeds or not noise_seeds or len(set(seeds)) != len(seeds) or len(set(noise_seeds)) != len(noise_seeds):
        raise ValueError('training and noise seeds must be nonempty and unique')
    names = ('apple', 'samsung') if scheme == 'both' else (scheme,)
    manifest_path = Path(manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    archive = None
    if payload.get('kind') == 'capture_tm_acquisition_dataset':
        for name in names:
            if name not in payload.get('schemes', {}):
                raise ValueError(f'acquisition archive does not contain requested scheme {name}')
        if payload.get('render_ev') != render_ev:
            raise ValueError('archive target intent does not match requested render_ev')
        if payload.get('noise_seeds') != list(noise_seeds):
            raise ValueError('requested noise_seeds must match archived base noise seeds')
        from .pipeline import load_acquisition_manifest
        from .types import SensorProfile
        archive = load_acquisition_manifest(manifest_path, splits=load_splits)
        sensor = SensorProfile.from_dict(archive['sensor'])
        for name in names:
            if set(row['split'] for row in archive['schemes'][name]['records']) != required_splits:
                label = 'train and val' if evaluation_split == 'val' else 'train, val and test'
                raise ValueError(f'separate {label} scene splits are required for {name}')
        scene_rows = archive['schemes'][names[0]]['records']
    else:
        sensor, scenes = load_manifest(manifest_path, splits=load_splits)
        if set(scene.split for scene in scenes) != required_splits:
            raise ValueError('separate train, val and test scene splits are required' if evaluation_split == 'test'
                             else 'separate train and val scene splits are required')
        scene_rows = [{'split': scene.split, 'source_kind': scene.source_kind} for scene in scenes]
    from .capture_plan import CapturePlan, build_plan_bank, plan_features
    from .learned_policy import TemporalExposurePolicy
    from .learned_tone import ConditionalToneMapper
    from .types import CaptureAction

    torch.set_num_threads(threads)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise FileExistsError('results already exist; use a new output directory')
    report = {'version': 1, 'manifest': str(manifest_path), 'schemes': {},
              'manifest_kind': payload.get('kind'),
              'acquisition': None if archive is None else archive['acquisition'],
              'config': {'epochs': epochs, 'warmup': warmup, 'seeds': list(seeds),
                         'noise_seeds': list(noise_seeds), 'threads': threads, 'prepare_threads': prepare_threads,
                         'candidate_chunk_size': candidate_chunk_size, 'ae_lr': ae_lr, 'tm_lr': tm_lr,
                         'render_ev': render_ev, 'weights': str(weights) if weights else 'shipped style 0',
                         'device': str(device), 'evaluation_split': evaluation_split,
                         'clip_risk_tolerance': clip_risk_tolerance},
              'scene_counts': {split: sum(row['split'] == split for row in scene_rows) for split in ('train', 'val', 'test')},
              'source_kinds': dict(Counter(row['source_kind'] for row in scene_rows)),
              'evidence': 'simulation protocol experiment; real sensor calibration and real captures remain required'}
    for name in names:
        prefix = 'A' if name == 'apple' else 'S'
        plans = (build_plan_bank(name, sensor) if archive is None else [
            CapturePlan(tuple(CaptureAction(**action) for action in value['actions']),
                        tuple(value['centers_s'])) for value in archive['schemes'][name]['plans']])
        features = plan_features(plans, sensor).to(device)
        torch.set_num_threads(prepare_threads)
        try:
            if archive is None:
                cache = build_capture_cache(scenes, sensor, plans, name, output / name / 'cache',
                                            noise_seeds=noise_seeds, weights=weights,
                                            candidate_chunk_size=candidate_chunk_size, render_ev=render_ev,
                                            scene_indices={row['scene_id']: index for index, row in enumerate(payload['scenes'])})
            else:
                cache = build_archive_cache(archive, name, output / name / 'cache', weights=weights,
                                            candidate_chunk_size=candidate_chunk_size, render_ev=render_ev)
        finally:
            torch.set_num_threads(threads)
        records = cache['records']
        collected = {prefix + suffix: [] for suffix in ('00', '10', '01', '11')}
        for seed in seeds:
            _seed(int(seed))
            kwargs = {'width': 16, 'history_length': 3}
            policy = TemporalExposurePolicy(**kwargs).to(device)
            tm = ConditionalToneMapper(name, weights=weights, trainable=False).to(device).eval()
            warm = _warmstart(policy, tm, [r for r in records if r['split'] == 'train'],
                              [r for r in records if r['split'] == 'val'], features,
                              epochs=warmup, lr=ae_lr, seed=int(seed), chunk_size=candidate_chunk_size,
                              clip_risk_tolerance=clip_risk_tolerance)
            initial = {'policy_kwargs': kwargs, 'policy_state': _state(policy), 'tm_state': _state(tm),
                       'warmstart': warm, 'sensor': sensor.to_dict() if archive is None else archive['sensor'],
                       'plans': cache['plans'],
                       'render_ev': render_ev, 'manifest_kind': payload.get('kind'),
                       'acquisition': None if archive is None else archive['acquisition'],
                       'clip_risk_tolerance': clip_risk_tolerance}
            seed_dir = output / name / f'seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(initial, seed_dir / 'initial.pt')
            _json(seed_dir / 'warmstart.json', warm)
            for suffix in ('00', '10', '01', '11'):
                group = prefix + suffix
                result = _train_group(group, name, initial, records, features, seed_dir / group,
                                      epochs=epochs, seed=int(seed), ae_lr=ae_lr, tm_lr=tm_lr,
                                      chunk_size=candidate_chunk_size, weights=weights,
                                      evaluation_split=evaluation_split)
                collected[group].append(result)
                print(f'{name} seed={seed} {group}: {evaluation_split}_cost={result["means"]["cost"]:.6f}, '
                      f'selected_epoch={result["selected_epoch"]}', flush=True)
        groups = {}
        for group, runs in collected.items():
            rows = [{**row, 'training_seed': run['seed']} for run in runs for row in run['per_scene']]
            groups[group] = {'means': _mean_metrics(rows), 'per_scene': rows,
                             'by_condition': {condition: _mean_metrics([row for row in rows if row['condition'] == condition])
                                              for condition in sorted({row['condition'] for row in rows})},
                             'seeds': [{'seed': run['seed'], 'selected_epoch': run['selected_epoch'],
                                        'means': run['means'], 'actions': run['actions']} for run in runs]}
        contrasts = paired_contrasts({suffix: groups[prefix + suffix]['per_scene']
                                     for suffix in ('00', '10', '01', '11')})
        report['schemes'][name] = {'groups': groups, 'contrasts': contrasts,
                                   'plans': cache['plans'], 'cache_seconds': cache['seconds']}
        _json(output / name / 'results.json', report['schemes'][name])
    _json(output / 'results.json', report)
    return report
