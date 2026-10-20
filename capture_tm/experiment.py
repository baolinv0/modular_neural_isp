"""Offline candidate labels, causal policy learning and reproducible evaluation."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .control import ExposureConstraints
from .data import load_manifest
from .modular import AnalyticBackend, ModularPhotofinishingBackend
from .objective import ScoreWeights, quality_cost
from .policy import ExposurePolicy, action_features, ranking_loss
from .simulation import action_grid, capture
from .tone import PatentToneMapper
from .types import CaptureAction


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _action_dict(action):
    return {'exposure_s': action.exposure_s, 'analog_gain': action.analog_gain,
            'digital_gain': action.digital_gain}


def _sharp(scene, center_s):
    """Offline-only latent target at the common future exposure midpoint."""
    if len(scene.frame_times_s) == 1:
        return scene.frames[0]
    times = scene.frame_times_s
    if center_s < float(times[0]) or center_s > float(times[-1]):
        raise ValueError('Target time falls outside scene support')
    j = min(int(torch.searchsorted(times, times.new_tensor(center_s))), len(times) - 1)
    if j == 0:
        return scene.frames[0]
    alpha = (center_s - float(times[j - 1])) / float(times[j] - times[j - 1])
    return scene.frames[j - 1] * (1 - alpha) + scene.frames[j] * alpha


def _frontend(rgb, sensor):
    """Fixed WB/CCM; no candidate-dependent auto brightness or color decisions."""
    wb = rgb.new_tensor(sensor.wb).reshape(3, 1, 1)
    ccm = rgb.new_tensor(sensor.ccm).reshape(3, 3)
    return torch.einsum('ij,jhw->ihw', ccm, rgb * wb).clamp_min(0)


def _preview(rgb):
    # Explicit policy observation: normalized sensor RGB, with clipping only in
    # the analysis thumbnail. The renderer still receives unbounded RGB.
    return rgb.clamp(0, 1)


def _constraints(actions, overrides=None):
    cfg = {'min_exposure_s': min(a.exposure_s for a in actions),
           'max_exposure_s': max(a.exposure_s for a in actions),
           'min_analog_gain': min(a.analog_gain for a in actions),
           'max_analog_gain': max(a.analog_gain for a in actions),
           'max_total_capture_s': .08, 'readout_s': .002,
           'control_delay_frames': 1}
    cfg.update(overrides or {})
    return ExposureConstraints(**cfg), cfg


def _temporal_mask(actions, observation_exposure_s, future_offset_s, readout_s):
    # The observation is usable only after its shutter and readout finish.
    return torch.tensor([future_offset_s-a.exposure_s/2 >=
                         observation_exposure_s/2+readout_s-1e-12
                         for a in actions], dtype=torch.bool)


def _implementation_identity():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ('experiment.py', 'objective.py', 'tone.py', 'modular.py',
                 'simulation.py', 'types.py', 'policy.py', 'control.py', 'data.py', 'sources.py'):
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    digest.update((root.parent/'photofinishing/photofinishing_model.py').read_bytes())
    digest.update((root.parent/'utils/constants.py').read_bytes())
    return digest.hexdigest()


def _physics_estimate(preview, observation_actions, actions, sensor):
    """Observable physical heuristic, not a clean-radiance or future oracle.

    Censored preview radiance is a lower-bound proxy. Motion is estimated from
    historical image change; this baseline is intentionally reported as an
    estimate, not a calibrated optimizer with guarantees.
    """
    if len(observation_actions) != len(preview):
        raise ValueError('Observed action history must match preview history')
    scales = preview.new_tensor([a.exposure_s/sensor.reference_exposure_s *
        a.analog_gain*a.digital_gain for a in observation_actions])[:, None, None, None]
    normalized = preview/scales
    proxy = normalized[-1]
    observation_action = observation_actions[-1]
    # Brightness changes induced by prior t/g decisions are not object motion.
    # Censoring and observation noise still bias this simple motion estimate.
    motion = float((normalized[-1] - normalized[0]).square().mean())
    result = []
    dn_span = (2 ** sensor.bit_depth - 1) - sensor.black_level_dn
    for a in actions:
        tr = a.exposure_s / sensor.reference_exposure_s
        photons = proxy * tr * sensor.full_well_e
        variance = (photons + sensor.read_noise_e ** 2) / max((tr * sensor.full_well_e) ** 2, 1e-12)
        variance = variance + (sensor.adc_noise_dn / dn_span / max(a.analog_gain * tr, 1e-8)) ** 2
        predicted = proxy * tr * a.analog_gain
        clipped = ((photons >= sensor.full_well_e) | (predicted >= 1.)).float().mean()
        result.append(float(variance.mean()) + .04 * float(clipped)
                      + motion * (a.exposure_s / observation_action.exposure_s) ** 2)
    return torch.tensor(result)


def build_oracle(manifest, output_dir, *, backend='modular', tm_mode='apple', checkpoint=None,
                 exposures_s=None, analog_gains=None, repeats=3, seed=0,
                 constraints=None, render_intent_ev=0., capture_center_s=.025,
                 preview_centers_s=(-.04, -.02, 0.), observation_action=None,
                 observation_actions=None, weights=None):
    """Simulate candidates and label final quality under one frozen appearance.

    Clean future scene data are used only by the offline teacher. Cache policy
    inputs contain three *past/current* observations and actual exposure state.
    """
    if backend not in ('analytic', 'modular') or tm_mode not in ('apple', 'samsung'):
        raise ValueError('Unknown renderer')
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if (len(preview_centers_s) != 3 or not all(math.isfinite(t) for t in preview_centers_s)
            or not all(a < b for a, b in zip(preview_centers_s, preview_centers_s[1:]))):
        raise ValueError('Three chronological preview times are required')
    if not math.isfinite(capture_center_s) or capture_center_s <= preview_centers_s[-1]:
        raise ValueError('Capture must follow the policy observation')
    if not math.isfinite(render_intent_ev):
        raise ValueError('Nonfinite render intent')
    sensor, scenes = load_manifest(manifest)
    actions = action_grid(exposures_s or [1/240, 1/120, 1/60, 1/30], analog_gains or [1., 2., 4.])
    limits, limits_cfg = _constraints(actions, constraints)
    feasible = limits.feasible_mask(actions)
    if not feasible.any():
        raise ValueError('No feasible capture actions')
    ref = CaptureAction(sensor.reference_exposure_s, 1., 1.)
    observation_action = observation_action or ref
    if observation_actions is not None:
        if not observation_actions:
            raise ValueError('observation_actions cannot be empty')
        observed_choices = [a if isinstance(a, CaptureAction) else CaptureAction(**a)
                            for a in observation_actions]
    else:
        observed_choices = [observation_action]
    rng = np.random.default_rng(seed)
    protocol = {'preview_centers_s': list(preview_centers_s), 'capture_center_s': capture_center_s,
                'future_center_offset_s': capture_center_s-preview_centers_s[-1],
                'observation_actions': [_action_dict(a) for a in observed_choices],
                'noise_seed': seed, 'noise_repeats': repeats,
                'budget_scope': 'remaining budget for one final capture; prior previews excluded',
                'control_scope': 'common future midpoint offline labels; driver schedules effective-frame requests'}
    weights = weights if isinstance(weights, ScoreWeights) else ScoreWeights(**(weights or {}))
    base = AnalyticBackend() if backend == 'analytic' else ModularPhotofinishingBackend(checkpoint)
    renderer = PatentToneMapper(tm_mode, backend=base).eval()
    for p in renderer.parameters():
        p.requires_grad_(False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for i, scene in enumerate(scenes):
            observed = [observed_choices[int(rng.integers(len(observed_choices)))] for _ in range(3)]
            for j in range(2):
                if (preview_centers_s[j]+observed[j].exposure_s/2+limits.readout_s >
                        preview_centers_s[j+1]-observed[j+1].exposure_s/2+1e-12):
                    raise ValueError('Chronological preview shutter/readout windows overlap')
            feasible = limits.feasible_mask(actions) & _temporal_mask(actions, observed[-1].exposure_s,
                protocol['future_center_offset_s'], limits.readout_s)
            if not feasible.any():
                raise ValueError('No feasible candidate after observation shutter/readout')
            preview_captures = [capture(scene, observed[j], sensor,
                center_s=t, seed=seed + i*10000 + j, noisy=True)
                for j, t in enumerate(preview_centers_s)]
            previews = torch.stack([_preview(c.rgb) for c in preview_captures])
            state = action_features(observed, sensor.reference_exposure_s)
            target = renderer(_frontend(_sharp(scene, capture_center_s), sensor).unsqueeze(0),
                              capture_bias_ev=0., render_intent_ev=render_intent_ev)['output']
            costs, component_rows, sample_outputs, candidate_metadata = [], [], [], []
            for k, a in enumerate(actions):
                if not bool(feasible[k]):
                    costs.append(float('inf'))
                    component_rows.append({})
                    sample_outputs.append(torch.zeros_like(target[0]))
                    candidate_metadata.append([])
                    continue
                bias = math.log2(a.scale(ref))
                clean = capture(scene, a, sensor, center_s=capture_center_s, seed=0, noisy=False)
                clean_output = renderer(_frontend(clean.rgb, sensor).unsqueeze(0),
                    capture_bias_ev=bias, render_intent_ev=render_intent_ev)['output']
                captures = [capture(scene, a, sensor, center_s=capture_center_s,
                    seed=seed + i*10000 + 100 + k*repeats + n, noisy=True) for n in range(repeats)]
                rendered = renderer(torch.stack([_frontend(c.rgb, sensor) for c in captures]),
                    capture_bias_ev=bias, render_intent_ev=render_intent_ev)['output']
                metrics = quality_cost(rendered, target, torch.stack([c.saturation_mask for c in captures]),
                    subject_mask=scene.subject_mask, clean_output=clean_output, weights=weights)
                row = {key: float(value.mean()) for key, value in metrics.items()}
                costs.append(row['cost'])
                component_rows.append(row)
                sample_outputs.append(rendered[0].cpu())
                candidate_metadata.append({'clean': clean.metadata, 'noisy': [c.metadata for c in captures]})
            baseline = min((k for k in range(len(actions)) if feasible[k]),
                           key=lambda k: abs(math.log2(actions[k].exposure_s/ref.exposure_s))
                           + abs(math.log2(actions[k].analog_gain)) + abs(math.log2(actions[k].digital_gain)))
            records.append({'scene_id': scene.scene_id, 'source_id': getattr(scene, 'source_id', None) or scene.scene_id,
                'split': scene.split, 'source_kind': scene.source_kind, 'domain': scene.domain,
                'provenance': scene.provenance,
                'observation_actions': [_action_dict(a) for a in observed],
                'preview_capture_metadata': [c.metadata for c in preview_captures],
                'candidate_capture_metadata': candidate_metadata,
                'preview_seeds': [seed+i*10000+j for j in range(3)],
                'candidate_seeds': [[seed+i*10000+100+k*repeats+n for n in range(repeats)]
                                    for k in range(len(actions))],
                'previews': previews.cpu(), 'capture_state': state.cpu(),
                'costs': torch.tensor(costs), 'feasible_mask': feasible.cpu(),
                'components': component_rows, 'physics_scores': _physics_estimate(previews, observed, actions, sensor),
                'baseline_index': baseline, 'target': target[0].cpu(), 'outputs': torch.stack(sample_outputs),
                'observation_frame_id': 0})
    renderer_info = {'backend': backend, 'tm_mode': tm_mode, 'render_intent_ev': render_intent_ev,
                     'checkpoint': str(checkpoint) if checkpoint else 'shipped-style-0' if backend == 'modular' else None,
                     'checkpoint_sha256': getattr(base, 'checkpoint_sha256', None),
                     'renderer_identity': base.renderer_identity,
                     'implementation_sha256': _implementation_identity(),
                     'frozen': True, 'frontend': 'fixed WB/CCM; no denoiser',
                     'objective_weights': asdict(weights)}
    payload = {'version': 1, 'sensor': sensor.to_dict(), 'actions': [_action_dict(a) for a in actions],
               'constraints': limits_cfg, 'records': records, 'renderer': renderer_info,
               'repeats': repeats, 'seed': seed, 'capture_center_s': capture_center_s,
               'preview_centers_s': list(preview_centers_s),
               'protocol': protocol,
               'dataset_sha256': hashlib.sha256(json.dumps([
                   {'scene_id': s.scene_id, 'source_id': s.source_id, 'split': s.split,
                    'frame_times_s': s.frame_times_s.tolist(), 'provenance': s.provenance}
                   for s in scenes], sort_keys=True).encode()).hexdigest(),
               'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
               'evidence': 'offline simulation; not real camera quality validation'}
    path = output_dir/'oracle.pt'
    torch.save(payload, path)
    _json(output_dir/'summary.json', {k: v for k, v in payload.items() if k != 'records'} |
          {'scene_counts': {s: sum(r['split'] == s for r in records) for s in ('train', 'val', 'test')},
           'num_actions': len(actions)})
    return path


def _load_cache(path):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    required = {'version', 'sensor', 'actions', 'constraints', 'records', 'renderer',
                'protocol', 'dataset_sha256', 'manifest_sha256'}
    if not isinstance(cache, dict) or not required.issubset(cache) or cache['version'] != 1:
        raise ValueError('Invalid oracle cache schema')
    splits = {}
    for r in cache['records']:
        sid = r['source_id']
        if sid in splits and splits[sid] != r['split']:
            raise ValueError('Source identity crosses train/val/test splits')
        splits[sid] = r['split']
    return cache


def _inputs(records, features):
    return (torch.stack([r['previews'] for r in records]).to(features.device),
            torch.stack([r['capture_state'] for r in records]).to(features.device),
            features.unsqueeze(0).expand(len(records), -1, -1),
            torch.stack([r['feasible_mask'] for r in records]).to(features.device))


def train_policy(cache_path, output_dir, *, epochs=20, width=24, seed=0, lr=1e-3,
                 use_auxiliary=True, batch_size=8, device='cpu'):
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Invalid training settings')
    _seed(seed)
    cache = _load_cache(cache_path)
    train = [r for r in cache['records'] if r['split'] == 'train']
    val = [r for r in cache['records'] if r['split'] == 'val']
    if not train or not val:
        raise ValueError('Separate train and val scenes are required')
    actions = [CaptureAction(**a) for a in cache['actions']]
    reference_t = cache['sensor']['reference_exposure_s']
    features = action_features(actions, reference_t, device=device)
    kwargs = {'num_actions': len(actions), 'width': width, 'histogram_bins': 32,
              'history_length': 3, 'use_auxiliary': use_auxiliary}
    model = ExposurePolicy(**kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch = [], float('inf'), None
    checkpoint = output_dir/'policy.pt'
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train)).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            rows = [train[j] for j in order[start:start+batch_size]]
            scores = model(*_inputs(rows, features))
            loss = ranking_loss(scores, torch.stack([r['costs'] for r in rows]).to(device),
                                torch.stack([r['feasible_mask'] for r in rows]).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            scores = model(*_inputs(val, features))
            choices = scores.argmax(1)
            costs = torch.stack([r['costs'] for r in val]).to(device)
            regret = float((costs.gather(1, choices[:, None]).squeeze(1) - costs.min(1).values).mean())
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'val_regret': regret})
        if regret < best:
            best, best_epoch = regret, epoch + 1
            torch.save({'version': 1, 'model_kwargs': kwargs,
                        'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'actions': cache['actions'], 'sensor': cache['sensor'],
                        'constraints': cache['constraints'], 'renderer': cache['renderer'],
                        'protocol': cache['protocol'], 'dataset_sha256': cache['dataset_sha256'],
                        'manifest_sha256': cache['manifest_sha256'], 'seed': seed,
                        'selected_epoch': best_epoch, 'val_regret': best,
                        'selection': 'test records excluded from optimization and selection'}, checkpoint)
    _json(output_dir/'training.json', {'history': history, 'best_epoch': best_epoch,
          'best_val_regret': best, 'train_scenes': len(train), 'val_scenes': len(val),
          'parameters': sum(p.numel() for p in model.parameters()), 'seed': seed,
          'evidence': 'software training run; not real camera quality validation'})
    return checkpoint


def _load_policy(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != 1:
        raise ValueError('Unsupported policy checkpoint')
    model = ExposurePolicy(**state['model_kwargs'])
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, state


def _check_pair(cache, state):
    for field in ('actions', 'renderer', 'manifest_sha256', 'dataset_sha256',
                  'sensor', 'constraints', 'protocol'):
        if cache[field] != state[field]:
            raise ValueError(f'Checkpoint/cache {field} mismatch; rebuild labels or retrain')


def evaluate_policy(cache_path, checkpoint, output_dir):
    cache = _load_cache(cache_path)
    model, state = _load_policy(checkpoint)
    _check_pair(cache, state)
    test = [r for r in cache['records'] if r['split'] == 'test']
    if not test:
        raise ValueError('Held-out test scenes are required')
    features = action_features([CaptureAction(**a) for a in cache['actions']], cache['sensor']['reference_exposure_s'])
    with torch.inference_mode():
        choices = model(*_inputs(test, features)).argmax(1).tolist()
    method_rows = {key: [] for key in ('baseline', 'physics', 'oracle', 'policy')}
    for record, choice in zip(test, choices):
        legal = record['feasible_mask']
        physics = record['physics_scores'].masked_fill(~legal, float('inf')).argmin().item()
        oracle = record['costs'].argmin().item()
        for method, idx in {'baseline': record['baseline_index'], 'physics': physics,
                            'oracle': oracle, 'policy': choice}.items():
            row = {'scene_id': record['scene_id'], 'source_id': record['source_id'], 'action_index': idx,
                   **cache['actions'][idx], **record['components'][idx],
                   'capture_time_s': cache['actions'][idx]['exposure_s']+cache['constraints']['readout_s'],
                   'regret': float(record['costs'][idx] - record['costs'][oracle])}
            method_rows[method].append(row)
    summaries = {}
    keys = ('cost', 'regret', 'mse', 'subject_luma_mae', 'raw_saturation',
            'rendered_noise_mse', 'noiseless_capture_bias_mse', 'exposure_s', 'capture_time_s')
    for method, rows in method_rows.items():
        summaries[method] = {'mean_'+key: float(np.mean([r[key] for r in rows])) for key in keys}
        summaries[method]['oracle_top1'] = float(np.mean([r['action_index'] == o['action_index']
                                                       for r, o in zip(rows, method_rows['oracle'])]))
    report = {'version': 1, 'test_scenes': len(test), 'methods': summaries,
              'per_scene': method_rows, 'renderer': cache['renderer'],
              'constraints': cache['constraints'], 'noise_repeats': cache['repeats'],
              'evidence': 'held-out synthetic/source-proxy simulation, not real camera validation',
              'physics_baseline': 'current-preview radiance lower bound and history-motion heuristic'}
    output_dir = Path(output_dir)
    _json(output_dir/'evaluation.json', report)
    lines = ['# Capture-TM evaluation', '', report['evidence'], '',
             '| Method | Cost | Regret | Saturation | Exposure (ms) |', '|---|---:|---:|---:|---:|']
    for name, row in summaries.items():
        lines.append(f"| {name} | {row['mean_cost']:.6f} | {row['mean_regret']:.6f} | {row['mean_raw_saturation']:.4f} | {row['mean_exposure_s']*1000:.3f} |")
    (output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Saving outputs supports visual inspection without generating or repainting detail.
    from PIL import Image
    for r, idx in zip(test, choices):
        strip = torch.cat([r['target'], r['outputs'][r['baseline_index']],
                           r['outputs'][idx], r['outputs'][r['costs'].argmin()]], dim=2)
        array = (strip.permute(1, 2, 0).clamp(0, 1).numpy()*255).round().astype(np.uint8)
        safe_name = hashlib.sha256(r['scene_id'].encode()).hexdigest()[:12]
        Image.fromarray(array).save(output_dir/f'{safe_name}_target_baseline_policy_oracle.png')
    return report


def request_from_observations(previews, capture_state, checkpoint, *, observation_frame_id=0,
                              remaining_capture_s=None, future_center_offset_s=None):
    """Deployable causal entry point. No candidate images, costs or latent scene."""
    model, state = _load_policy(checkpoint)
    if isinstance(observation_frame_id, bool) or not isinstance(observation_frame_id, int) or observation_frame_id < 0:
        raise ValueError('observation_frame_id must be a nonnegative integer')
    actions = [CaptureAction(**a) for a in state['actions']]
    limits = ExposureConstraints(**state['constraints'])
    feasible = limits.feasible_mask(actions, remaining_capture_s=remaining_capture_s)
    offset = (state['protocol']['future_center_offset_s'] if future_center_offset_s is None
              else future_center_offset_s)
    if not math.isfinite(offset) or offset <= 0:
        raise ValueError('future_center_offset_s must be positive and finite')
    if capture_state.ndim != 2 or capture_state.shape != (3, 3) or not torch.isfinite(capture_state).all():
        raise ValueError('capture_state must be finite [3,3] physical log2 features')
    observed_t = state['sensor']['reference_exposure_s'] * float(torch.exp2(capture_state[-1, 0].double()))
    if not math.isfinite(observed_t) or observed_t <= 0:
        raise ValueError('Observed exposure must be positive and finite')
    feasible &= _temporal_mask(actions, observed_t, offset, limits.readout_s)
    if not feasible.any():
        raise ValueError('No feasible action within remaining budget and observation timing')
    features = action_features(actions, state['sensor']['reference_exposure_s'])
    with torch.inference_mode():
        scores = model(previews.unsqueeze(0), capture_state.unsqueeze(0), features.unsqueeze(0), feasible.unsqueeze(0))
    idx = int(scores.argmax(1))
    action = actions[idx]
    t0 = state['sensor']['reference_exposure_s']
    bias = math.log2(action.exposure_s/t0 * action.analog_gain * action.digital_gain)
    return {'execution': 'request_only', 'action_index': idx, 'requested_action': _action_dict(action),
            'observation_frame_id': observation_frame_id,
            'requested_frame_id': observation_frame_id+1,
            'effective_frame_id': observation_frame_id+1+limits.control_delay_frames,
            'planned_future_center_offset_s': offset,
            'remaining_capture_s': min(limits.max_total_capture_s, remaining_capture_s)
                if remaining_capture_s is not None else limits.max_total_capture_s,
            'budget_assumption': 'caller subtracts prior/pending capture costs; default is checkpoint remaining budget',
            'capture_bias_ev': bias, 'requested_render_compensation_ev': -bias,
            'render_intent_ev': state['renderer']['render_intent_ev'],
            'tm_mode': state['renderer']['tm_mode'],
            'renderer': state['renderer'], 'sensor': state['sensor'],
            'policy_checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'requirements': 'Use actual effective-frame t/ga/gd to render; request is not a captured frame.'}


def infer_request(cache_path, checkpoint, *, scene_id=None):
    """Convenient replay wrapper around the causal API; labels are not inputs."""
    cache = _load_cache(cache_path)
    _, state = _load_policy(checkpoint)
    _check_pair(cache, state)
    rows = [r for r in cache['records'] if r['scene_id'] == scene_id] if scene_id else [
        r for r in cache['records'] if r['split'] == 'test']
    if not rows:
        raise ValueError('Requested scene was not found')
    row = rows[0]
    result = request_from_observations(row['previews'], row['capture_state'], checkpoint,
                                       observation_frame_id=row['observation_frame_id'])
    result['scene_id'] = row['scene_id']
    return result
