"""Reusable Bayer acquisition archives for single-frame and HDR AE/TM.

Scene values use one relative radiance scale for the entire episode. Targets
and preview observations are shared across candidates and both schemes. The
archive stores physical measurements before any learned TM or virtual gain.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch

from .capture_plan import (CAPTURE_WINDOW_S, CapturePlan, build_plan_bank,
                           compose_captures, rule_plan_index)
from .types import CaptureAction, Scene, SensorProfile


def _profile(value):
    from .acquisition import AcquisitionProfile
    if value is None:
        return AcquisitionProfile()
    return AcquisitionProfile.from_dict(value) if isinstance(value, dict) else value


def _seed(seed, *identity):
    """Stable across record ordering, output paths and Python hash randomization."""
    payload = json.dumps([int(seed), *identity], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63 - 1)


def validate_capture_protocol(plans, acquisition, preview_meta=None):
    """Validate full rolling shutter and readout, not only nominal shutters."""
    acquisition = _profile(acquisition)
    if not plans:
        raise ValueError('capture plan bank must be nonempty')
    tol = 1e-10
    for plan in plans:
        starts = [t - (a.exposure_s + acquisition.rolling_shutter_s) / 2
                  for t, a in zip(plan.centers_s, plan.actions)]
        ends = [t + (a.exposure_s + acquisition.rolling_shutter_s) / 2 + acquisition.readout_s
                for t, a in zip(plan.centers_s, plan.actions)]
        if any(a.digital_gain != 1. for a in plan.actions):
            raise ValueError('physical acquisition fixes digital_gain=1; virtual gain belongs in TM')
        if starts[0] < CAPTURE_WINDOW_S[0] - tol or ends[-1] > CAPTURE_WINDOW_S[1] + tol:
            raise ValueError('rolling shutter plus readout exceeds capture window')
        if any(end > start + tol for end, start in zip(ends[:-1], starts[1:])):
            raise ValueError('capture slots overlap after rolling shutter/readout')
        if preview_meta is not None:
            if max(preview_meta['readout_ends_s']) >= min(starts) - tol:
                raise ValueError('preview readout must finish before final shutter begins')
    return True


def build_acquisition_plans(scheme, sensor, acquisition=None):
    """Keep fixed geometry/time slots while reserving actual readout time.

    All HDR shutters are reduced by a common factor when necessary; this keeps
    bracket ratios and equal-EV gain alternatives while fitting a 100 ms budget.
    """
    acquisition = _profile(acquisition)
    plans = build_plan_bank(scheme, sensor)
    if scheme == 'samsung':
        usable = 1 / 30 - acquisition.rolling_shutter_s - 2 * acquisition.readout_s
        if usable <= 0:
            raise ValueError('no exposure duration remains in HDR slot after rolling shutter/readout')
        longest = max(a.exposure_s for p in plans for a in p.actions)
        factor = min(1., usable / longest)
        plans = [CapturePlan(tuple(CaptureAction(a.exposure_s * factor, a.analog_gain, 1.)
                                  for a in p.actions), p.centers_s) for p in plans]
    validate_capture_protocol(plans, acquisition)
    return plans


def observe_raw_previews(scene, sensor, acquisition=None, seed=0):
    """Three historical Bayer observations with real effective exposure state."""
    from .acquisition import capture_raw
    acquisition = _profile(acquisition)
    action = CaptureAction(min(sensor.reference_exposure_s, 1 / 120))
    centers = (-4 / 30, -3 / 30, -2 / 30)
    starts = [t - (action.exposure_s + acquisition.rolling_shutter_s) / 2 for t in centers]
    ends = [t + (action.exposure_s + acquisition.rolling_shutter_s) / 2 + acquisition.readout_s
            for t in centers]
    if any(e > s + 1e-10 for e, s in zip(ends[:-1], starts[1:])):
        raise ValueError('preview shutter/readout intervals overlap')
    results = [capture_raw(scene, action, sensor, acquisition=acquisition, center_s=t,
                           seed=_seed(seed, 'preview', i)) for i, t in enumerate(centers)]
    previews = torch.stack([r.rgb.clamp(0, 1) for r in results])
    state = previews.new_tensor([[math.log2(action.exposure_s / sensor.reference_exposure_s), 0., 0.]]).repeat(3, 1)
    metadata = {'centers_s': list(centers),
                'shutter_ends_s': [r.metadata['shutter_interval_s'][1] for r in results],
                'readout_ends_s': [r.metadata['readout_end_s'] for r in results],
                'action': action.to_dict(), 'domain': 'bounded_demosaiced_sensor_linear_rgb',
                'causal': True, 'noise_seed': int(seed),
                'captures': [r.metadata for r in results]}
    return previews, state, metadata


def _plan_dict(plan):
    return {'actions': [a.to_dict() for a in plan.actions], 'centers_s': list(plan.centers_s),
            'integration_time_s': sum(a.exposure_s for a in plan.actions)}


def _record(scene, sensor, acquisition, scheme, plans, previews, state, preview_meta,
            *, noise_seeds, seed, render_ev, noisy):
    from .acquisition import capture_raw
    from .joint_experiment import _sharp_reference
    from .joint_objective import fixed_target
    validate_capture_protocol(plans, acquisition, preview_meta)
    sharp = _sharp_reference(scene, sensor)
    target = fixed_target(sharp[None], render_ev=render_ev)[0]
    effective = [n + {'train': 0, 'val': 5000, 'test': 10000}[scene.split] for n in noise_seeds]
    banks = {key: [] for key in ('images', 'reliability', 'missing', 'capture_ev',
                                 'radiance_mse', 'raw_dn', 'raw_noise_variance',
                                 'raw_saturation_mask')}
    metadata = []
    for repeat in effective:
        values = {key: [] for key in banks}
        meta_repeat = []
        for index, plan in enumerate(plans):
            results = [capture_raw(scene, action, sensor, acquisition=acquisition, center_s=center,
                        seed=_seed(seed, scene.source_id, scene.scene_id, scene.split, scheme,
                                   int(repeat), index, frame), noisy=noisy)
                       for frame, (action, center) in enumerate(zip(plan.actions, plan.centers_s))]
            fused = compose_captures(results, sensor)
            for key in ('images', 'reliability', 'missing'):
                values[key].append(fused['image' if key == 'images' else key].cpu())
            ev = float(fused['capture_ev'])
            values['capture_ev'].append(torch.tensor(ev))
            values['radiance_mse'].append(((fused['image'] * (2**-ev) - sharp)**2).mean().cpu())
            for key in ('raw_dn', 'raw_noise_variance', 'raw_saturation_mask'):
                values[key].append(torch.stack([getattr(r, key).cpu() for r in results]))
            meta_repeat.append([r.metadata for r in results])
        for key in banks:
            banks[key].append(torch.stack(values[key]))
        metadata.append(meta_repeat)
    return {'scene_id': scene.scene_id, 'source_id': scene.source_id, 'split': scene.split,
            'source_kind': scene.source_kind, 'provenance': dict(scene.provenance),
            'scheme': scheme, 'previews': previews.cpu(), 'state': state.cpu(),
            'preview_metadata': preview_meta,
            'rule_index': int(rule_plan_index(previews, state, plans, sensor)),
            'sharp_reference': sharp.cpu(), 'target': target.cpu(),
            'subject_mask': None if scene.subject_mask is None else scene.subject_mask.cpu(),
            'noise_seeds': effective, 'num_plans': len(plans), 'render_ev': float(render_ev),
            'capture_metadata': metadata,
            **{key: torch.stack(values) for key, values in banks.items()}}


def generate_acquisition_dataset(manifest, output, *, scheme='both', acquisition=None,
        noise_seeds=(0, 1), seed=0, threads=1, render_ev=0., noisy=True):
    """Stream scenes, export RAW candidates once, publish manifest atomically."""
    from .acquisition import prepare_scene
    from .pipeline_sources import iter_source_scenes
    if scheme not in ('apple', 'samsung', 'both'):
        raise ValueError('scheme must be apple, samsung or both')
    if (not noise_seeds or len(set(noise_seeds)) != len(noise_seeds)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in noise_seeds)):
        raise ValueError('noise seeds must be nonempty unique nonnegative integers')
    if not math.isfinite(render_ev) or abs(render_ev) > 16:
        raise ValueError('render_ev must be finite within [-16,16]')
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('output already exists; choose a new acquisition directory')
    manifest = Path(manifest).expanduser().resolve()
    source_payload = json.loads(manifest.read_text(encoding='utf-8'))
    sensor = SensorProfile.from_dict(source_payload['sensor'])
    acquisition = _profile(acquisition)
    names = ('apple', 'samsung') if scheme == 'both' else (scheme,)
    plans = {name: build_acquisition_plans(name, sensor, acquisition) for name in names}
    profile = acquisition.to_dict()
    info = {'version': 2, 'kind': 'capture_tm_acquisition_dataset', 'sensor': sensor.to_dict(),
            'acquisition': profile, 'seed': int(seed), 'noise_seeds': list(noise_seeds),
            'render_ev': float(render_ev), 'source_manifest': str(manifest),
            'noisy': bool(noisy), 'capture_window_s': list(CAPTURE_WINDOW_S),
            'target_definition': 'fixed_target(post-optics pixel-area linear reference at t=0; fixed WB/CCM)',
            'evidence': 'relative-radiance simulation; real-device calibration and training gains unverified',
            'schemes': {name: {'plans': [_plan_dict(p) for p in plans[name]], 'records': []} for name in names}}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            for index, (_, source) in enumerate(iter_source_scenes(manifest)):
                scene = prepare_scene(source, acquisition)
                preview_seed = _seed(seed, scene.source_id, scene.scene_id, 'history')
                previews, state, preview_meta = observe_raw_previews(scene, sensor, acquisition, preview_seed)
                for name in names:
                    sample = _record(scene, sensor, acquisition, name, plans[name], previews, state,
                        preview_meta, noise_seeds=noise_seeds, seed=seed, render_ev=render_ev, noisy=noisy)
                    relative = f'{name}/scene_{index:06d}.pt'
                    destination = staging / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(sample, destination)
                    info['schemes'][name]['records'].append({key: sample[key] for key in
                        ('scene_id', 'source_id', 'split', 'source_kind', 'provenance')} | {'path': relative})
                print(f'acquisition {index + 1}: {scene.scene_id} ({scene.split}), '
                      f'{scene.frames.shape[-2]}x{scene.frames.shape[-1]}, {scheme}', flush=True)
        if not any(value['records'] for value in info['schemes'].values()):
            raise ValueError('source manifest has no scenes')
        (staging / 'manifest.json').write_text(json.dumps(info, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        load_acquisition_manifest(staging / 'manifest.json')
        # Rename publishes only a fully validated archive. Existing output is never removed.
        if output.exists():
            raise FileExistsError('output appeared during generation')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        torch.set_num_threads(old_threads)
    return output / 'manifest.json'


def load_acquisition_manifest(path, *, validate_records=True, splits=None):
    from .pipeline_validation import load_acquisition_manifest as load
    return load(path, validate_records=validate_records, splits=splits)


def validate_acquisition_dataset(path, output=None, *, verify_composition=True, threads=1):
    """Validate RAW→RGB composition from observations without recapturing scenes.

    The training loader validates schema/targets/clocks without repeating
    fusion. This standalone check additionally reconstructs recorded fusion
    inputs from Bayer DN and checks every composed image and reliability mask.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('validation threads must be a positive integer')
    manifest = load_acquisition_manifest(path)
    report = {'valid': True, 'version': 1, 'manifest': str(Path(path).resolve()),
              'sensor_calibration_status': manifest['sensor']['calibration_status'],
              'acquisition': manifest['acquisition'], 'schemes': {},
              'evidence': manifest.get('evidence'),
              'raw_composition_verified': bool(verify_composition),
              'limitations': ['relative-radiance anchor is not absolute photometric calibration',
                  'bilinear Bayer demosaic and fixed global-translation fusion are research baselines',
                  'demosaic noise is correlated; fusion observed-RGB variance is an approximation',
                  'validation checks simulator/schema consistency, not real-camera training benefit']}
    for name, value in manifest['schemes'].items():
        rows = value['records']
        if verify_composition:
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(threads)
            try:
                _verify_composition(rows, value['plans'], manifest)
            finally:
                torch.set_num_threads(previous_threads)
        report['schemes'][name] = {'scenes': len(rows), 'plans': len(value['plans']),
            'splits': {split: sum(r['split'] == split for r in rows) for split in ('train', 'val', 'test')},
            'clean_references_declared': sum(bool(r['provenance'].get('clean_reference_verified')) for r in rows),
            'source_kinds': sorted({r['source_kind'] for r in rows})}
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


@torch.no_grad()
def _verify_composition(rows, plan_values, manifest):
    from .acquisition import demosaic_bayer
    from .types import CaptureResult
    sensor = SensorProfile.from_dict(manifest['sensor'])
    pattern = manifest['acquisition']['cfa_pattern']
    adc_range = 2**sensor.bit_depth - 1 - sensor.black_level_dn
    for row in rows:
        sample = torch.load(row['path'], map_location='cpu', weights_only=True)
        for repeat in range(len(sample['noise_seeds'])):
            for candidate, plan in enumerate(plan_values):
                captures = []
                for frame, action_value in enumerate(plan['actions']):
                    raw = sample['raw_dn'][repeat, candidate, frame]
                    rgb = demosaic_bayer((raw - sensor.black_level_dn) / adc_range, pattern)
                    # Legacy latent fields are intentionally ignored by F0;
                    # all statistics are estimated from observed RGB/profile.
                    captures.append(CaptureResult(rgb, torch.zeros_like(rgb),
                        torch.zeros_like(rgb, dtype=torch.bool), CaptureAction(**action_value),
                        sample['capture_metadata'][repeat][candidate][frame]))
                composed = compose_captures(captures, sensor)
                expected = {'images': composed['image'], 'reliability': composed['reliability'],
                            'missing': composed['missing']}
                for key, tensor in expected.items():
                    recorded = sample[key][repeat, candidate]
                    equal = torch.equal(tensor, recorded) if tensor.dtype == torch.bool else torch.allclose(
                        tensor, recorded, atol=2e-6, rtol=2e-6)
                    if not equal:
                        raise ValueError(f'RAW composition disagrees with {key}: {row["scene_id"]}, candidate {candidate}')
                mse = (composed['image'] * 2**-composed['capture_ev'] - sample['sharp_reference']).square().mean()
                if not torch.allclose(mse, sample['radiance_mse'][repeat, candidate], atol=2e-6, rtol=2e-6):
                    raise ValueError('RAW composition radiance error differs from archived reference')
