"""Acquisition archive checks shared by training and the standalone validator."""
import copy
import json
import math
from pathlib import Path

import torch

from .capture_plan import CapturePlan
from .joint_objective import fixed_target
from .types import CaptureAction, SensorProfile
from .data import _payload_hash


def _plans(value, scheme):
    if not isinstance(value, list) or not value:
        raise ValueError('archive needs nonempty plans')
    try:
        plans = [CapturePlan(tuple(CaptureAction(**a) for a in p['actions']), tuple(p['centers_s']))
                 for p in value]
    except (TypeError, KeyError) as error:
        raise ValueError('invalid archived plans') from error
    count = 1 if scheme == 'apple' else 3
    if any(len(p.actions) != count for p in plans):
        raise ValueError(f'{scheme} plans must have {count} frame(s)')
    return plans


def _tensor(sample, key, shape, *, boolean=False, nonnegative=False):
    value = sample.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'archive {key} must have shape {shape}')
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(f'archive {key} must be boolean')
    elif value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError(f'archive {key} must be finite float32 data')
    elif nonnegative and (value < 0).any():
        raise ValueError(f'archive {key} must be nonnegative')
    return value


def _validate_record(sample, row, plans, manifest, sensor, acquisition):
    from .pipeline import validate_capture_protocol
    for key in ('scene_id', 'source_id', 'split', 'source_kind'):
        if sample.get(key) != row[key]:
            raise ValueError(f'archive record {key} disagrees with manifest')
    if sample.get('provenance') != row.get('provenance'):
        raise ValueError('archive record provenance disagrees with manifest')
    if sample.get('render_ev') != manifest['render_ev']:
        raise ValueError('archive target render_ev disagrees with manifest')
    if sample.get('num_plans') != len(plans):
        raise ValueError('archive candidate count disagrees with plans')
    repeats, candidates, frames = len(manifest['noise_seeds']), len(plans), len(plans[0].actions)
    images = sample.get('images')
    if not isinstance(images, torch.Tensor) or images.ndim != 5:
        raise ValueError('archive images must be [R,K,3,H,W]')
    h, w = images.shape[-2:]
    if min(h, w) < 12 or h % 2 or w % 2:
        raise ValueError('native Bayer/TM resolution must be even and at least 12')
    _tensor(sample, 'images', (repeats, candidates, 3, h, w), nonnegative=True)
    _tensor(sample, 'reliability', (repeats, candidates, 1, h, w), nonnegative=True)
    if (sample['reliability'] > 1).any():
        raise ValueError('archive reliability must be within [0,1]')
    _tensor(sample, 'missing', (repeats, candidates, 3, h, w), boolean=True)
    _tensor(sample, 'capture_ev', (repeats, candidates))
    expected_ev = torch.tensor([math.log2(p.actions[0].exposure_s / sensor.reference_exposure_s
                                         * p.actions[0].analog_gain)
                                if frames == 1 else 0. for p in plans])
    if ((sample['capture_ev'].abs() > 16).any()
            or not torch.allclose(sample['capture_ev'], expected_ev[None].expand(repeats, -1), atol=1e-6, rtol=1e-6)):
        raise ValueError('capture EV does not match archived physical plan/common HDR radiance scale')
    _tensor(sample, 'radiance_mse', (repeats, candidates), nonnegative=True)
    raw_shape = (repeats, candidates, frames, 1, h, w)
    raw = _tensor(sample, 'raw_dn', raw_shape, nonnegative=True)
    if (raw > 2**sensor.bit_depth - 1).any() or not torch.equal(raw, raw.round()):
        raise ValueError('native RAW DN must be quantized within ADC range')
    _tensor(sample, 'raw_noise_variance', raw_shape, nonnegative=True)
    _tensor(sample, 'raw_saturation_mask', raw_shape, boolean=True)
    sharp = _tensor(sample, 'sharp_reference', (3, h, w), nonnegative=True)
    target = _tensor(sample, 'target', (3, h, w), nonnegative=True)
    if not torch.allclose(target, fixed_target(sharp[None], manifest['render_ev'])[0], atol=2e-6, rtol=2e-6):
        raise ValueError('archive target differs from fixed independent reference')
    _tensor(sample, 'previews', (3, 3, h, w), nonnegative=True)
    _tensor(sample, 'state', (3, 3))
    if (sample['previews'] > 1).any():
        raise ValueError('AE previews must be bounded observations')
    if sample.get('subject_mask') is not None:
        mask = _tensor(sample, 'subject_mask', (1, h, w), nonnegative=True)
        if (mask > 1).any():
            raise ValueError('subject mask must lie within [0,1]')
    if (isinstance(sample.get('rule_index'), bool) or not isinstance(sample.get('rule_index'), int)
            or sample.get('rule_index') not in range(candidates)):
        raise ValueError('archive rule index must be an actual candidate')
    effective_seeds = [int(n) + {'train': 0, 'val': 5000, 'test': 10000}[row['split']]
                       for n in manifest['noise_seeds']]
    if sample.get('noise_seeds') != effective_seeds:
        raise ValueError('archived effective noise seeds do not match split protocol')
    preview = sample.get('preview_metadata', {})
    if (preview.get('causal') is not True or len(preview.get('readout_ends_s', [])) != 3
            or not all(math.isfinite(float(t)) for t in preview['readout_ends_s'])):
        raise ValueError('archive must declare three causal completed preview readouts')
    validate_capture_protocol(plans, acquisition, preview)
    preview_captures = preview.get('captures', [])
    if len(preview_captures) != 3 or len(preview.get('centers_s', [])) != 3:
        raise ValueError('archive must retain complete preview capture metadata')
    for i, meta in enumerate(preview_captures):
        try:
            action = CaptureAction(**meta['action'])
            center = float(meta['center_s'])
            end = center + (action.exposure_s + acquisition.rolling_shutter_s) / 2
            readout = end + acquisition.readout_s
            state = torch.tensor([math.log2(action.exposure_s / sensor.reference_exposure_s),
                                  math.log2(action.analog_gain), math.log2(action.digital_gain)])
            valid = (math.isfinite(center) and action.digital_gain == 1.
                and abs(center - preview['centers_s'][i]) < 1e-9
                and abs(end - preview['shutter_ends_s'][i]) < 1e-9
                and abs(readout - preview['readout_ends_s'][i]) < 1e-9
                and abs(readout - meta['readout_end_s']) < 1e-9
                and meta['acquisition'] == acquisition.to_dict()
                and torch.allclose(sample['state'][i], state, atol=1e-6, rtol=1e-6))
        except (KeyError, TypeError, ValueError, IndexError):
            valid = False
        if not valid:
            raise ValueError('preview capture clock/profile/state disagrees with observed metadata')
    marker = row['provenance'].get('physical_acquisition', {})
    if marker.get('profile') != acquisition.to_dict():
        raise ValueError('archive acquisition profile differs from prepared source profile')
    metadata = sample.get('capture_metadata')
    if not isinstance(metadata, list) or len(metadata) != repeats:
        raise ValueError('capture metadata must cover noise repeats')
    for bank in metadata:
        if len(bank) != candidates:
            raise ValueError('capture metadata must cover every candidate')
        for plan, records in zip(plans, bank):
            if len(records) != frames:
                raise ValueError('capture metadata must cover every frame')
            for a, center, meta in zip(plan.actions, plan.centers_s, records):
                try:
                    clock = [float(meta['center_s']), float(meta['readout_end_s']),
                             *[float(t) for t in meta['shutter_interval_s']]]
                except (KeyError, TypeError, ValueError):
                    clock = []
                if len(clock) != 4 or not all(math.isfinite(t) for t in clock):
                    raise ValueError('final capture clock must be finite center/shutter/readout metadata')
                start = center - (a.exposure_s + acquisition.rolling_shutter_s) / 2
                end = center + (a.exposure_s + acquisition.rolling_shutter_s) / 2
                if (meta.get('action') != a.to_dict()
                        or meta.get('acquisition') != acquisition.to_dict()
                        or abs(float(meta.get('center_s', math.inf)) - center) > 1e-9
                        or len(meta.get('shutter_interval_s', [])) != 2
                        or any(abs(float(x) - y) > 1e-9 for x, y in zip(meta['shutter_interval_s'], (start, end)))
                        or abs(float(meta.get('readout_end_s', math.inf)) - end - acquisition.readout_s) > 1e-9):
                    raise ValueError('capture metadata differs from physical acquisition profile/exposure/readout plan')


def load_acquisition_manifest(path, *, validate_records=True):
    """Validate metadata and optionally samples, returning relocatable absolute rows."""
    from .acquisition import AcquisitionProfile
    from .pipeline import validate_capture_protocol
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('version') != 2 or value.get('kind') != 'capture_tm_acquisition_dataset':
        raise ValueError('expected version 2 capture_tm_acquisition_dataset manifest')
    sensor = SensorProfile.from_dict(value.get('sensor'))
    acquisition = AcquisitionProfile.from_dict(value.get('acquisition'))
    seeds = value.get('noise_seeds')
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in seeds)):
        raise ValueError('archive noise_seeds must be unique nonnegative integers')
    if not math.isfinite(float(value.get('render_ev', math.nan))) or abs(value['render_ev']) > 16:
        raise ValueError('archive render_ev must be finite within [-16,16]')
    schemes = value.get('schemes')
    if not isinstance(schemes, dict) or not schemes or not set(schemes) <= {'apple', 'samsung'}:
        raise ValueError('archive requires apple/samsung schemes')
    result = copy.deepcopy(value)
    sources = {}
    targets = {}
    for scheme, info in result['schemes'].items():
        plans = _plans(info.get('plans'), scheme)
        validate_capture_protocol(plans, acquisition)
        rows = info.get('records')
        if not isinstance(rows, list) or not rows:
            raise ValueError('archive needs nonempty record rows')
        identities = set()
        for row in rows:
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k]
                    for k in ('scene_id', 'source_id', 'split', 'source_kind', 'path')):
                raise ValueError('archive rows need scene/source/split/path strings')
            if row['split'] not in ('train', 'val', 'test') or row['scene_id'] in identities:
                raise ValueError('invalid split or duplicate scene within scheme')
            identities.add(row['scene_id'])
            provenance = row.get('provenance', {})
            if not isinstance(provenance, dict):
                raise ValueError('archive provenance must be an object')
            keys = [row['source_id'], *provenance.get('dependency_ids', [])]
            for key in ('original_frames_sha256', 'frames_sha256', 'input_sha256'):
                if provenance.get(key):
                    keys.append(provenance[key])
            for key in keys:
                if not isinstance(key, str) or not key:
                    raise ValueError('source dependency identities must be nonempty strings')
                if key in sources and sources[key] != row['split']:
                    raise ValueError('source or dependency crosses dataset split')
                sources[key] = row['split']
            sample_path = Path(row['path'])
            sample_path = sample_path if sample_path.is_absolute() else path.parent / sample_path
            row['path'] = str(sample_path.resolve())
            if validate_records:
                sample = torch.load(row['path'], map_location='cpu', weights_only=True)
                _validate_record(sample, row, plans, value, sensor, acquisition)
                target_identity = _payload_hash(sample['target'].numpy())
                if row['scene_id'] in targets and targets[row['scene_id']] != target_identity:
                    raise ValueError('Apple/Samsung targets for same scene must be identical')
                targets[row['scene_id']] = target_identity
    return result
