"""Run with python -m capture_tm. Every output path is explicit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import write_demo_dataset
from .experiment import (build_oracle, evaluate_policy, infer_request,
                         request_from_observations, train_policy)
from .types import SensorProfile


def _common_train(parser):
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--image-only', action='store_true')


def _run(config, output):
    data_cfg = config.get('data', {})
    if data_cfg.get('manifest'):
        manifest = Path(data_cfg['manifest'])
    else:
        manifest = write_demo_dataset(output/'data', seed=data_cfg.get('seed', 0),
            size=data_cfg.get('size', 64), scenes=data_cfg.get('scenes', 12))
    cache = build_oracle(manifest, output/'oracle', **config.get('oracle', {}))
    checkpoint = train_policy(cache, output/'train', **config.get('train', {}))
    report = evaluate_policy(cache, checkpoint, output/'eval')
    request = infer_request(cache, checkpoint)
    (output/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    return {'manifest': str(manifest), 'cache': str(cache), 'checkpoint': str(checkpoint),
            'evaluation': str(output/'eval'/'evaluation.json'), 'test_scenes': report['test_scenes']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='C: physical exposure selection and patent-inspired TM')
    parser.add_argument('--threads', type=int, default=2)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('make-demo')
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--scenes', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('build-oracle')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--backend', choices=['analytic', 'modular'], default='modular')
    p.add_argument('--tm-mode', choices=['apple', 'samsung'], default='apple')
    p.add_argument('--checkpoint')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exposures-s', type=float, nargs='+')
    p.add_argument('--analog-gains', type=float, nargs='+')
    p = sub.add_parser('train')
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    _common_train(p)
    p = sub.add_parser('evaluate')
    p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('infer')
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--cache', help='Replay observation records; labels are not policy inputs')
    inp.add_argument('--observations', help='NPZ: previews[T,3,H,W] in [0,1], capture_state[T,3] physical log2 features')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--scene-id')
    p.add_argument('--observation-frame-id', type=int, default=0)
    p.add_argument('--remaining-capture-s', type=float,
                   help='Remaining sequence budget, including one readout per future capture')
    p.add_argument('--future-center-offset-s', type=float,
                   help='Scheduled future midpoint minus last observation midpoint (seconds)')
    p.add_argument('--output', required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('smoke')
    p.add_argument('--output', required=True)
    p.add_argument('--backend', choices=['analytic', 'modular'], default='modular')
    p.add_argument('--tm-mode', choices=['apple', 'samsung'], default='apple')
    p.add_argument('--epochs', type=int, default=2)
    p = sub.add_parser('render')
    p.add_argument('--input', required=True, help='float32 NPY [3,H,W] actual captured linear sensor RGB, HDR allowed')
    p.add_argument('--output', required=True, help='Directory for rendered NPY, PNG, curve and gain ledger')
    p.add_argument('--tm-mode', choices=['apple', 'samsung'], default='apple')
    p.add_argument('--backend', choices=['modular', 'analytic'], default='modular')
    p.add_argument('--checkpoint')
    p.add_argument('--capture-bias-ev', type=float, default=0.)
    p.add_argument('--render-intent-ev', type=float, default=0.)
    p.add_argument('--virtual-gain', type=float, default=1., help='Additional unapplied rendering gain')
    p.add_argument('--sensor-profile', help='JSON SensorProfile with fixed WB/CCM into renderer input color space')
    p = sub.add_parser('import-rawgen')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scene-id', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--xyz-to-sensor', required=True, help='JSON containing an explicit 3x3 camera matrix')
    importers = [p]
    p = sub.add_parser('import-rl3a')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scene-id', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--exposure-s', type=float, required=True)
    p.add_argument('--analog-gain', type=float, required=True)
    p.add_argument('--digital-gain', type=float, default=1.)
    importers.append(p)
    p = sub.add_parser('import-linear')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scene-id', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--input-domain', choices=['sensor_linear_relative_radiance', 'linear_xyz'], required=True)
    p.add_argument('--xyz-to-sensor', help='JSON 3x3 matrix, required for linear_xyz')
    importers.append(p)
    for importer in importers:
        importer.add_argument('--source-id', help='Original scene/episode identity shared by crops, variants and denoised/noisy pairs')
        importer.add_argument('--input-layout', choices=['HWC','CHW','THWC','TCHW'], default='HWC')
        importer.add_argument('--frame-times', help='JSON file of timestamps in seconds, required for temporal stacks')
        importer.add_argument('--sensor-profile', help='JSON SensorProfile with explicit reference exposure, WB/CCM and calibration provenance')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        output = Path(args.output)
        if args.command == 'make-demo':
            result = {'manifest': str(write_demo_dataset(output, seed=args.seed, size=args.size, scenes=args.scenes))}
        elif args.command == 'build-oracle':
            result = {'cache': str(build_oracle(args.manifest, output, backend=args.backend, tm_mode=args.tm_mode,
                 checkpoint=args.checkpoint, repeats=args.repeats, seed=args.seed,
                 exposures_s=args.exposures_s, analog_gains=args.analog_gains))}
        elif args.command == 'train':
            result = {'checkpoint': str(train_policy(args.cache, output, epochs=args.epochs, width=args.width,
                  seed=args.seed, device=args.device, use_auxiliary=not args.image_only))}
        elif args.command == 'evaluate':
            report = evaluate_policy(args.cache, args.checkpoint, output)
            result = {'evaluation': str(output/'evaluation.json'), 'test_scenes': report['test_scenes']}
        elif args.command == 'infer':
            if args.cache:
                result = infer_request(args.cache, args.checkpoint, scene_id=args.scene_id)
            else:
                with np.load(args.observations, allow_pickle=False) as data:
                    result = request_from_observations(torch.from_numpy(data['previews']).float(),
                        torch.from_numpy(data['capture_state']).float(), args.checkpoint,
                        observation_frame_id=args.observation_frame_id,
                        remaining_capture_s=args.remaining_capture_s,
                        future_center_offset_s=args.future_center_offset_s)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        elif args.command == 'run':
            config_path = Path(args.config).resolve()
            config = yaml.safe_load(config_path.read_text())
            if config.get('data', {}).get('manifest'):
                config['data']['manifest'] = str((config_path.parent/config['data']['manifest']).resolve())
            result = _run(config, output)
        elif args.command == 'smoke':
            result = _run({'data': {'size': 32, 'scenes': 9},
                'oracle': {'backend': args.backend, 'tm_mode': args.tm_mode, 'repeats': 2,
                           'exposures_s': [1/240, 1/120, 1/60], 'analog_gains': [1., 2.],
                           'observation_actions': [{'exposure_s': 1/120, 'analog_gain': g}
                                                   for g in (1., 2.)]},
                'train': {'epochs': args.epochs, 'width': 8}}, output)
        elif args.command == 'render':
            from PIL import Image
            from .experiment import _frontend
            from .modular import AnalyticBackend, ModularPhotofinishingBackend
            from .tone import PatentToneMapper
            sensor = SensorProfile.from_dict(json.loads(Path(args.sensor_profile).read_text())) if args.sensor_profile else SensorProfile()
            rgb = np.load(args.input, allow_pickle=False)
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[0] != 3:
                raise ValueError('render input must be float32 NPY [3,H,W] captured linear RGB')
            if not np.isfinite(rgb).all():
                raise ValueError('Captured RGB contains nonfinite values')
            base = ModularPhotofinishingBackend(args.checkpoint) if args.backend == 'modular' else AnalyticBackend()
            mapper = PatentToneMapper(args.tm_mode, backend=base).eval()
            with torch.inference_mode():
                rendered = mapper(_frontend(torch.from_numpy(rgb), sensor).unsqueeze(0),
                    capture_bias_ev=args.capture_bias_ev, render_intent_ev=args.render_intent_ev,
                    virtual_gain=args.virtual_gain)
            output.mkdir(parents=True, exist_ok=True)
            pixels = rendered['output'][0].numpy()
            np.save(output/'rendered.npy', pixels)
            Image.fromarray((pixels.transpose(1,2,0)*255).round().astype(np.uint8)).save(output/'rendered.png')
            np.savez(output/'curve.npz', input=rendered['tone_curve_input'].numpy(), output=rendered['tone_curve'].numpy())
            ledger = {k: rendered[k].tolist() for k in ('virtual_gain','capture_compensation_ev',
                'requested_capture_compensation_ev','requested_render_ev','applied_render_ev')}
            ledger.update({'renderer_identity': base.renderer_identity,
                'checkpoint_sha256': getattr(base,'checkpoint_sha256',None), 'tm_mode': args.tm_mode,
                'sensor': sensor.to_dict(), 'capture_bias_ev': args.capture_bias_ev,
                'render_intent_ev': args.render_intent_ev,
                'domain': 'captured sensor-linear RGB -> fixed WB/CCM -> patent-inspired TM -> display RGB',
                'input_contract': 'no previously applied additional virtual render gain'})
            (output/'render.json').write_text(json.dumps(ledger, indent=2, allow_nan=False)+'\n')
            result = {'image': str(output/'rendered.png'), 'ledger': str(output/'render.json')}
        else:
            from .sources import import_linear_scene, import_rawgen_xyz, import_rl3a_proxy
            sensor = SensorProfile.from_dict(json.loads(Path(args.sensor_profile).read_text())) if args.sensor_profile else SensorProfile()
            shared = {'scene_id': args.scene_id, 'split': args.split, 'source_id': args.source_id,
                      'input_layout': args.input_layout, 'sensor': sensor,
                      'reference_exposure_s': sensor.reference_exposure_s,
                      'frame_times_s': json.loads(Path(args.frame_times).read_text()) if args.frame_times else None}
            if args.command == 'import-rl3a':
                manifest = import_rl3a_proxy(args.input, output, **shared, exposure_s=args.exposure_s,
                    analog_gain=args.analog_gain, digital_gain=args.digital_gain)
            else:
                matrix = json.loads(Path(args.xyz_to_sensor).read_text()) if args.xyz_to_sensor else None
                if args.command == 'import-rawgen':
                    manifest = import_rawgen_xyz(args.input, output, **shared, xyz_to_sensor=matrix)
                else:
                    manifest = import_linear_scene(args.input, output, **shared,
                        input_domain=args.input_domain, xyz_to_sensor=matrix)
            result = {'manifest': str(manifest)}
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result
