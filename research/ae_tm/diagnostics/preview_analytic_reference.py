"""Post-hoc known-appearance-contract control; never feeds GT to the renderer."""
import argparse
import json
from pathlib import Path
import torch
from capture_tm.pipeline import load_acquisition_manifest
from capture_tm.joint_experiment import _load
from capture_tm.joint_objective import fixed_target, image_metrics
from research.ae_tm.diagnostics.preview_risk_ablation import interval, mean_present


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--study', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    archive = load_acquisition_manifest(args.manifest, splits={'val'})
    study = json.loads((Path(args.study) / 'summary.json').read_text())
    report = {'contract': 'same RAW and frontend; fixed_target applied to exposure-normalized captured image, not latent or GT',
              'evidence': 'post-hoc known analytic-target control; not a general expert-style renderer', 'schemes': {}}
    for scheme, prefix in [('apple', 'A'), ('samsung', 'S')]:
        physical = {r['scene_id']: r for r in archive['schemes'][scheme]['records']}
        rule = [{'scene_id': sid, 'action_index': _load(row)['rule_index'], 'training_seed': 0}
                for sid, row in physical.items()]
        guarded_ae = study['schemes'][scheme]['groups']['guarded'][prefix + '10']['per_scene']
        controls = {}
        for name, selections in [('rule_ae', rule), ('guarded_ae_only', guarded_ae)]:
            rows = []
            for selected in selections:
                raw = _load(physical[selected['scene_id']])
                idx = selected['action_index']
                measured = raw['images'][:, idx] * torch.exp2(-raw['capture_ev'][:, idx, None, None, None])
                output = fixed_target(measured)  # no target/latent argument
                metrics = image_metrics(output, raw['target'][None], raw['missing'][:, idx], subject_mask=raw['subject_mask'])
                bright = ((raw['target'] * torch.tensor([.2126, .7152, .0722])[:, None, None]).sum(0) > .8)
                row = {'scene_id': raw['scene_id'], 'action_index': idx,
                       'training_seed': selected.get('training_seed', 0), 'bright_pixel_count': int(bright.sum()),
                       **{k: float(v.mean()) for k, v in metrics.items()},
                       'radiance_mse': float(raw['radiance_mse'][:, idx].mean())}
                rows.append(row)
            means = {k: mean_present([r[k] for r in rows]) for k in metrics}
            means['radiance_mse'] = mean_present([r['radiance_mse'] for r in rows])
            means['highlight_mse_valid_regions'] = mean_present([r['highlight_mse'] for r in rows if r['bright_pixel_count']])
            joint = study['schemes'][scheme]['groups']['guarded'][prefix + '11']['per_scene']
            controls[name] = {'means': means, 'per_scene': rows,
                'guarded_joint_minus_control': {k: interval(joint, rows, k) for k in ('mse', 'detail_mae', 'highlight_mse', 'radiance_mse')}}
        report['schemes'][scheme] = controls
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({scheme: {name: value['means'] for name, value in controls.items()}
                      for scheme, controls in report['schemes'].items()}, indent=2))
    return report


if __name__ == '__main__':
    main()
