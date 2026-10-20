"""Reproduce why a censored preview cannot certify HDR highlight preservation."""
import json
import torch
from capture_tm.types import SensorProfile, Scene
from capture_tm.capture_plan import plan_features
from capture_tm.acquisition import AcquisitionProfile, capture_raw
from capture_tm.pipeline import build_acquisition_plans, observe_raw_previews
from capture_tm.learned_policy import preview_clipping_guard


def main():
    torch.set_num_threads(1)
    sensor, acquisition = SensorProfile(), AcquisitionProfile()
    plans = build_acquisition_plans('samsung', sensor, acquisition)
    scene = Scene('censored-preview-witness', 'val',
                  torch.full((1, 3, 32, 32), 2.), torch.tensor([0.]))
    previews, state, _ = observe_raw_previews(scene, sensor, acquisition, seed=2026)
    features = plan_features(plans, sensor)
    mask, risk = preview_clipping_guard(previews[None], state[None], features[None],
                                       torch.tensor([0]), tolerance=.01)
    measurements = [capture_raw(scene, action, sensor, acquisition=acquisition,
                    center_s=center, noisy=False) for action, center in zip(plans[11].actions, plans[11].centers_s)]
    result = {'scene': 'static uniform relative radiance 2; 32x32 RGGB',
        'preview_noisy': True, 'final_capture_noisy': False,
        'guard_predicted_all_frame_clipping': float(risk[0, 11]),
        'guard_keeps_plan11': bool(mask[0, 11]),
        'actual_native_all_frame_saturation': float(torch.stack(
            [r.raw_saturation_mask for r in measurements]).all(0).float().mean()),
        'max_plan_min_frame_relative_exposure': float(torch.exp2(features.sum(-1)).amin(-1).max()),
        'threshold': .98,
        'explanation': 'censored preview is a radiance lower bound; .94 predicted scale cannot trigger .98 threshold',
        'evidence': 'separate deterministic simulator counterexample; not a quality benchmark'}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
