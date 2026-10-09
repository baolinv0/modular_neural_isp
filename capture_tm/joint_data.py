"""Diverse analytic scenes for protocol verification, not a phone benchmark."""
from pathlib import Path
import numpy as np
import torch
from .types import Scene, SensorProfile
from .data import _write_scene


def write_joint_dataset(directory, *, scenes=40, size=32, seed=0):
    """Every split contains all four conditions; require complete quartets.

    Each scene uses its own random geometry/color/exposure. Temporal samples
    cover causal previews and all future shutter windows. The latent signal is
    clean linear RGB with HDR highlights; the capture simulator supplies noise.
    """
    if scenes < 12 or scenes % 4 or size < 16:
        raise ValueError('require scenes>=12 in multiples of4 and size>=16')
    directory = Path(directory).expanduser().resolve()
    if (directory / 'manifest.json').exists():
        raise FileExistsError('manifest already exists')
    rng = np.random.default_rng(seed)
    sensor = SensorProfile(provenance={'source': 'joint analytic protocol', 'seed': seed})
    groups = scenes // 4
    train_groups = max(1, int(.6 * groups))
    val_groups = max(1, int(.2 * groups))
    train_groups = min(train_groups, groups - val_groups - 1)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)].astype(np.float32)
    times = np.linspace(-.25, .15, 65, dtype=np.float64)
    conditions = ('dark_static', 'dark_motion', 'backlit', 'ordinary')
    for index in range(scenes):
        group = index // 4
        split = 'train' if group < train_groups else 'val' if group < train_groups + val_groups else 'test'
        condition = conditions[index % 4]
        color = rng.uniform(.45, 1., (3, 1, 1)).astype(np.float32)
        ambient = {'dark_static': .006, 'dark_motion': .009, 'backlit': .045, 'ordinary': .23}[condition]
        ambient *= float(2 ** rng.uniform(-1, 1))
        speed = float(rng.uniform(2, 5)) if condition == 'dark_motion' else (float(rng.uniform(.4, 1.4)) if condition == 'backlit' else 0.)
        phase, cy = float(rng.uniform(-.6, .6)), float(rng.uniform(-.3, .3))
        frequency = float(rng.uniform(9, 22))
        background = ambient * (1.5 + .35 * x + .2 * np.sin(x * frequency) * np.sin(y * frequency))
        frames = []
        for t in times:
            cx = phase + speed * t
            mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / .08)
            texture = .65 + .35 * np.sin((x - cx) * 35) * np.cos((y - cy) * 29)
            subject = mask * texture * ambient * 4
            highlight_strength = float(8 if condition == 'backlit' else .5)
            highlight = highlight_strength * np.exp(-((x + .65) ** 2 + (y - .6) ** 2) / .025)
            frames.append((background + subject)[None] * color + highlight[None])
        subject_mask = torch.from_numpy((((x - phase) ** 2 + (y - cy) ** 2) < .16).astype(np.float32)[None])
        scene = Scene(f'joint_{index:04d}', split, torch.from_numpy(np.stack(frames).astype(np.float32)),
                      torch.from_numpy(times), subject_mask, 'synthetic',
                      provenance={'generator': 'analytic_joint_conditions_v1', 'condition': condition,
                                  'seed': seed, 'speed_normalized_per_s': speed,
                                  'clean_reference_verified': True, 'absolute_radiance_calibrated': False})
        path = _write_scene(directory, scene, sensor)
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenes', type=int, default=40)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(write_joint_dataset(args.output, scenes=args.scenes, size=args.size, seed=args.seed))
