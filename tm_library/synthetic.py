"""Deterministic paired fixtures for software validation, not real-camera IQ.

Outputs train.jsonl and val.jsonl with disjoint IDs/scenes and float32 NPY
inputs, sRGB targets, soft person/skin/sky masks, and confidence. Targets use
one known scalar luminance transform in the default ``global`` task.
The ``semantic`` task uses role and illumination dependent bounded scalar gains
and explicitly protected scenes. Shapes, lighting, labels, and targets are
software fixtures only: they do not model physical portraits or establish IQ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEMANTIC_SCENARIOS = ('portrait_backlight', 'two_people_unequal', 'side_light',
                      'no_person_window', 'night_silhouette', 'already_good')


def semantic_target(image: np.ndarray, semantics: np.ndarray, scenario: str) -> np.ndarray:
    """Software target with bounded role+illumination exposure, never whitening.

    Corrective cases brighten darker subject pixels by at most 1.5x; luminance
    determines the gain, not a fixed skin brightness or RGB target. Protected
    scenarios retain their input linear values exactly before display encoding.
    Inputs obey the library normalized linear sRGB contract.
    """
    if scenario not in SEMANTIC_SCENARIOS:
        raise ValueError(f'unknown semantic scenario {scenario}')
    if image.shape != semantics.shape or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('semantic target requires aligned HWC RGB and three-channel masks')
    if not np.isfinite(image).all() or not np.isfinite(semantics).all():
        raise ValueError('semantic target requires finite pixels')
    if (image < 0).any() or (image > 1).any() or (semantics < 0).any() or (semantics > 1).any():
        raise ValueError('semantic target requires normalized [0,1] pixels')
    if scenario in ('no_person_window', 'night_silhouette', 'already_good'):
        return linear_to_srgb(image)
    y = image @ np.array([.2126,.7152,.0722], np.float32)
    # Person and skin are roles; skin is not assigned a target color/brightness.
    role = np.clip(.8*semantics[...,0] + .2*semantics[...,1], 0, 1)
    darkness = np.clip((.22-y)/.22, 0, 1)
    gain = 1 + .5*role*darkness
    return linear_to_srgb(np.clip(image*gain[...,None], 0, 1))


def _semantic_fixture(xx, yy, rng, scenario):
    person = np.zeros_like(xx); skin = np.zeros_like(xx)
    centers = (.29,.72) if scenario == 'two_people_unequal' else (.5,)
    colors = np.stack([.05+.1*xx, .07+.1*xx, .1+.1*xx], axis=-1)
    # Independent color draws cover different reflectance/color combinations;
    # the target always keeps channel ratios under a scalar exposure gain.
    skin_colors = np.array([[.38,.21,.13], [.18,.105,.075], [.12,.065,.045],
                            [.46,.32,.23], [.25,.15,.12]], np.float32)
    if scenario != 'no_person_window':
        for index, cx in enumerate(centers):
            body = np.exp(-((xx-cx)**2/.018 + (yy-.64)**2/.075))
            face = np.exp(-((xx-cx)**2/.008 + (yy-.39)**2/.011))
            person = np.maximum(person, np.maximum(body, face))
            skin = np.maximum(skin, face)
            color = skin_colors[int(rng.integers(len(skin_colors)))]
            colors = colors*(1-face[...,None]) + color*face[...,None]
            if scenario == 'two_people_unequal':
                illumination = .28 if index == 0 else 1.1
                colors *= 1-body[...,None]*(1-illumination)
    sky = np.clip((.3-yy)*5, 0, 1)
    if scenario in ('portrait_backlight','no_person_window'):
        window = np.clip((xx-.6)*4,0,1)*(1-person)
        colors = colors*(1-window[...,None]) + np.array([.75,.8,.9])*window[...,None]
        if scenario == 'portrait_backlight':
            colors *= 1-.65*person[...,None]
    elif scenario == 'side_light':
        colors *= (.22 + .9*xx)[...,None]
    elif scenario == 'night_silhouette':
        colors *= .13*(1-.7*person[...,None])
    elif scenario == 'already_good':
        colors *= 1.2
    image = np.clip(colors,0,1).astype(np.float32)
    masks = np.stack([person,skin,sky],axis=-1).astype(np.float32)
    return image, masks


def linear_to_srgb(image):
    return np.where(image <= .0031308, image * 12.92,
                    1.055 * np.power(np.maximum(image, 0), 1 / 2.4) - .055).astype(np.float32)


def known_target(image: np.ndarray) -> np.ndarray:
    """Neutral scalar compression/brightening, applied equally to RGB channels."""
    y = image @ np.array([.2126,.7152,.0722],np.float32)
    new_y = 1.35 * y / (1 + .35 * y)
    gain = np.divide(new_y, y, out=np.full_like(y,1.35), where=y>1e-8)
    return linear_to_srgb(np.clip(image * gain[...,None],0,1))


def generate(output, num_train=8, num_val=2, size=64, seed=123, task='global'):
    if task not in ('global','semantic'):
        raise ValueError('task must be global or semantic')
    for name, value, minimum in [('num_train',num_train,1),('num_val',num_val,1),('size',size,1)]:
        if not isinstance(value,int) or isinstance(value,bool) or value<minimum:
            raise ValueError(f'{name} must be a positive integer')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    paths = {}
    for split, count, stream in [('train',num_train,0),('val',num_val,1)]:
        # Independent split streams mean changing train count does not change val.
        rng = np.random.default_rng(np.random.SeedSequence([seed,stream]))
        directory = output/split
        directory.mkdir(exist_ok=True)
        yy, xx = np.mgrid[0:size,0:size].astype(np.float32)
        xx = (xx+.5)/size
        yy = (yy+.5)/size
        rows = []
        for index in range(count):
            center_x = rng.uniform(.38,.62)
            center_y = rng.uniform(.5,.7)
            body = np.exp(-((xx-center_x)**2/.032 + (yy-center_y)**2/.08))
            skin = np.exp(-((xx-center_x)**2/.012 + (yy-(center_y-.18))**2/.02))
            sky = np.clip((.45-yy)*5,0,1)
            masks = np.stack([body,skin,sky],axis=-1).astype(np.float32)
            base = .02 + .27*xx + .12*yy
            colors = np.stack([base*.85,base,base*1.1],axis=-1)
            colors = colors*(1-skin[...,None]) + np.array([.43,.26,.17],np.float32)*skin[...,None]
            colors = colors*(1-.5*sky[...,None]) + np.array([.1,.2,.4],np.float32)*(.5*sky[...,None])
            exposure = rng.uniform(.55,1.3)
            noise = rng.normal(0,.005,(size,size,3))
            image = np.clip(colors*exposure+noise,0,1).astype(np.float32)
            scenario = 'global_transform'
            if task == 'semantic':
                scenario = SEMANTIC_SCENARIOS[index % len(SEMANTIC_SCENARIOS)]
                image, masks = _semantic_fixture(xx, yy, rng, scenario)
                target = semantic_target(image, masks, scenario)
            else:
                target = known_target(image)
            identifier = f'{split}_{index:04d}'
            values = {'input':image,'target':target,'semantics':masks,
                      'confidence':np.ones((size,size),np.float32)}
            row = {'id':identifier,'scene':f'synthetic_{split}_{index:04d}',
                   'camera':'synthetic_fixture','input_encoding':'linear_srgb',
                   'target_aligned':True,'scenario':scenario,
                   'burst_id':f'synthetic_{split}_burst_{index:04d}',
                   'subject_id':f'synthetic_{split}_subject_{index:04d}',
                   'synthetic_task':task}
            for field, array in values.items():
                filename = f'{identifier}_{field}.npy'
                np.save(directory/filename,array)
                row[field] = f'{split}/{filename}'
                if array.ndim==3:
                    row[f'{field}_layout']='HWC'
            rows.append(row)
        path = output/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        paths[split] = path
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--num-train',type=int,default=8)
    parser.add_argument('--num-val',type=int,default=2)
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--task',choices=('global','semantic'),default='global')
    args = vars(parser.parse_args(argv))
    for split,path in generate(**args).items():
        print(f'{split}: {path}')


if __name__=='__main__':
    main()
