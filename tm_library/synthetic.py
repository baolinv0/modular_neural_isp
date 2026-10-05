"""Deterministic paired fixtures for software validation, not real-camera IQ.

Outputs train.jsonl and val.jsonl with disjoint IDs/scenes and float32 NPY
inputs, sRGB targets, soft person/skin/sky masks, and confidence. Targets use
one known scalar luminance transform and the standard sRGB transfer function.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def linear_to_srgb(image):
    return np.where(image <= .0031308, image * 12.92,
                    1.055 * np.power(np.maximum(image, 0), 1 / 2.4) - .055).astype(np.float32)


def known_target(image: np.ndarray) -> np.ndarray:
    """Neutral scalar compression/brightening, applied equally to RGB channels."""
    y = image @ np.array([.2126,.7152,.0722],np.float32)
    new_y = 1.35 * y / (1 + .35 * y)
    gain = np.divide(new_y, y, out=np.full_like(y,1.35), where=y>1e-8)
    return linear_to_srgb(np.clip(image * gain[...,None],0,1))


def generate(output, num_train=8, num_val=2, size=64, seed=123):
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
            target = known_target(image)
            identifier = f'{split}_{index:04d}'
            values = {'input':image,'target':target,'semantics':masks,
                      'confidence':np.ones((size,size),np.float32)}
            row = {'id':identifier,'scene':f'synthetic_{split}_{index:04d}',
                   'camera':'synthetic_fixture','input_encoding':'linear_srgb',
                   'target_aligned':True}
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
    args = vars(parser.parse_args(argv))
    for split,path in generate(**args).items():
        print(f'{split}: {path}')


if __name__=='__main__':
    main()
