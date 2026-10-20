#!/usr/bin/env python3
"""Read-only AE–TM/S24 preflight. Header checks are NOT numerical-domain validation."""
from __future__ import annotations
import argparse
import ast
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

ROLES = {'raw_images': {'.png'}, 'denoised_raw_images': {'.png'},
         'srgb_images_style_0': {'.jpg','.jpeg'}, 'data': {'.json'}}


def audit_source(root: Path) -> dict:
    root = Path(root)
    cli = root / 'capture_tm/joint_cli.py'
    train = root / 'capture_tm/joint_experiment.py'
    result = {'joint_cli_exists': cli.exists(), 'trainer_exists': train.exists(),
              'device_cli_declared': False, 'device_parameter_declared': False,
              'cuda_execution_verified': False}
    if cli.exists():
        tree = ast.parse(cli.read_text(encoding='utf-8'))
        result['device_cli_declared'] = any(isinstance(n, ast.Constant) and n.value == '--device'
                                           for n in ast.walk(tree))
    if train.exists():
        tree = ast.parse(train.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'run_factorial':
                args = n.args.posonlyargs + n.args.args + n.args.kwonlyargs
                result['device_parameter_declared'] = any(a.arg == 'device' for a in args)
    result['note'] = 'Flags/signatures are indicators only; actual parameter/tensor device, backward and reload must be tested.'
    return result


def png_header(path: Path) -> dict:
    with path.open('rb') as f:
        data = f.read(33)
    if len(data) < 33 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        raise ValueError('invalid PNG signature/IHDR')
    if struct.unpack('>I', data[8:12])[0] != 13:
        raise ValueError('invalid IHDR length')
    width, height, bits, color, compression, filtering, interlace = struct.unpack('>IIBBBBB',data[16:29])
    if width <= 0 or height <= 0:
        raise ValueError('invalid image dimensions')
    return {'width':width, 'height':height, 'bits':bits, 'color_type':color}


def finite_number(value) -> bool:
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def valid_color_metadata(data: dict) -> bool:
    illum, ccm = data.get('cam_illum'), data.get('ccm')
    return (isinstance(illum,list) and len(illum)==3 and all(finite_number(x) and x>0 for x in illum)
            and isinstance(ccm,list) and len(ccm)==3
            and all(isinstance(row,list) and len(row)==3 and all(finite_number(x) for x in row) for row in ccm))


def audit_s24(root: Path | None, sample_limit: int = 16) -> dict:
    result = {'status':'blocked','training_ready':False,'splits':{},'cross_split_stem_collisions':[],
              'scope':'train/val file pairing, sampled PNG headers and color metadata only; test untouched',
              'not_checked':['pixel decoding and ranges','RGB/BGR order','geometry alignment','WB/CCM application',
                             'content duplicates/source-family leakage','source clipping','privacy-mask handling']}
    if root is None or not Path(root).is_dir():
        result['reason'] = 'S24_DATA_ROOT was not set or does not point to an accessible dataset root.'
        return result
    if isinstance(sample_limit, bool) or sample_limit < 1:
        raise ValueError('sample_limit must be positive')
    split_ids, blocked = {}, False
    for split in ('train','val'):
        maps, errors, duplicates = {}, [], []
        for role, extensions in ROLES.items():
            folder = Path(root) / split / role
            values = {}
            if not folder.is_dir():
                errors.append(f'missing folder: {split}/{role}')
            else:
                for p in sorted(folder.iterdir()):
                    if p.is_file() and p.suffix.lower() in extensions:
                        if p.stem in values:
                            duplicates.append(f'{role}/{p.stem}')
                        values[p.stem] = p
            maps[role] = values
        ids = set().union(*(set(v) for v in maps.values()))
        complete = set.intersection(*(set(v) for v in maps.values()))
        incomplete = sorted(ids - complete)
        for stem in sorted(complete)[:sample_limit]:
            try:
                noisy = png_header(maps['raw_images'][stem])
                clean = png_header(maps['denoised_raw_images'][stem])
                for header in (noisy,clean):
                    if header['bits'] != 16 or header['color_type'] != 2:
                        raise ValueError('RAW must be 16-bit truecolor RGB PNG')
                if (noisy['width'],noisy['height']) != (clean['width'],clean['height']):
                    raise ValueError('noisy/denoised dimensions differ')
                data = json.loads(maps['data'][stem].read_text(encoding='utf-8'))
                if not isinstance(data,dict) or not valid_color_metadata(data):
                    raise ValueError('cam_illum/ccm missing, malformed or invalid')
            except (OSError,ValueError,TypeError,struct.error) as error:
                errors.append(f'{stem}: {error}')
        split_ids[split] = ids
        result['splits'][split] = {'counts':{k:len(v) for k,v in maps.items()},'complete_pairs':len(complete),
                                   'headers_sampled':min(sample_limit,len(complete)),
                                   'incomplete_pair_count':len(incomplete),'incomplete_examples':incomplete[:10],
                                   'duplicate_stems':duplicates[:10],'errors':errors[:20]}
        blocked |= bool(errors or duplicates or incomplete or not complete)
    collision = sorted(split_ids['train'] & split_ids['val'])
    result['cross_split_stem_collisions'] = collision[:20]
    blocked |= bool(collision)
    result['status'] = 'blocked' if blocked else 'header_checks_passed'
    result['note'] = ('Stem collisions need review; different stems can still share content. '
                      'No content-duplicate or native-RAW-validity claim is made.')
    return result


def runtime_probe() -> dict:
    script = '''import json, torch
out={'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
     'visible_device_count':torch.cuda.device_count(),'devices':[],'cuda_training_verified':False}
for i in range(torch.cuda.device_count()):
 p=torch.cuda.get_device_properties(i)
 out['devices'].append({'index':i,'name':p.name,'total_gib':round(p.total_memory/2**30,2)})
print(json.dumps(out))'''
    try:
        p = subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=40)
        if p.returncode == 0:
            return json.loads(p.stdout.strip().splitlines()[-1])
        return {'status':'unavailable','cuda_training_verified':False,
                'reason':'Torch/CUDA probe failed; inspect dependencies on GPU host. No GPU claim made.'}
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
        return {'status':'unavailable','cuda_training_verified':False,'reason':'Runtime probe unavailable or timed out.'}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--s24-root',type=Path)
    parser.add_argument('--sample-limit',type=int,default=16)
    args=parser.parse_args()
    data_root=args.s24_root or (Path(os.environ['S24_DATA_ROOT']) if os.environ.get('S24_DATA_ROOT') else None)
    try:
        source=audit_source(args.repo_root)
        data=audit_s24(data_root,args.sample_limit)
        report={'source':source,'runtime':runtime_probe(),'s24':data,
                'readiness':{'preflight_executed':True,'cuda_training_ready':False,'s24_training_ready':False,
                             'next_gate':'R1 actual CUDA smoke and R2 S24 numerical-domain integration remain required'}}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({'preflight':'completed','data_status':data['status'],'training_ready':False}))
        return 0
    except (OSError,ValueError,SyntaxError) as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
