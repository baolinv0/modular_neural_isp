"""Target-free inference, PNG export, and lossless numeric diagnostic maps."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from .data import PairedImageDataset
from .engine import load_model,move_batch,deployment_semantics,apply_provider


def _stem(identifier):
    cleaned=re.sub(r'[^A-Za-z0-9_.-]+','_',identifier).strip('._')[:80] or 'image'
    return cleaned+'_'+hashlib.sha256(identifier.encode()).hexdigest()[:10]


def _flatten_maps(value,prefix='',result=None):
    result={} if result is None else result
    if torch.is_tensor(value):
        result[prefix]=value.detach().cpu().numpy()
    elif isinstance(value,dict):
        for key,child in value.items():
            _flatten_maps(child,f'{prefix}/{key}' if prefix else str(key),result)
    elif isinstance(value,(list,tuple)):
        for index,child in enumerate(value):
            _flatten_maps(child,f'{prefix}/{index}',result)
    return result


def infer(checkpoint,manifest,output,device='cpu',save_maps=False,cpu_threads=1,semantic_config=None,semantic_checkpoint=None):
    torch.set_num_threads(cpu_threads)
    model,payload=load_model(checkpoint,device)
    provider,segmentation=deployment_semantics(payload,model.config,device,semantic_config,semantic_checkpoint)
    dataset=PairedImageDataset(manifest,semantic_channels=model.config.semantic_channels,require_target=False)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    records=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0):
            batch=move_batch(batch,device)
            batch=apply_provider(batch,provider,model.config.semantic_mode,False)
            result=model(batch['input'],batch['semantics'],batch['confidence'],return_maps=save_maps)
            image=result['output'][0]
            if not image.isfinite().all():
                raise FloatingPointError('nonfinite model output during inference')
            stem=_stem(batch['id'][0]); filename=stem+'.png'
            pixels=(image.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(pixels).save(output/filename)
            record={'id':batch['id'][0],'scene':batch['scene'][0],'camera':batch['camera'][0],
                    'image':filename,'height':image.shape[-2],'width':image.shape[-1]}
            if save_maps:
                maps=_flatten_maps(result.get('maps',{}))
                np.savez_compressed(output/(stem+'.npz'),**maps)
                record.update(maps=stem+'.npz',map_shapes={key:list(value.shape) for key,value in maps.items()})
            records.append(record)
    metadata={'model_config':model.config.to_dict(),'checkpoint_epoch':payload.get('epoch'),
              'encoding':'8-bit sRGB PNG; diagnostic NPZ values are unnormalized float arrays including batch axes',
              'semantic_mode':model.config.semantic_mode,'segmentation':segmentation,'images':records}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest'); inputs.add_argument('--input',help='Single demosaiced RGB image (not Bayer raw)')
    parser.add_argument('--input-encoding',choices=['linear_srgb','srgb','camera_rgb'])
    parser.add_argument('--metadata',help='camera_rgb metadata JSON path')
    parser.add_argument('--semantics'); parser.add_argument('--confidence')
    parser.add_argument('--output',required=True); parser.add_argument('--device',default='cpu')
    parser.add_argument('--save-maps',action='store_true'); parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--semantic-config',help='YAML semantic adapter override for S2 deployment')
    parser.add_argument('--semantic-checkpoint',help='Override external segmentation weight path for S2')
    args=parser.parse_args()
    if args.input:
        if not args.input_encoding:
            parser.error('--input requires explicit --input-encoding')
        row={'id':Path(args.input).stem,'input':str(Path(args.input).resolve()),'input_encoding':args.input_encoding}
        for key in ('metadata','semantics','confidence'):
            if getattr(args,key):
                row[key]=str(Path(getattr(args,key)).resolve())
        with tempfile.TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'input.json'
            manifest.write_text(json.dumps([row]))
            metadata=infer(args.checkpoint,manifest,args.output,args.device,args.save_maps,args.cpu_threads,args.semantic_config,args.semantic_checkpoint)
    else:
        if any((args.input_encoding,args.metadata,args.semantics,args.confidence)):
            parser.error('single-input options require --input; manifest rows already specify these fields')
        metadata=infer(args.checkpoint,args.manifest,args.output,args.device,args.save_maps,args.cpu_threads,args.semantic_config,args.semantic_checkpoint)
    print(f"Saved {len(metadata['images'])} image(s) to {args.output}")


if __name__=='__main__':
    main()
