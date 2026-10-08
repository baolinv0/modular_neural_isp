"""Evaluate paired aligned targets using the configuration stored in a checkpoint."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from .data import PairedImageDataset
from .engine import load_model,move_batch,deployment_semantics,apply_provider
from .metrics import image_metrics,aggregate


def synchronize(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    elif device.type=='mps':
        torch.mps.synchronize()


def evaluate(checkpoint,manifest,output,device='cpu',cpu_threads=1,semantic_config=None,semantic_checkpoint=None):
    torch.set_num_threads(cpu_threads)
    device=torch.device(device)
    model,payload=load_model(checkpoint,device)
    provider,segmentation=deployment_semantics(payload,model.config,device,semantic_config,semantic_checkpoint)
    dataset=PairedImageDataset(manifest,semantic_channels=model.config.semantic_channels,require_target=True)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    records=[]; durations=[]; combined_durations=[]
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            batch=move_batch(batch,device)
            if index==0:
                warmed=apply_provider(batch,provider,model.config.semantic_mode,False)
                model(warmed['input'],warmed['semantics'],warmed['confidence'],return_maps=False)
                synchronize(device)
            synchronize(device); start=time.perf_counter()
            conditioned=apply_provider(batch,provider,model.config.semantic_mode,False)
            result=model(conditioned['input'],conditioned['semantics'],conditioned['confidence'],return_maps=False)
            synchronize(device); combined_durations.append((time.perf_counter()-start)*1000)
            # Reuse ready semantic tensors to measure only the TM renderer.
            synchronize(device); start=time.perf_counter()
            result=model(conditioned['input'],conditioned['semantics'],conditioned['confidence'],return_maps=False)
            synchronize(device); durations.append((time.perf_counter()-start)*1000)
            batch=conditioned
            if not result['output'].isfinite().all():
                raise FloatingPointError('nonfinite model output during evaluation')
            metrics=image_metrics(result['output'],batch['target'],batch['semantics'],batch['semantic_valid'],batch['confidence'],linear=result['linear'])
            records.append({'id':batch['id'][0],'scene':batch['scene'][0],'camera':batch['camera'][0],
                            'scenario':batch.get('scenario',['unknown'])[0],'analysis_size':model.config.analysis_size,
                            'height':batch['input'].shape[-2],'width':batch['input'].shape[-1],'metrics':metrics})
    report=aggregate(records)
    report.update(model_config=model.config.to_dict(),segmentation=segmentation,checkpoint_epoch=payload.get('epoch'),images=records,
                  parameters={'total':sum(p.numel() for p in model.parameters()),
                              'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)},
                  latency={'device':str(device),'cpu_threads':cpu_threads,'batch_size':1,
                           'scope':'TM-only reuses already prepared semantic tensors; segmentation_plus_tm includes segmenter preprocessing, inference, semantic alignment and TM forward. Both exclude image IO, host-to-device transfer, metrics and export; first sample pipeline warm-up excluded; synchronized before and after each timing.',
                           'external_segmenter_active':provider is not None,
                           'tm_only':{'mean_ms':sum(durations)/len(durations),'per_image_ms':durations},
                           'segmentation_plus_tm':{'mean_ms':sum(combined_durations)/len(combined_durations),'per_image_ms':combined_durations},
                           'mean_ms':sum(durations)/len(durations),'per_image_ms':durations},
                  conventions={'aggregation':'arithmetic mean of per-image metrics; each region mean excludes unlabelled/zero-mass images',
                               'psnr':'sRGB unit data range, MSE floor 1e-12 (perfect image = 120 dB)',
                               'ssim':'RGB Gaussian 11x11 sigma 1.5, replicated borders, population covariance',
                               'luma':'Rec.709 weighted sRGB code values; not linear-light luminance',
                               'regions':'soft semantic * validity * confidence; default channel order person/skin/sky',
                               'linear_stage':'pre-chroma operator-stage compared with decoded final sRGB target; diagnostic, not calibrated intermediate-stage ground truth',
                               'preclip':'final display preclip values are unavailable here; no final-display clipping diagnostic is reported'})
    output=Path(output)
    path=output if output.suffix.lower()=='.json' else output/'metrics.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--device',default='cpu')
    parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--semantic-config',help='YAML semantic adapter override for S2 deployment')
    parser.add_argument('--semantic-checkpoint',help='Override external segmentation weight path for S2')
    args=parser.parse_args()
    report=evaluate(args.checkpoint,args.manifest,args.output,args.device,args.cpu_threads,args.semantic_config,args.semantic_checkpoint)
    print(json.dumps(report['global'],allow_nan=False))


if __name__=='__main__':
    main()
