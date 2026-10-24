"""Deterministic epoch-boundary training and strict library checkpoints."""
from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .config import ModelConfig
from .data import PairedImageDataset
from .losses import CompositeLoss, LossConfig
from .model import TMModel


@dataclass
class TrainingConfig:
    epochs: int = 2
    batch_size: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    seed: int = 7
    device: str = 'cpu'
    num_workers: int = 0
    gradient_clip: float = 1.0
    scheduler_step: int = 1
    scheduler_gamma: float = 0.95
    cpu_threads: int = 1
    protocol: str = 'joint'

    def __post_init__(self):
        if self.protocol not in ('joint','operator_diagnostic'):
            raise ValueError('protocol must be joint or operator_diagnostic')
        for name in ('epochs','batch_size','scheduler_step','cpu_threads'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        # Global RNG restoration alone cannot reproduce prefetched worker states.
        if self.num_workers != 0:
            raise ValueError('num_workers must be 0 for reproducible epoch-boundary resume')
        for name in ('learning_rate','weight_decay','gradient_clip','scheduler_gamma'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.learning_rate==0 or self.gradient_clip==0 or not 0<self.scheduler_gamma<=1:
            raise ValueError('learning_rate and gradient_clip must be positive; scheduler_gamma in (0,1]')


def load_config(path):
    """Relative paths resolve against the config directory, never the caller cwd."""
    path=Path(path).resolve()
    config=yaml.safe_load(path.read_text())
    if not isinstance(config,dict):
        raise ValueError('configuration must be a YAML mapping')
    for key in ('output_dir','upstream_checkpoint'):
        if config.get(key):
            config[key]=str((path.parent / config[key]).resolve())
    for key in ('train_manifest','val_manifest'):
        if config.get('data',{}).get(key):
            config['data'][key]=str((path.parent / config['data'][key]).resolve())
    segmentation=config.get('segmentation',{})
    if segmentation.get('weights'):
        segmentation['weights']=str((path.parent / segmentation['weights']).resolve())
    return config


def validate_config(config):
    allowed={'model','data','training','loss','output_dir','upstream_checkpoint','segmentation'}
    if not isinstance(config,dict) or set(config)-allowed:
        raise ValueError(f'unknown top-level configuration keys: {sorted(set(config)-allowed)}')
    model=ModelConfig.from_dict(config.get('model',{}))
    training=TrainingConfig(**config.get('training',{}))
    loss=LossConfig(**config.get('loss',{}))
    data=dict(config.get('data',{}))
    if set(data)-{'train_manifest','val_manifest','image_size','augment','split_policy','group_checks'}:
        raise ValueError('unknown data configuration key')
    if not data.get('train_manifest') or not data.get('val_manifest'):
        raise ValueError('data requires train_manifest and val_manifest')
    if 'augment' in data and type(data['augment']) is not bool:
        raise ValueError('augment must be boolean')
    if not config.get('output_dir'):
        raise ValueError('output_dir required')
    data.setdefault('split_policy','scene')
    if data['split_policy'] not in ('scene','id_only'):
        raise ValueError('split_policy must be scene or id_only')
    groups=data.get('group_checks',[])
    if not isinstance(groups,list) or any(x not in ('burst_id','subject_id') for x in groups):
        raise ValueError('group_checks must contain burst_id and/or subject_id')
    if training.protocol=='operator_diagnostic' and (not model.freeze_backbone or model.post_mode!='fixed_reference'):
        raise ValueError('operator_diagnostic requires freeze_backbone and fixed_reference post_mode')
    if model.effective_base in ('baseline','gtm') and model.correction_algorithm=='none' and model.freeze_backbone:
        raise ValueError('baseline freeze_backbone leaves no trainable parameters')
    if not any((loss.rgb,loss.log_luma,loss.gradient)):
        raise ValueError('an ungated image loss (rgb, log_luma or gradient) is required for validation and checkpoint selection')
    return model,training,loss,data


def split_report(train_data,val_data,policy='scene',group_checks=None):
    ids=lambda dataset:{row['id'] for row in dataset.records}
    paths=lambda dataset:{str((dataset.root/row['input']).resolve()) for row in dataset.records}
    if ids(train_data)&ids(val_data) or paths(train_data)&paths(val_data):
        raise ValueError('train/validation overlap in IDs or input paths')
    result={}
    for label in ('scene','camera','burst_id','subject_id'):
        train_values={str(row[label]) for row in train_data.records if row.get(label)}
        val_values={str(row[label]) for row in val_data.records if row.get(label)}
        result[f'{label}_overlap']=sorted(train_values&val_values)
    if policy=='scene':
        for dataset in (train_data,val_data):
            if any(not row.get('scene') or str(row['scene']).strip().lower() in ('unknown','none','') for row in dataset.records):
                raise ValueError('scene split policy requires meaningful scene IDs on every row; use explicit id_only for legacy manifests')
        if result['scene_overlap']:
            raise ValueError(f"scene overlap across train/validation: {result['scene_overlap']}")
    for label in group_checks or ():
        if result[f'{label}_overlap']:
            raise ValueError(f'{label} overlap across train/validation')
    result['policy']=policy
    return result


def seed_all(seed,device='cpu'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDA grid_sample backward lacks a deterministic implementation. CPU
    # resume is bit-exact; accelerators may warn and are not bit-exact promises.
    torch.use_deterministic_algorithms(True,warn_only=torch.device(device).type!='cpu')
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark=False


def capture_rng():
    np_state=np.random.get_state()
    return {'python':random.getstate(),'numpy':(np_state[0],np_state[1].tolist(),np_state[2],np_state[3],np_state[4]),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name,values,pos,gauss,cached=state['numpy']
    np.random.set_state((name,np.asarray(values,dtype=np.uint32),pos,gauss,cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def load_checkpoint(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict):
        raise ValueError('expected a tm_library_v2 checkpoint; upstream weights use upstream_checkpoint')
    if checkpoint.get('format')=='tm_library_v1':
        raise ValueError('tm_library_v1 architecture is incompatible with v2; retrain or implement an explicit migration. Original upstream weights use upstream_checkpoint.')
    if checkpoint.get('format')!='tm_library_v2' or checkpoint.get('config',{}).get('architecture_version')!=2 or 'state_dict' not in checkpoint:
        raise ValueError('expected a tm_library_v2 checkpoint with architecture_version=2; upstream weights use upstream_checkpoint')
    return checkpoint


def load_model(path,device='cpu'):
    checkpoint=load_checkpoint(path)
    model=TMModel.from_checkpoint(checkpoint).to(device).eval()
    return model,checkpoint


def semantic_configuration(value=None):
    from .semantics import SegmentationConfig
    return SegmentationConfig.from_dict(value or {}).to_dict()


def create_semantic_provider(segmentation,model_config,device='cpu',training=False):
    """A deployment S1 path never loads or requires external segmenter weights."""
    if model_config.semantic_mode=='none' or (model_config.semantic_mode=='train_only' and not training):
        return None
    from .semantics import SemanticProvider
    if segmentation.get('backend','none')=='none' or segmentation.get('source','manifest')=='manifest':
        return None
    return SemanticProvider(segmentation,semantic_channels=model_config.semantic_channels,device=device)


def apply_provider(batch,provider,mode,training):
    from .semantics import apply_semantic_provider
    return apply_semantic_provider(batch,provider,mode,training)


def deployment_semantics(payload,model_config,device='cpu',semantic_config=None,semantic_checkpoint=None):
    config=dict(payload.get('segmentation',{}))
    if semantic_config is not None:
        if isinstance(semantic_config,(str,Path)):
            path=Path(semantic_config).resolve()
            override=yaml.safe_load(path.read_text())
            override=override.get('segmentation',override)
            if override.get('weights'):
                override['weights']=str((path.parent/override['weights']).resolve())
        else:
            override=dict(semantic_config)
        config.update(override)
    if semantic_checkpoint is not None:
        config['weights']=str(Path(semantic_checkpoint).resolve())
    config=semantic_configuration(config)
    return create_semantic_provider(config,model_config,device,training=False),config


def move_batch(batch,device):
    return {key:value.to(device) if torch.is_tensor(value) else value for key,value in batch.items()}


def run_epoch(model,loader,criterion,device,optimizer=None,gradient_clip=1,provider=None):
    training=optimizer is not None
    model.train(training)
    totals={}; count=0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch=move_batch(batch,device)
            batch=apply_provider(batch,provider,model.config.semantic_mode,training)
            if training:
                optimizer.zero_grad(set_to_none=True)
            result=model(batch['input'],batch['semantics'],batch['confidence'],return_maps=False)
            loss,terms=criterion(result,batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss; no checkpoint was written for this epoch')
            if training:
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip,error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError('nonfinite gradient')
                optimizer.step()
            size=batch['input'].shape[0]
            for name,value in {'loss':loss,**terms}.items():
                totals[name]=totals.get(name,0.)+value.detach().item()*size
            count+=size
    return {name:value/count for name,value in totals.items()}


def _signature(model_cfg,training_cfg,loss_cfg,data_cfg,segmentation):
    optim=asdict(training_cfg); optim.pop('epochs')
    data=dict(data_cfg)
    for key in ('train_manifest','val_manifest'):
        data[key]=str(Path(data[key]).resolve())
    return {'model':model_cfg.to_dict(),'training':optim,'loss':asdict(loss_cfg),'data':data,'segmentation':segmentation}


def _save(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def train(config,resume=None):
    model_cfg,settings,loss_cfg,data_cfg=validate_config(config)
    torch.set_num_threads(settings.cpu_threads)
    seed_all(settings.seed,settings.device)
    device=torch.device(settings.device)
    train_data=PairedImageDataset(data_cfg['train_manifest'],image_size=data_cfg.get('image_size'),
                                 augment=data_cfg.get('augment',False),semantic_channels=model_cfg.semantic_channels)
    val_data=PairedImageDataset(data_cfg['val_manifest'],image_size=data_cfg.get('image_size'),
                               semantic_channels=model_cfg.semantic_channels)
    separation=split_report(train_data,val_data,data_cfg['split_policy'],data_cfg.get('group_checks'))
    if settings.batch_size>1 and data_cfg.get('image_size') is None:
        augment=train_data.augment
        train_data.augment=False
        try:
            shapes={tuple(train_data[index]['input'].shape) for index in range(len(train_data))}
        finally:
            train_data.augment=augment
        if len(shapes)>1:
            raise ValueError('batch_size > 1 requires equal input shapes or explicit image_size')
    train_loader=DataLoader(train_data,batch_size=settings.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_data,batch_size=1,shuffle=False,num_workers=0)
    model=TMModel(model_cfg).to(device)
    if config.get('upstream_checkpoint') and not resume:
        model.load_upstream(config['upstream_checkpoint'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings.learning_rate,weight_decay=settings.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=settings.scheduler_step,gamma=settings.scheduler_gamma)
    segmentation=semantic_configuration(config.get('segmentation'))
    signature=_signature(model_cfg,settings,loss_cfg,data_cfg,segmentation)
    history=[]; start=0; best=float('inf'); best_epoch=0; best_state=None; best_checkpoint=None
    if resume:
        checkpoint=load_checkpoint(resume)
        required={'optimizer','scheduler','rng','training_signature','epoch','best_validation_loss','best_epoch','history'}
        if required-set(checkpoint):
            raise ValueError('resume requires a complete training checkpoint with optimizer, scheduler and RNG')
        if checkpoint.get('training_signature')!=signature:
            raise ValueError('resume configuration differs from checkpoint (only epochs and output_dir may change)')
        provider=create_semantic_provider(segmentation,model_cfg,device,training=True)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        start=checkpoint['epoch']; best=checkpoint['best_validation_loss']; best_epoch=checkpoint['best_epoch']
        history=checkpoint['history']
        best_state=checkpoint.get('best_state_dict',checkpoint['state_dict'])
        best_checkpoint=checkpoint.get('best_checkpoint')
        if best_checkpoint is None:
            source_best=Path(resume).parent/'best.pt'
            if checkpoint['epoch']==best_epoch:
                best_checkpoint=copy.deepcopy({key:value for key,value in checkpoint.items() if key!='best_state_dict'})
            elif source_best.exists():
                best_checkpoint=load_checkpoint(source_best)
            else:
                raise ValueError('resume checkpoint lacks a full best checkpoint snapshot and sibling best.pt is missing')
        restore_rng(checkpoint['rng'])
    else:
        provider=create_semantic_provider(segmentation,model_cfg,device,training=True)
    if start>=settings.epochs:
        raise ValueError('epochs must exceed the completed checkpoint epoch')
    output=Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True)
    criterion=CompositeLoss(loss_cfg)
    for epoch in range(start,settings.epochs):
        learning_rate=optimizer.param_groups[0]['lr']
        training=run_epoch(model,train_loader,criterion,device,optimizer,settings.gradient_clip,provider)
        validation=run_epoch(model,val_loader,criterion,device,provider=provider)
        scheduler.step()
        history.append({'epoch':epoch+1,'learning_rate':learning_rate,'train':training,'validation':validation})
        improved=validation['loss']<best
        if improved:
            best=validation['loss']; best_epoch=epoch+1
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        payload={'format':'tm_library_v2',**model.checkpoint(),'epoch':epoch+1,
                 'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':capture_rng(),
                 'segmentation':segmentation,'training_signature':signature,'best_validation_loss':best,'best_epoch':best_epoch,
                 'best_state_dict':best_state,'history':history,'split_report':separation,
                 'determinism':{'algorithms_enabled':True,'warn_only':device.type!='cpu',
                                'exact_resume_scope':'CPU, epoch boundaries, num_workers=0, unchanged data/runtime'}}
        if improved:
            # Clone optimizer tensors: state_dict() references mutable Adam moments.
            # Never nest snapshots recursively. The prior validation winner stays
            # fully resumable even when training resumes into another directory.
            best_checkpoint=copy.deepcopy({key:value for key,value in payload.items() if key!='best_state_dict'})
            _save(output/'best.pt',best_checkpoint)
        elif not (output/'best.pt').exists():
            _save(output/'best.pt',best_checkpoint)
        payload['best_checkpoint']=best_checkpoint
        _save(output/'last.pt',payload)
        (output/'history.json').write_text(json.dumps({'epochs':history,'split_report':separation},indent=2,allow_nan=False)+'\n')
        print(json.dumps(history[-1],allow_nan=False),flush=True)
    return history
