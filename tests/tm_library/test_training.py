import json
from pathlib import Path
import numpy as np
import pytest
import torch

from tm_library.config import ALGORITHMS, ModelConfig
from tm_library.model import TMModel
from tm_library.losses import CompositeLoss, LossConfig, masked_semantic_bce
from tm_library.metrics import image_metrics, aggregate
from tm_library.engine import train, load_checkpoint
from tm_library.evaluate import evaluate
from tm_library.infer import infer


torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'photofinishing/models/photofinishing_s24-style-0.pth'


def fixture(tmp_path):
    paths=[]
    for split in ('train','val'):
        rows=[]
        for idx in range(2):
            rng=np.random.default_rng(idx + (10 if split=='val' else 0))
            image=rng.uniform(.04,.8,(24,25,3)).astype('float32')
            name=f'{split}_{idx}'
            np.save(tmp_path/f'{name}.npy',image)
            np.save(tmp_path/f'{name}_target.npy',np.power(image,1/2.2).astype('float32'))
            rows.append({'id':name,'input':f'{name}.npy','target':f'{name}_target.npy',
                         'input_encoding':'linear_srgb','target_aligned':True,'scene':split,'camera':'synthetic'})
        path=tmp_path/f'{split}.json'
        path.write_text(json.dumps(rows))
        paths.append(path)
    return paths


def configuration(tmp_path, algorithm='gain_residual', epochs=1):
    train_path,val_path=fixture(tmp_path)
    return {'model':{'algorithm':algorithm,'width':4,'analysis_size':8,'grid_size':3,'grid_depth':3,'pyramid_levels':2},
            'data':{'train_manifest':str(train_path),'val_manifest':str(val_path),'image_size':24,'augment':True},
            'training':{'epochs':epochs,'batch_size':1,'learning_rate':.001,'seed':42,'cpu_threads':1},
            'upstream_checkpoint':str(UPSTREAM),'output_dir':str(tmp_path/'run')}


@pytest.mark.parametrize('algorithm',ALGORITHMS)
def test_actual_training_update(tmp_path,algorithm):
    cfg=configuration(tmp_path,algorithm)
    torch.manual_seed(42)
    before=TMModel(ModelConfig.from_dict(cfg['model'])).load_upstream(UPSTREAM)
    history=train(cfg)
    payload=load_checkpoint(tmp_path/'run/last.pt')
    changed=[not torch.equal(value,payload['state_dict'][name]) for name,value in before.state_dict().items()]
    assert any(changed)
    assert history[-1]['validation']['rgb'] >= 0
    assert (tmp_path/'run/best.pt').exists()


def test_missing_labels_have_zero_auxiliary_gradient():
    logits=torch.randn(2,3,7,9,requires_grad=True)
    loss=masked_semantic_bce(logits,torch.zeros_like(logits),torch.zeros_like(logits))
    loss.backward()
    assert loss.item()==0 and torch.count_nonzero(logits.grad)==0
    logits.grad=None
    valid=torch.ones_like(logits)
    masked_semantic_bce(logits,torch.ones_like(logits),valid).backward()
    assert logits.grad.abs().sum()>0


def test_resume_exact_cpu_and_evaluation_targetless_inference(tmp_path):
    cfg=configuration(tmp_path,epochs=2)
    train(cfg)
    expected=load_checkpoint(tmp_path/'run/last.pt')
    cfg['output_dir']=str(tmp_path/'resumed')
    cfg['training']['epochs']=1
    train(cfg)
    cfg['training']['epochs']=2
    train(cfg,resume=tmp_path/'resumed/last.pt')
    resumed=load_checkpoint(tmp_path/'resumed/last.pt')
    for key in expected['state_dict']:
        torch.testing.assert_close(expected['state_dict'][key],resumed['state_dict'][key],rtol=0,atol=0)
    assert expected['history']==resumed['history']
    report=evaluate(tmp_path/'run/best.pt',cfg['data']['val_manifest'],tmp_path/'eval')
    assert report['global']['count']==2 and report['global']['ssim'] <= 1.00001
    rows=json.loads(Path(cfg['data']['val_manifest']).read_text())
    for row in rows:
        row.pop('target'); row.pop('target_aligned')
    manifest=tmp_path/'targetless.json'; manifest.write_text(json.dumps(rows))
    metadata=infer(tmp_path/'run/best.pt',manifest,tmp_path/'infer',save_maps=True)
    assert len(metadata['images'])==2
    assert len(list((tmp_path/'infer').glob('*.png')))==2
    assert (tmp_path/'infer/metadata.json').exists()


def test_split_leakage_rejected(tmp_path):
    cfg=configuration(tmp_path)
    cfg['data']['val_manifest']=cfg['data']['train_manifest']
    with pytest.raises(ValueError,match='overlap'):
        train(cfg)


def test_metrics_regions_and_perfect_image():
    image=torch.ones(1,3,2,3)*.4
    metrics=image_metrics(image,image,torch.ones_like(image),torch.ones_like(image))
    assert metrics['psnr']==120 and metrics['ssim']==pytest.approx(1)
    report=aggregate([{'id':'x','scene':'s','camera':'c','metrics':metrics}])
    assert report['by_region']['skin']['count']==1


def test_require_image_validation_objective(tmp_path):
    cfg=configuration(tmp_path)
    cfg['model']['semantic_mode']='train_only'
    cfg['loss']={'rgb':0,'log_luma':0,'gradient':0,'region':0,'semantic':1}
    with pytest.raises(ValueError,match='image loss'):
        train(cfg)


def test_mixed_size_batch_rejected_before_updates(tmp_path):
    cfg=configuration(tmp_path)
    cfg['training']['batch_size']=2
    cfg['data'].pop('image_size')
    np.save(tmp_path/'train_1.npy',np.full((22,25,3),.2,dtype='float32'))
    np.save(tmp_path/'train_1_target.npy',np.full((22,25,3),.3,dtype='float32'))
    with pytest.raises(ValueError,match='equal input shapes'):
        train(cfg)
    assert not Path(cfg['output_dir']).exists()


@pytest.mark.parametrize('mode',['train_only','explicit'])
def test_semantic_training_modes_with_and_without_annotations(tmp_path,mode):
    cfg=configuration(tmp_path)
    cfg['model']['semantic_mode']=mode
    rows=json.loads(Path(cfg['data']['train_manifest']).read_text())
    masks=np.zeros((24,25,3),dtype='float32'); masks[:12,:,0]=1; masks[12:,:,2]=1
    np.save(tmp_path/'masks.npy',masks)
    rows[0]['semantics']='masks.npy'
    Path(cfg['data']['train_manifest']).write_text(json.dumps(rows))
    train(cfg)
    payload=load_checkpoint(tmp_path/'run/last.pt')
    assert payload['config']['semantic_mode']==mode
    if mode=='train_only':
        assert payload['history'][0]['train']['semantic']>0
        assert payload['history'][0]['validation']['semantic']==0


def test_short_full_model_loss_reduction_probe():
    """Fixed aligned target generated by a teacher C-model; optimize rendered output."""
    torch.manual_seed(29)
    config=ModelConfig(algorithm='gain_residual',width=4,analysis_size=8,freeze_backbone=True)
    model=TMModel(config).load_upstream(UPSTREAM)
    # A fixed teacher exposure creates an achievable target and isolates operator learning.
    teacher=TMModel(config).load_upstream(UPSTREAM)
    teacher.load_state_dict(model.state_dict())
    # Locate the EV prediction head through its zero-initialized output bias.
    heads=[layer for layer in teacher.operator.modules() if isinstance(layer,torch.nn.Conv2d) and layer.out_channels==1]
    with torch.no_grad():
        heads[-1].bias.fill_(.25)
    image=torch.rand(1,3,24,25)*.7+.05
    teacher.eval()
    with torch.no_grad():
        target=teacher(image)['output']
    optimizer=torch.optim.AdamW(model.operator.parameters(),lr=.005,weight_decay=0)
    losses=[]
    for _ in range(12):
        optimizer.zero_grad()
        loss=(model(image)['output']-target).abs().mean()
        loss.backward(); optimizer.step(); losses.append(loss.item())
    assert losses[-1] < losses[0]*.6


def test_region_only_objective_with_missing_labels_is_rejected(tmp_path):
    cfg=configuration(tmp_path)
    cfg['loss']={'rgb':0,'log_luma':0,'gradient':0,'region':1,'semantic':0}
    with pytest.raises(ValueError,match='ungated image loss'):
        train(cfg)


def test_equal_rectangular_augmentation_batches_train(tmp_path):
    cfg=configuration(tmp_path)
    cfg['training'].update(batch_size=2,seed=9)
    cfg['data'].pop('image_size')
    history=train(cfg)
    assert history[0]['train']['rgb']>0


def test_relocated_best_retains_complete_resume_state(tmp_path):
    cfg=configuration(tmp_path)
    train(cfg)
    last_path=tmp_path/'run/last.pt'
    last=load_checkpoint(last_path)
    # Force a non-improving continuation to exercise prior-winner preservation.
    last['best_validation_loss']=0.
    last['best_checkpoint']['best_validation_loss']=0.
    torch.save(last,last_path)
    cfg['output_dir']=str(tmp_path/'new_run')
    cfg['training']['epochs']=2
    train(cfg,resume=last_path)
    best=load_checkpoint(tmp_path/'new_run/best.pt')
    assert {'optimizer','scheduler','rng','history','training_signature'} <= set(best)
    assert best['epoch']==1
    cfg['output_dir']=str(tmp_path/'from_best')
    train(cfg,resume=tmp_path/'new_run/best.pt')
    assert (tmp_path/'from_best/last.pt').exists()
