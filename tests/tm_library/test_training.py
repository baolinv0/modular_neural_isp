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
    cfg['training']['epochs']=2
    train(cfg,resume=tmp_path/'run/last.pt')
    assert evaluate(tmp_path/'run/best.pt',cfg['data']['val_manifest'],tmp_path/'eval')['global']['count']==2
    assert len(infer(tmp_path/'run/best.pt',cfg['data']['val_manifest'],tmp_path/'infer')['images'])==2


def test_missing_labels_have_zero_auxiliary_gradient():
    logits=torch.randn(2,3,7,9,requires_grad=True)
    loss=masked_semantic_bce(logits,torch.zeros_like(logits),torch.zeros_like(logits))
    loss.backward()
    assert loss.item()==0 and torch.count_nonzero(logits.grad)==0
    logits.grad=None
    valid=torch.ones_like(logits)
    masked_semantic_bce(logits,torch.ones_like(logits),valid).backward()
    assert logits.grad.abs().sum()>0


def assert_checkpoint_values_equal(left,right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            assert_checkpoint_values_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):
            assert_checkpoint_values_equal(a,b)
    else:
        assert left==right


@pytest.mark.parametrize('mode',['none','train_only','explicit'])
def test_resume_exact_cpu_and_evaluation_targetless_inference(tmp_path,mode):
    cfg=configuration(tmp_path,epochs=2)
    cfg['model']['semantic_mode']=mode
    if mode!='none':
        cfg['segmentation']=segmenter_config(tmp_path)
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
    for key in ('optimizer','scheduler','rng','segmentation','best_checkpoint'):
        assert_checkpoint_values_equal(expected[key],resumed[key])
    assert 'best_checkpoint' not in resumed['best_checkpoint']
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


def test_v1_library_checkpoint_rejected_actionably(tmp_path):
    path=tmp_path/'legacy.pt'
    torch.save({'format':'tm_library_v1','config':{},'state_dict':{}},path)
    with pytest.raises(ValueError,match='retrain|migration'):
        load_checkpoint(path)


def test_strict_scene_split_and_explicit_legacy_mode(tmp_path):
    cfg=configuration(tmp_path)
    rows=json.loads(Path(cfg['data']['val_manifest']).read_text())
    for row in rows:
        row['scene']='train'
    Path(cfg['data']['val_manifest']).write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='scene.*overlap'):
        train(cfg)
    cfg['data']['split_policy']='id_only'
    train(cfg)


@pytest.mark.parametrize('base,correction',[('baseline','gain_residual'),('spatial_grid','gain_residual'),('exposure_fusion','region_curves'),('base_detail','gain_residual')])
def test_composed_operator_optimizer_update(tmp_path,base,correction):
    cfg=configuration(tmp_path)
    cfg['model'].update(algorithm='baseline',base_algorithm=base,correction_algorithm=correction,freeze_backbone=True)
    torch.manual_seed(42)
    before=TMModel(ModelConfig.from_dict(cfg['model'])).load_upstream(UPSTREAM)
    train(cfg)
    after=load_checkpoint(tmp_path/'run/last.pt')['state_dict']
    for prefix in ('operator.','correction_operator.'):
        names=[name for name,value in before.named_parameters() if name.startswith(prefix) and value.requires_grad]
        if names:
            assert any(not torch.equal(before.state_dict()[name],after[name]) for name in names),prefix
    cfg['training']['epochs']=2
    train(cfg,resume=tmp_path/'run/last.pt')
    assert evaluate(tmp_path/'run/best.pt',cfg['data']['val_manifest'],tmp_path/'eval')['global']['count']==2
    assert len(infer(tmp_path/'run/best.pt',cfg['data']['val_manifest'],tmp_path/'infer')['images'])==2


class TinyLoadedSegmenter(torch.nn.Module):
    def forward(self,image):
        return image*2-0.5


def segmenter_config(tmp_path):
    path=tmp_path/'segmenter.pt'
    torch.jit.trace(TinyLoadedSegmenter(),torch.ones(1,3,8,9)).save(str(path))
    return {'backend':'torchscript','weights':str(path),'source':'model','input_size':[8,9],
            'input_encoding':'linear_srgb','output_mode':'sigmoid_logits','class_groups':[[0],[1],[2]]}


@pytest.mark.parametrize('mode',['train_only','explicit'])
def test_real_segmenter_training_and_deployment_modes(tmp_path,mode):
    cfg=configuration(tmp_path)
    cfg['model']['semantic_mode']=mode
    cfg['segmentation']=segmenter_config(tmp_path)
    train(cfg)
    checkpoint=tmp_path/'run/last.pt'
    payload=load_checkpoint(checkpoint)
    assert payload['segmentation']==payload['training_signature']['segmentation']
    if mode=='train_only':
        assert payload['history'][0]['train']['semantic']>0
        assert payload['history'][0]['validation']['semantic']==0
        Path(cfg['segmentation']['weights']).unlink()
        infer(checkpoint,cfg['data']['val_manifest'],tmp_path/'infer')
    else:
        Path(cfg['segmentation']['weights']).unlink()
        with pytest.raises((ValueError,FileNotFoundError,RuntimeError),match='segmenter|weights|checkpoint|exist|file'):
            infer(checkpoint,cfg['data']['val_manifest'],tmp_path/'missing')
        override=segmenter_config(tmp_path)
        infer(checkpoint,cfg['data']['val_manifest'],tmp_path/'override',semantic_checkpoint=override['weights'])
        report=evaluate(checkpoint,cfg['data']['val_manifest'],tmp_path/'metrics',semantic_config=override)
        assert report['latency']['segmentation_plus_tm']['mean_ms']>=0
        assert report['latency']['tm_only']['mean_ms']>=0
    cfg['training']['epochs']=2
    cfg['segmentation']['source']='prefer_manifest'
    with pytest.raises(ValueError,match='configuration differs'):
        train(cfg,resume=checkpoint)


def test_matched_semantic_loss_configs():
    from tm_library.engine import load_config,validate_config
    weights=[]
    for filename in ('gain_residual.yaml','gain_residual_train_only.yaml','gain_residual_explicit.yaml'):
        config=load_config(ROOT/'configs/tm_library'/filename)
        _,_,loss,_=validate_config(config)
        weights.append((loss.rgb,loss.log_luma,loss.gradient,loss.region))
    assert len(set(weights))==1 and weights[0][-1]==0


def test_diagnostic_protocol_requires_actual_fixed_controls(tmp_path):
    cfg=configuration(tmp_path)
    cfg['training']['protocol']='operator_diagnostic'
    with pytest.raises(ValueError,match='fixed_reference'):
        train(cfg)
    cfg['model'].update(freeze_backbone=True,post_mode='fixed_reference')
    train(cfg)


def test_s0_never_loads_configured_external_segmenter(tmp_path):
    cfg=configuration(tmp_path)
    cfg['segmentation']=segmenter_config(tmp_path)
    Path(cfg['segmentation']['weights']).unlink()
    train(cfg)
    report=evaluate(tmp_path/'run/last.pt',cfg['data']['val_manifest'],tmp_path/'eval')
    assert not report['latency']['external_segmenter_active']
    infer(tmp_path/'run/last.pt',cfg['data']['val_manifest'],tmp_path/'infer')


def test_config_relative_segmenter_and_group_overlap(tmp_path):
    from tm_library.engine import load_config
    import yaml
    cfg=configuration(tmp_path)
    cfg['segmentation']={'backend':'torchscript','weights':'weights/seg.ts','source':'model'}
    path=tmp_path/'experiment.yaml'; path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)['segmentation']['weights']==str(tmp_path/'weights/seg.ts')
    cfg.pop('segmentation')
    for manifest in (cfg['data']['train_manifest'],cfg['data']['val_manifest']):
        rows=json.loads(Path(manifest).read_text())
        for row in rows:
            row['burst_id']='shared'
        Path(manifest).write_text(json.dumps(rows))
    cfg['data']['group_checks']=['burst_id']
    with pytest.raises(ValueError,match='burst_id overlap'):
        train(cfg)


@pytest.mark.parametrize('trust',[0.,0.001,0.1,1.])
def test_pseudo_label_trust_attenuates_auxiliary_loss_and_gradient(trust):
    logits=torch.zeros(1,3,32,32,requires_grad=True)
    output=torch.full_like(logits,.5)
    labels=torch.ones_like(logits)
    valid=torch.ones_like(logits)
    batch={'target':output,'semantics':labels,'semantic_valid':valid,
           'confidence':torch.ones(1,1,32,32),'semantic_supervision_weight':valid*trust}
    loss,terms=CompositeLoss(LossConfig(rgb=0,log_luma=0,gradient=0,region=0,semantic=1))(
        {'output':output,'semantic_logits':logits},batch)
    loss.backward()
    assert terms['semantic'].item()==pytest.approx(float(np.log(2))*trust)
    assert logits.grad.abs().sum().item()==pytest.approx(.5*trust)


def test_auxiliary_partial_manual_validity_and_pseudo_trust_are_distinct():
    logits=torch.zeros(1,3,4,4,requires_grad=True)
    valid=torch.ones_like(logits)
    valid[...,:2]=.5
    valid[:,2]=0
    weight=valid.clone()
    weight[...,2:]*=.1
    loss=masked_semantic_bce(logits,torch.ones_like(logits),valid,weight)
    loss.backward()
    assert loss.item()==pytest.approx(float(np.log(2))*.4)
    assert logits.grad.abs().sum().item()==pytest.approx(.2)
    assert torch.count_nonzero(logits.grad[:,2])==0
