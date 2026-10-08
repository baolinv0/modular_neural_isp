"""Paired IO invariants; removing validation or changing conversion breaks these."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def manifest(tmp_path, rows):
    p = tmp_path / 'pairs.jsonl'
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return p


def pair(tmp_path, **extra):
    x = np.full((5, 7, 3), .25, np.float32)
    np.save(tmp_path / 'in.npy', x)
    np.save(tmp_path / 'gt.npy', x)
    return dict(id='a', input='in.npy', target='gt.npy', target_aligned=True,
                input_encoding='linear_srgb', **extra)


def dataset(*args, **kwargs):
    from tm_library.data import PairedImageDataset
    return PairedImageDataset(*args, **kwargs)


def test_missing_labels_not_supervised_background(tmp_path):
    sample = dataset(manifest(tmp_path, [pair(tmp_path)]))[0]
    assert sample['input'].shape == (3, 5, 7)
    assert sample['semantic_valid'].shape == (3, 5, 7)
    assert torch.count_nonzero(sample['semantic_valid']) == 0
    assert torch.count_nonzero(sample['confidence']) == 0
    assert set(sample) == {'input','target','semantics','confidence','semantic_valid','id','scene','camera',
                           'burst_id','subject_id','scenario'}


def test_srgb_decoding_and_camera_green_normalized_wb(tmp_path):
    row = pair(tmp_path)
    row['input_encoding'] = 'srgb'
    srgb = dataset(manifest(tmp_path, [row]))[0]['input']
    torch.testing.assert_close(srgb, torch.full_like(srgb, ((.25 + .055) / 1.055) ** 2.4))
    row['input_encoding'] = 'camera_rgb'
    row['metadata'] = {'cam_illum':[2., 4., 8.], 'ccm':[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}
    x = dataset(manifest(tmp_path, [row]))[0]['input']
    torch.testing.assert_close(x[:,0,0], torch.tensor([.5,.25,.125]))


def test_png16_preserves_precision_and_rgb_channel_order(tmp_path):
    import cv2
    data = np.zeros((5,7,3),np.uint16)
    data[:] = [12345,23456,45678]
    cv2.imwrite(str(tmp_path/'in.png'), data[:,:,::-1])
    row=pair(tmp_path)
    row['input']='in.png'
    x=dataset(manifest(tmp_path,[row]))[0]['input']
    torch.testing.assert_close(x[:,0,0], torch.tensor([12345,23456,45678],dtype=torch.float32)/65535)


def test_geometry_synchronized_and_labelled_background_valid(tmp_path):
    y=np.linspace(0,1,35,dtype=np.float32).reshape(5,7)
    x=np.repeat(y[:,:,None],3,axis=-1)
    np.save(tmp_path/'in.npy',x)
    np.save(tmp_path/'gt.npy',x)
    np.save(tmp_path/'sem.npy',x)
    row={'id':'same','input':'in.npy','target':'gt.npy','target_aligned':True,
         'input_encoding':'linear_srgb','semantics':'sem.npy'}
    sample=dataset(manifest(tmp_path,[row]),image_size=8,augment=True)[0]
    torch.testing.assert_close(sample['input'],sample['target'])
    torch.testing.assert_close(sample['input'],sample['semantics'])
    assert sample['semantic_valid'].min()==1
    assert sample['confidence'].min()==1


@pytest.mark.parametrize('change,match',[
    ({'target_aligned':False},'aligned'),
    ({'input_encoding':'bayer'},'encoding'),
    ({'input':'missing.npy'},'missing'),
    ({'target':None},'target'),
    ({'reference':'gt.npy','target':None},'target'),
])
def test_invalid_manifest_rejected(tmp_path,change,match):
    row=pair(tmp_path);row.update(change)
    with pytest.raises((ValueError,FileNotFoundError),match=match):
        dataset(manifest(tmp_path,[row]))[0]


def test_duplicate_ids_rejected(tmp_path):
    row=pair(tmp_path)
    with pytest.raises(ValueError,match='duplicate'):
        dataset(manifest(tmp_path,[row,row]))


@pytest.mark.parametrize('values,match',[(np.nan,'finite'),(1.01,'range'),(-.1,'range')])
def test_bad_pixel_values_rejected(tmp_path,values,match):
    row=pair(tmp_path)
    np.save(tmp_path/'in.npy',np.full((5,7,3),values,np.float32))
    with pytest.raises(ValueError,match=match):
        dataset(manifest(tmp_path,[row]))[0]


def test_ambiguous_array_requires_layout_and_npz_key(tmp_path):
    row=pair(tmp_path)
    x=np.full((3,5,3),.2,np.float32)
    np.save(tmp_path/'in.npy',x)
    np.save(tmp_path/'gt.npy',x)
    with pytest.raises(ValueError,match='layout'):
        dataset(manifest(tmp_path,[row]))[0]
    row.update(input_layout='HWC',target_layout='HWC')
    np.savez(tmp_path/'in.npz',image=x)
    row['input']='in.npz'
    assert dataset(manifest(tmp_path,[row]))[0]['input'].shape==(3,3,5)


def test_shape_mismatch_rejected_and_inference_has_no_reference_leak(tmp_path):
    row=pair(tmp_path)
    np.save(tmp_path/'gt.npy',np.full((6,7,3),.9,np.float32))
    with pytest.raises(ValueError,match='shape'):
        dataset(manifest(tmp_path,[row]))[0]
    row.pop('target');row['reference']='gt.npy'
    sample=dataset(manifest(tmp_path,[row]),require_target=False)[0]
    assert torch.count_nonzero(sample['target'])==0
    assert sample['target'].shape==sample['input'].shape


def test_confidence_and_semantic_valid_masks(tmp_path):
    row=pair(tmp_path,semantics='sem.npy',confidence='conf.npy',semantic_valid='valid.npy')
    np.save(tmp_path/'sem.npy',np.ones((5,7,3),np.float32))
    np.save(tmp_path/'conf.npy',np.full((5,7),.4,np.float32))
    np.save(tmp_path/'valid.npy',np.zeros((5,7),np.float32))
    sample=dataset(manifest(tmp_path,[row]))[0]
    assert sample['semantic_valid'].max()==0
    assert sample['confidence'].shape==(1,5,7)
    assert sample['confidence'].mean().item()==pytest.approx(.4)


def test_synthetic_deterministic_distinct_splits_and_known_target(tmp_path):
    from tm_library.synthetic import generate
    a=generate(tmp_path/'a',num_train=2,num_val=1,size=11,seed=4)
    b=generate(tmp_path/'b',num_train=2,num_val=1,size=11,seed=4)
    train=dataset(a['train']);val=dataset(a['val'])
    assert len(train)==2 and len(val)==1
    assert train[0]['id']!=val[0]['id']
    torch.testing.assert_close(train[0]['input'],dataset(b['train'])[0]['input'])
    assert train[0]['semantics'][1].max()>0
    assert not torch.equal(train[0]['input'],train[0]['target'])


def test_prepare_pairs_by_exact_stem_rejects_unmatched(tmp_path):
    from tm_library.prepare_data import build_manifest
    i=tmp_path/'inputs';t=tmp_path/'targets';i.mkdir();t.mkdir()
    np.save(i/'b.npy',np.full((5,7,3),.1,np.float32))
    np.save(t/'a.npy',np.full((5,7,3),.9,np.float32))
    with pytest.raises(ValueError,match='unmatched'):
        build_manifest(i,t,tmp_path/'out.jsonl',input_encoding='linear_srgb',target_aligned=True)
    np.save(t/'b.npy',np.full((5,7,3),.8,np.float32))
    (t/'a.npy').unlink()
    out=build_manifest(i,t,tmp_path/'out.jsonl',input_encoding='linear_srgb',target_aligned=True)
    sample=dataset(out)[0]
    assert sample['id']=='b' and sample['target'].mean().item()==pytest.approx(.8)


def test_float_tiff_chw_and_json_manifest(tmp_path):
    import tifffile
    row=pair(tmp_path)
    x=np.full((5,7,3),.123456,np.float32)
    tifffile.imwrite(tmp_path/'in.tiff',x,photometric='rgb')
    np.save(tmp_path/'gt.npy',x.transpose(2,0,1))
    row['input']='in.tiff'
    p=tmp_path/'list.json';p.write_text(json.dumps([row]))
    sample=dataset(p)[0]
    torch.testing.assert_close(sample['input'],sample['target'])
    assert sample['input'][0,0,0].item()==pytest.approx(.123456)


def test_target_encoding_cannot_silently_be_linear(tmp_path):
    row=pair(tmp_path,target_encoding='linear_srgb')
    with pytest.raises(ValueError,match='target_encoding'):
        dataset(manifest(tmp_path,[row]))[0]


def test_invalid_orphan_confidence_not_silently_ignored(tmp_path):
    row=pair(tmp_path,confidence='conf.npy')
    np.save(tmp_path/'conf.npy',np.full((5,7),np.nan,np.float32))
    with pytest.raises(ValueError,match='finite'):
        dataset(manifest(tmp_path,[row]))[0]


@pytest.mark.parametrize('field,shape', [('input',(5,7,4)),('semantics',(5,7,2)),('confidence',(5,7,3))])
def test_wrong_channel_count_rejected(tmp_path,field,shape):
    row=pair(tmp_path,semantics='sem.npy')
    np.save(tmp_path/'sem.npy',np.ones((5,7,3),np.float32))
    np.save(tmp_path/f'{field}_wrong.npy',np.full(shape,.5,np.float32))
    row[field]=f'{field}_wrong.npy'
    with pytest.raises(ValueError,match='channels'):
        dataset(manifest(tmp_path,[row]))[0]


def test_camera_metadata_file_matches_numpy_upstream_formula(tmp_path):
    row=pair(tmp_path)
    row['input_encoding']='camera_rgb'
    metadata={'cam_illum':[.3,.6,.5], 'ccm':[[1.2,-.1,-.1],[-.1,1.1,0.],[0.,-.2,1.2]]}
    (tmp_path/'meta.json').write_text(json.dumps(metadata))
    row['metadata']='meta.json'
    result=dataset(manifest(tmp_path,[row]))[0]['input']
    illum=np.asarray(metadata['cam_illum'],np.float32)
    ccm=np.asarray(metadata['ccm'],np.float32)
    expected=np.clip(np.full((35,3),.25,np.float32) @ (ccm @ np.diag(illum[1]/illum)).T,0,1)
    torch.testing.assert_close(result[:,0,0],torch.from_numpy(expected[0]))


def test_npz_requires_documented_image_key(tmp_path):
    row=pair(tmp_path)
    np.savez(tmp_path/'bad.npz',foo=np.ones((5,7,3),np.float32))
    row['input']='bad.npz'
    with pytest.raises(ValueError,match='key image'):
        dataset(manifest(tmp_path,[row]))[0]


def test_prepare_json_destination_and_failed_validation_preserves_previous(tmp_path):
    from tm_library.prepare_data import build_manifest
    i=tmp_path/'inputs';t=tmp_path/'targets';i.mkdir();t.mkdir()
    np.save(i/'a.npy',np.full((5,7,3),.1,np.float32))
    np.save(t/'a.npy',np.full((5,7,3),.9,np.float32))
    out=build_manifest(i,t,tmp_path/'out.json',input_encoding='linear_srgb',target_aligned=True)
    assert dataset(out)[0]['id']=='a'
    previous=out.read_text()
    np.save(i/'a.npy',np.full((5,7,3),np.inf,np.float32))
    with pytest.raises(ValueError,match='finite'):
        build_manifest(i,t,out,input_encoding='linear_srgb',target_aligned=True)
    assert out.read_text()==previous


def test_prepare_python_api_requires_alignment_assertion(tmp_path):
    from tm_library.prepare_data import build_manifest
    i=tmp_path/'inputs';t=tmp_path/'targets';i.mkdir();t.mkdir()
    np.save(i/'a.npy',np.full((5,7,3),.1,np.float32))
    np.save(t/'a.npy',np.full((5,7,3),.9,np.float32))
    with pytest.raises(ValueError,match='aligned target assertion'):
        build_manifest(i,t,tmp_path/'out.jsonl',input_encoding='linear_srgb')


def test_rectangular_augmentation_preserves_batch_shapes_and_synchronized_geometry(tmp_path):
    from torch.utils.data import DataLoader
    values=np.linspace(0,1,21*25,dtype=np.float32).reshape(21,25)
    rgb=np.repeat(values[...,None],3,axis=-1)
    for filename in ('in.npy','gt.npy','sem.npy'):
        np.save(tmp_path/filename,rgb)
    np.save(tmp_path/'confidence.npy',values)
    np.save(tmp_path/'valid.npy',values)
    rows=[{'id':str(index),'input':'in.npy','target':'gt.npy','target_aligned':True,
           'input_encoding':'linear_srgb','semantics':'sem.npy','confidence':'confidence.npy',
           'semantic_valid':'valid.npy'} for index in range(8)]
    samples=dataset(manifest(tmp_path,rows),augment=True)
    for seed in (6,9,17):
        torch.manual_seed(seed)
        for batch in DataLoader(samples,batch_size=2):
            assert batch['input'].shape==(2,3,21,25)
            torch.testing.assert_close(batch['input'],batch['target'])
            torch.testing.assert_close(batch['input'],batch['semantics'])
            torch.testing.assert_close(batch['input'][:,:1],batch['confidence'])
            torch.testing.assert_close(batch['input'],batch['semantic_valid'])


def test_group_metadata_propagates_and_builder_records_it(tmp_path):
    from tm_library.prepare_data import build_manifest
    fields = dict(scene='living_room', burst_id='burst_3', subject_id='person_2', scenario='side_light')
    sample = dataset(manifest(tmp_path, [pair(tmp_path, **fields)]))[0]
    assert {key:sample[key] for key in fields} == fields
    inputs = tmp_path/'inputs'; targets = tmp_path/'targets'
    inputs.mkdir(); targets.mkdir()
    np.save(inputs/'a.npy', np.full((5,7,3), .2, np.float32))
    np.save(targets/'a.npy', np.full((5,7,3), .4, np.float32))
    path = build_manifest(inputs, targets, tmp_path/'prepared.jsonl',
                          input_encoding='linear_srgb', target_aligned=True, **fields)
    assert {key:dataset(path)[0][key] for key in fields} == fields


def test_semantic_fixture_targets_depend_on_role_and_illumination_and_protect_intent(tmp_path):
    from tm_library.synthetic import generate, semantic_target, linear_to_srgb, SEMANTIC_SCENARIOS
    paths = generate(tmp_path/'semantic', num_train=6, num_val=6, size=32, seed=7, task='semantic')
    train, val = dataset(paths['train']), dataset(paths['val'])
    assert [row['scenario'] for row in train.records] == list(SEMANTIC_SCENARIOS)
    assert {row['scene'] for row in train.records}.isdisjoint({row['scene'] for row in val.records})
    for index, row in enumerate(train.records):
        sample = train[index]
        image = sample['input'].permute(1,2,0).numpy()
        masks = sample['semantics'].permute(1,2,0).numpy()
        reference = semantic_target(image, masks, row['scenario'])
        np.testing.assert_allclose(reference, sample['target'].permute(1,2,0).numpy())
        no_labels = semantic_target(image, np.zeros_like(masks), row['scenario'])
        if row['scenario'] in ('portrait_backlight','two_people_unequal','side_light'):
            assert np.max(np.abs(reference-no_labels)) > .01
            # A scalar exposure changes brightness, preserving RGB chromaticity.
            from tm_library.data import srgb_to_linear
            restored = srgb_to_linear(reference)
            ratios = restored / np.maximum(image, 1e-6)
            np.testing.assert_allclose(ratios[...,0], ratios[...,1], atol=2e-5)
            np.testing.assert_allclose(ratios[...,1], ratios[...,2], atol=2e-5)
        else:
            np.testing.assert_allclose(reference, linear_to_srgb(image), atol=1e-7)
    repeat = generate(tmp_path/'repeat', num_train=1, num_val=6, size=32, seed=7, task='semantic')
    torch.testing.assert_close(val[2]['input'], dataset(repeat['val'])[2]['input'])


def test_semantic_target_has_bounded_illumination_gain_without_skin_color_target():
    from tm_library.data import srgb_to_linear
    from tm_library.synthetic import semantic_target
    # Two subject roles with different illumination and different skin RGB.
    image = np.array([[[.03,.02,.01], [.4,.25,.15], [.04,.06,.025]]],np.float32)
    masks = np.ones_like(image); masks[...,2] = 0
    target = srgb_to_linear(semantic_target(image,masks,'two_people_unequal'))
    gain = target/image
    assert gain.min() >= 1-1e-6 and gain.max() <= 1.5+1e-6
    assert gain[0,0,0] > gain[0,1,0]
    np.testing.assert_allclose(gain[...,0],gain[...,1],atol=2e-6)
    assert not np.allclose(target[0,0],target[0,2])


@pytest.mark.parametrize('field', ['scene','burst_id','subject_id','scenario'])
def test_empty_group_metadata_is_rejected(tmp_path,field):
    with pytest.raises(ValueError,match=field):
        dataset(manifest(tmp_path,[pair(tmp_path,**{field:''})]))
