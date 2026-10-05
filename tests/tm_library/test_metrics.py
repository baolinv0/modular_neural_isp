import math

import pytest
import torch

from tm_library.metrics import aggregate, image_metrics, srgb_to_lab


def test_d65_lab_known_neutral_and_primary_values():
    values = torch.tensor([0.,0.,0., 1.,1.,1., 1.,0.,0.]).reshape(1,3,3).transpose(1,2)[...,None]
    lab = srgb_to_lab(values)[0,:,:,0]
    torch.testing.assert_close(lab[:,0], torch.zeros(3), atol=1e-4, rtol=0)
    torch.testing.assert_close(lab[:,1], torch.tensor([100.,0.,0.]), atol=.02, rtol=0)
    torch.testing.assert_close(lab[:,2], torch.tensor([53.2408,80.0925,67.2032]), atol=.02, rtol=0)


def test_perfect_and_perturbed_color_metrics_and_raw_linear_diagnostics():
    target = torch.tensor([.8,.3,.2])[None,:,None,None].expand(1,3,5,7)
    linear = torch.where(target <= .04045, target/12.92, ((target+.055)/1.055)**2.4)
    perfect = image_metrics(target, target, linear=linear, preclip=target)
    assert perfect['delta_e76'] == pytest.approx(0)
    assert perfect['chroma_l1'] == pytest.approx(0)
    assert perfect['hue_degrees'] == pytest.approx(0)
    assert perfect['linear_l1'] == pytest.approx(0)
    perturbed = image_metrics(target.roll(1,1), target, linear=linear+.1, preclip=target+1)
    for key in ('delta_e76','chroma_l1','hue_degrees','linear_l1','linear_log_l1'):
        assert perturbed[key] > 0
    assert perturbed['preclip_above_one_fraction'] == 1
    assert perturbed['preclip_above_one_excess'] > 0


def test_black_and_empty_regions_have_no_nan_or_fabricated_hue():
    black = torch.zeros(1,3,1,1)
    result = image_metrics(black, black, black, torch.ones_like(black), linear=black)
    assert result['regions'] == {}
    assert result['hue_valid_fraction'] == 0
    assert 'hue_degrees' not in result
    assert 'semantic_boundary_luma_gradient_l1' not in result
    assert all(math.isfinite(value) for key,value in result.items() if key != 'regions')
    report = aggregate([dict(id='black', scene='s', camera='c', scenario='night', metrics=result)], include_per_image=True)
    assert report['by_scenario']['night']['count'] == 1
    assert report['per_image'][0]['id'] == 'black'
    white = image_metrics(torch.ones_like(black),torch.ones_like(black))
    assert white['hue_valid_fraction'] == 0
    assert 'hue_degrees' not in white


def test_target_referenced_semantic_boundary_gradient_error():
    target = torch.zeros(1,3,4,6); target[:,:,:,3:] = .8
    masks = torch.zeros_like(target); masks[:,0,:,:3] = 1
    valid = torch.ones_like(masks)
    perfect = image_metrics(target, target, masks, valid)
    changed = target.clone(); changed[:,:,:,3:] = .6
    perturbed = image_metrics(changed, target, masks, valid)
    assert perfect['semantic_boundary_luma_gradient_l1'] == pytest.approx(0)
    assert perturbed['semantic_boundary_luma_gradient_l1'] == pytest.approx(.2, abs=1e-6)
    assert perfect['regions']['person']['delta_e76'] == pytest.approx(0)
    hidden = image_metrics(changed,target,masks,torch.zeros_like(valid))
    assert 'semantic_boundary_luma_gradient_l1' not in hidden


def test_aggregate_excludes_missing_optional_metrics_and_regions():
    colored = torch.tensor([.9,.1,.2])[None,:,None,None]
    a = image_metrics(colored, colored)
    b = image_metrics(torch.zeros_like(colored), torch.zeros_like(colored))
    result = aggregate([dict(scene='s',camera='c',metrics=a), dict(scene='s',camera='c',metrics=b)], group_scenario=False)
    assert result['global']['hue_degrees'] == 0
    assert result['global']['count'] == 2
    assert 'by_scenario' not in result
