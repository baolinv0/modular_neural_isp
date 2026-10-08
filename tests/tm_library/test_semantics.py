"""Locally loaded, frozen segmentation and annotation-source regressions."""
import pytest
import torch
from torch import nn
from tm_library.semantics import SegmentationConfig, SemanticProvider, apply_semantic_provider


class TinySegmenter(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 1)

    def forward(self, image):
        return self.conv(image)


def _fixture(tmp_path, **overrides):
    model = TinySegmenter()
    with torch.no_grad():
        model.conv.weight.zero_()
        model.conv.bias.copy_(torch.tensor([0., 1., 2., 3.]))
    path = tmp_path / 'segmenter.pt'
    torch.jit.trace(model, torch.zeros(1, 3, 3, 4)).save(str(path))
    config = dict(backend='torchscript', weights=str(path), source='model',
                  input_size=[3, 4], class_groups=[[1, 2], [2], [3]])
    config.update(overrides)
    return SemanticProvider(SegmentationConfig.from_dict(config), semantic_channels=3), path


def test_actual_torchscript_loading_mapping_freeze_and_targetless(tmp_path):
    provider, _ = _fixture(tmp_path)
    image = torch.rand(2, 3, 7, 9, requires_grad=True)
    result = provider(image)
    probabilities = torch.tensor([0., 1., 2., 3.]).softmax(0)
    assert result['semantics'].shape == (2, 3, 3, 4)
    torch.testing.assert_close(result['semantics'][0, :, 0, 0],
                               torch.tensor([probabilities[1]+probabilities[2], probabilities[2], probabilities[3]]))
    assert not result['semantics'].requires_grad
    assert not provider.model.training
    assert all(not p.requires_grad for p in provider.model.parameters())
    batch = apply_semantic_provider({'input':image}, provider, 'explicit', False)
    assert batch['semantics'].shape == (2, 3, 7, 9)
    assert batch['semantic_valid'].all()
    torch.testing.assert_close(batch['semantic_supervision_weight'], batch['confidence'].expand_as(batch['semantics']))
    assert 'target' not in batch


def test_linear_srgb_preprocessing_known_values(tmp_path):
    provider, _ = _fixture(tmp_path, input_encoding='srgb', mean=[.1, .2, .3], std=[.5, 1., 2.])
    image = torch.tensor([0., .0031308, 1.]).reshape(1,3,1,1).expand(1,3,5,7)
    prepared = provider.preprocess(image)
    expected = (torch.tensor([0., .040449936, 1.]) - torch.tensor([.1,.2,.3])) / torch.tensor([.5,1.,2.])
    torch.testing.assert_close(prepared[0,:,0,0], expected)
    assert prepared.shape == (1,3,3,4)


def test_factory_strict_nested_state_dict_loading(tmp_path):
    factory = tmp_path / 'local_segmenter.py'
    factory.write_text('from torch import nn\ndef build(channels=3):\n    return nn.Conv2d(3, channels, 1)\n')
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        model = nn.Conv2d(3, 3, 1)
        path = tmp_path / 'weights.pt'
        torch.save({'network':model.state_dict()}, path)
        config = SegmentationConfig(backend='factory', weights=str(path), source='model', factory='local_segmenter:build',
                                    factory_kwargs={'channels':3}, state_dict_key='network',
                                    output_mode='sigmoid_logits', class_groups=[[0],[1],[2]])
        provider = SemanticProvider(config)
        image = torch.rand(1,3,4,5)
        torch.testing.assert_close(provider(image)['semantics'], model(image).sigmoid())
        torch.save({'network':{'wrong':torch.zeros(1)}}, path)
        with pytest.raises(RuntimeError):
            SemanticProvider(config)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop('local_segmenter', None)


def test_prefer_manifest_partial_annotations_and_pseudo_confidence(tmp_path):
    provider, _ = _fixture(tmp_path, source='prefer_manifest')
    valid = torch.zeros(1,3,7,9)
    valid[:,0,:,:4] = 1
    manual = torch.full((1,3,7,9), .2)
    batch = {'input':torch.rand(1,3,7,9), 'semantics':manual, 'semantic_valid':valid,
             'confidence':torch.ones(1,1,7,9)}
    updated = apply_semantic_provider(batch, provider, 'train_only', True)
    assert torch.equal(updated['semantics'][:,0,:,:4], manual[:,0,:,:4])
    assert torch.equal(updated['semantic_supervision_weight'][:,0,:,:4], valid[:,0,:,:4])
    assert (updated['semantic_supervision_weight'][:,1] < 1).all()
    assert updated['semantic_valid'].all()
    assert torch.equal(batch['semantics'], manual)


def test_modes_and_manifest_source_never_call_provider(tmp_path):
    provider, path = _fixture(tmp_path, source='manifest')
    batch = {'input':torch.rand(1,3,5,7)}
    def forbidden(*args):
        raise AssertionError('provider must not execute')
    provider.__class__.__call__, original = forbidden, provider.__class__.__call__
    try:
        assert apply_semantic_provider(batch, provider, 'none', True) == batch
        assert apply_semantic_provider(batch, provider, 'train_only', False) == batch
        result = apply_semantic_provider(batch, provider, 'explicit', False)
        assert not result['semantic_valid'].any()
        assert not result['semantic_supervision_weight'].any()
    finally:
        provider.__class__.__call__ = original
    path.unlink()
    # S1 inference does not need construction or external weights.
    assert apply_semantic_provider(batch, None, 'train_only', False) == batch


@pytest.mark.parametrize('values', [dict(backend='download'), dict(input_encoding='camera_rgb'),
    dict(std=[1,0,1]), dict(input_size=[0,2]), dict(class_groups=[[0],[0],[]]),
    dict(output_key='out', output_index=0), dict(source='unknown'), dict(output_mode='labels')])
def test_invalid_config(values):
    with pytest.raises((ValueError, TypeError)):
        SegmentationConfig.from_dict(values)


@pytest.mark.parametrize('failure', ['nan', 'range', 'channels', 'rank'])
def test_invalid_model_output(tmp_path, failure):
    provider, _ = _fixture(tmp_path, output_mode='probabilities')
    class Bad(nn.Module):
        def forward(self, x):
            if failure == 'rank':
                return torch.zeros(1,3,2)
            if failure == 'channels':
                return torch.zeros(x.shape[0],1,2,2)
            value = float('nan') if failure == 'nan' else 2.
            return torch.full((x.shape[0],4,2,2), value)
    provider.model = Bad()
    with pytest.raises(ValueError):
        provider(torch.rand(1,3,3,4))


class DictSegmenter(nn.Module):
    def forward(self, image):
        masks = image[:, :2]
        return {'out':masks, 'trust':torch.ones_like(image[:, :1]) * .75}


class TupleSegmenter(nn.Module):
    def forward(self, image):
        return image[:, :2], torch.ones_like(image[:, :1]) * .6


@pytest.mark.parametrize('kind', ['dict', 'tuple'])
def test_loaded_structured_output_and_explicit_confidence(tmp_path, kind):
    path = tmp_path / f'{kind}.pt'
    network = DictSegmenter() if kind == 'dict' else TupleSegmenter()
    torch.jit.script(network).save(str(path))
    keys = dict(output_key='out', confidence_key='trust') if kind == 'dict' else dict(output_index=0, confidence_index=1)
    provider = SemanticProvider(SegmentationConfig(backend='torchscript', weights=str(path), source='model',
                                output_mode='probabilities', class_groups=[[0],[1],[0,1]], **keys))
    image = torch.full((1,3,2,3), .2, requires_grad=True)
    predicted = provider(image)
    torch.testing.assert_close(predicted['semantics'][0,:,0,0], torch.tensor([.2,.2,.4]))
    torch.testing.assert_close(predicted['confidence'], torch.full((1,1,2,3), .75 if kind == 'dict' else .6))
    assert image.grad is None and not predicted['confidence'].requires_grad


@pytest.mark.parametrize('shape,value', [((1,2,2,3), .5), ((1,1,1,3), .5), ((1,1,2,3), float('nan')), ((1,1,2,3), 1.5)])
def test_invalid_explicit_confidence(tmp_path, shape, value):
    provider, _ = _fixture(tmp_path, output_key='out', confidence_key='trust')
    class BadConfidence(nn.Module):
        def forward(self, image):
            return {'out':torch.zeros(1,4,2,3), 'trust':torch.full(shape,value)}
    provider.model = BadConfidence()
    with pytest.raises(ValueError, match='confidence'):
        provider(torch.rand(1,3,2,3))


def test_manifest_low_resolution_alignment_and_disabled_backend():
    provider = SemanticProvider(SegmentationConfig(backend='none', source='model'))
    batch = {'input':torch.rand(1,3,7,9), 'semantics':torch.full((1,3,2,4), .3),
             'confidence':torch.full((1,1,2,4), .4), 'semantic_valid':torch.ones(1,1,2,4)}
    result = apply_semantic_provider(batch, provider, 'train_only', True)
    assert result['semantics'].shape == (1,3,7,9)
    assert result['semantic_valid'].shape == (1,3,7,9)
    torch.testing.assert_close(result['semantic_supervision_weight'], torch.ones_like(result['semantics']))
    torch.testing.assert_close(result['confidence'], torch.full((1,1,7,9), .4))


def test_annotations_without_optional_confidence_or_validity():
    batch = {'input':torch.rand(1,3,7,9), 'semantics':torch.full((1,3,2,4), .3)}
    result = apply_semantic_provider(batch, None, 'explicit', False)
    assert result['confidence'].all() and result['semantic_valid'].all()


def test_prefer_manifest_complete_annotations_skip_forward(tmp_path, monkeypatch):
    provider, _ = _fixture(tmp_path, source='prefer_manifest')
    def forbidden(*args):
        raise AssertionError('complete manual annotations need no model execution')
    monkeypatch.setattr(SemanticProvider, '__call__', forbidden)
    batch = {'input':torch.rand(1,3,7,9), 'semantics':torch.rand(1,3,7,9)}
    result = apply_semantic_provider(batch, provider, 'explicit', False)
    torch.testing.assert_close(result['semantics'], batch['semantics'])


@pytest.mark.parametrize('value', [float('nan'), -.1, 1.1])
def test_invalid_input_values(tmp_path, value):
    provider, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match='input'):
        provider(torch.full((1,3,2,3), value))
