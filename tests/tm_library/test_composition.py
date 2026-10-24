"""Version-two composition and explicit stage boundaries."""
import pytest
import torch
from tm_library.config import ModelConfig
from tm_library.common import ConditionEncoder
from tm_library.model import TMModel

torch.set_num_threads(1)


def test_config_v2_selection_and_validation():
    cfg = ModelConfig(base_algorithm='spatial_grid', correction_algorithm='gain_residual')
    assert cfg.architecture_version == 2 and cfg.effective_base == 'spatial_grid'
    for kwargs in [dict(architecture_version=1), dict(algorithm='base_detail', base_algorithm='spatial_grid'),
                   dict(correction_algorithm='base_detail'), dict(correction_strength=1.1),
                   dict(correction_max_delta=0), dict(correction_semantic_channel=0),
                   dict(post_mode='fixed_reference'), dict(semantic_mode='train_only')]:
        with pytest.raises(ValueError):
            ModelConfig(**kwargs)


def test_encoder_accepts_lowres_semantics_and_only_concatenates_analysis_inputs(monkeypatch):
    cfg = ModelConfig(algorithm='region_curves', semantic_mode='explicit', analysis_size=5, width=4)
    encoder = ConditionEncoder(cfg)
    seen = []
    concat_shapes = []
    original_cat = torch.cat
    def checked_cat(tensors, *args, **kwargs):
        concat_shapes.append([value.shape[-2:] for value in tensors])
        return original_cat(tensors, *args, **kwargs)
    monkeypatch.setattr(torch, 'cat', checked_cat)
    hook = encoder.net[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
    image = torch.rand(2, 3, 31, 37)
    masks = torch.rand(2, 3, 7, 9, dtype=torch.float64)
    confidence = torch.zeros(2, 1, 7, 9, dtype=torch.float64)
    torch.testing.assert_close(encoder(image, image, masks, confidence), encoder(image, image), rtol=0, atol=0)
    hook.remove()
    assert seen == [torch.Size([2, 9, 5, 5])] * 2
    assert concat_shapes == [[torch.Size([5, 5])] * 3] * 2


@pytest.mark.parametrize('base', ['baseline', 'spatial_grid', 'exposure_fusion', 'base_detail'])
@pytest.mark.parametrize('correction', ['region_curves', 'gain_residual'])
def test_zero_control_preserves_selected_anchor_exactly(base, correction):
    model = TMModel(ModelConfig(base_algorithm=base, correction_algorithm=correction,
                               width=4, analysis_size=5)).eval()
    image = torch.rand(1, 3, 23, 25)
    with torch.no_grad():
        ctx = model.prepare_global(image)
        controls = model.predict_controls(ctx)
        anchor = model.render(ctx, controls)['image']
        zero_strength = model(image, correction_strength=0)
        zero_mask = model(image, correction_mask=torch.zeros(1, 1, 23, 25))
    assert torch.equal(zero_strength['linear'], anchor)
    assert torch.equal(zero_mask['linear'], anchor)


def test_original_baseline_plus_zero_ev_is_exact_baseline():
    baseline = TMModel(ModelConfig()).eval()
    model = TMModel(ModelConfig(base_algorithm='baseline', correction_algorithm='gain_residual',
                               width=4, analysis_size=5)).eval()
    model.backbone.load_state_dict(baseline.backbone.state_dict())
    image = torch.rand(1, 3, 23, 25)
    with torch.no_grad():
        expected, actual = baseline(image), model(image)
    assert torch.equal(actual['linear'], expected['linear'])
    assert torch.equal(actual['output'], expected['output'])


def test_staged_forward_equivalence_and_disabled_diagnostics():
    model = TMModel(ModelConfig(base_algorithm='spatial_grid', correction_algorithm='gain_residual',
                               width=4, analysis_size=5)).eval()
    with torch.no_grad():
        model.correction_operator.ev_head.bias.fill_(.4)
    image = torch.rand(1, 3, 23, 25)
    ctx = model.prepare_global(image)
    base_controls = model.predict_controls(ctx)
    base = model.render(ctx, base_controls, return_maps=False)
    correction_controls = model.predict_controls(ctx, stage='correction', anchor=base['image'])
    corrected = model.render(ctx, correction_controls, stage='correction', anchor=base['image'], return_maps=False)
    staged = model.finish(corrected['image'], corrected['controls'], return_maps=False)
    direct = model(image, return_maps=False)
    assert torch.equal(staged['output'], direct['output'])
    assert base['maps'] == corrected['maps'] == staged['maps'] == direct['maps'] == {}
    staged['output'].mean().backward()
    assert model.correction_operator.ev_head.weight.grad.abs().sum() > 0


def test_semantic_correction_missing_labels_or_zero_trust_is_exact_anchor():
    model = TMModel(ModelConfig(base_algorithm='baseline', correction_algorithm='gain_residual',
        semantic_mode='explicit', correction_semantic_channel=1, width=4, analysis_size=5)).eval()
    with torch.no_grad():
        model.correction_operator.ev_head.bias.fill_(1)
    image = torch.rand(1, 3, 23, 25)
    with torch.no_grad():
        anchor = model(image, correction_strength=0)['linear']
        missing = model(image)['linear']
        zero = model(image, torch.ones(1, 3, 6, 7), torch.zeros(1, 1, 6, 7))['linear']
    assert torch.equal(missing, anchor) and torch.equal(zero, anchor)


def test_fixed_reference_post_controls_ignore_candidate_perturbations():
    model = TMModel(ModelConfig(algorithm='gain_residual', post_mode='fixed_reference',
                               freeze_backbone=True, width=4, analysis_size=5)).eval()
    image = torch.rand(1, 3, 23, 25)
    with torch.no_grad():
        before = model(image)
        model.operator.ev_head.bias.fill_(.8)
        after = model(image)
    assert not torch.equal(before['linear'], after['linear'])
    assert torch.equal(before['maps']['gamma_factor'], after['maps']['gamma_factor'])
    assert torch.equal(before['maps']['cbcr_lut'], after['maps']['cbcr_lut'])
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_v1_library_checkpoint_is_rejected_with_migration_guidance():
    model = TMModel(ModelConfig())
    payload = model.checkpoint()
    assert payload['format'] == 'tm_library_v2' and payload['architecture_version'] == 2
    old = {'config': {'algorithm': 'baseline'}, 'state_dict': payload['state_dict']}
    with pytest.raises(ValueError, match='retrain|migration'):
        TMModel.from_checkpoint(old)


def test_tiny_baseline_stages_preserve_original_padded_compatibility_forward():
    import torch.nn.functional as F
    model = TMModel(ModelConfig()).eval()
    image = torch.rand(1, 3, 7, 9)
    with torch.no_grad():
        expected = model.backbone(F.pad(image, (0, 11, 0, 13), mode='replicate'), training_mode=True)
        actual = model(image, return_maps=False)
    assert torch.equal(actual['output'], expected['output'][..., :7, :9])


@pytest.mark.parametrize('base,correction', [(name, 'none') for name in
    ['baseline', 'gtm', 'region_curves', 'spatial_grid', 'gain_residual', 'exposure_fusion', 'base_detail']]
    + [('baseline', 'gain_residual'), ('spatial_grid', 'gain_residual'),
       ('exposure_fusion', 'region_curves'), ('base_detail', 'gain_residual')])
def test_every_staged_path_matches_forward_values_and_parameter_gradients(base, correction):
    torch.manual_seed(42)
    model = TMModel(ModelConfig(base_algorithm=base, correction_algorithm=correction,
                               width=3, analysis_size=4, freeze_backbone=True)).eval()
    if model.correction_operator is not None and correction == 'gain_residual':
        with torch.no_grad():
            model.correction_operator.ev_head.bias.fill_(.3)
    image = torch.rand(1, 3, 21, 23)
    context = model.prepare_global(image)
    controls = model.predict_controls(context)
    rendered = model.render(context, controls, return_maps=False)
    if correction != 'none':
        controls = model.predict_controls(context, stage='correction', anchor=rendered['image'])
        rendered = model.render(context, controls, stage='correction', anchor=rendered['image'], return_maps=False)
    staged = model.finish(rendered['image'], rendered['controls'], return_maps=False)
    direct = model(image, return_maps=False)
    assert torch.equal(staged['linear'], direct['linear'])
    assert torch.equal(staged['output'], direct['output'])
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if parameters:
        staged_grads = torch.autograd.grad(staged['output'].mean(), parameters, allow_unused=True)
        direct_grads = torch.autograd.grad(direct['output'].mean(), parameters, allow_unused=True)
        for actual, expected in zip(staged_grads, direct_grads):
            if actual is None or expected is None:
                assert actual is expected
            else:
                assert torch.equal(actual, expected)


def test_semantic_channel_confidence_is_gated_before_reduction():
    cfg = ModelConfig(algorithm='region_curves', semantic_mode='explicit', analysis_size=1, width=2)
    encoder = ConditionEncoder(cfg)
    image = torch.full((1, 3, 4, 4), .3)
    labels = torch.zeros(1, 3, 4, 4)
    labels[:, 1, :2] = 1
    confidence = torch.ones_like(labels)
    confidence[:, 1, :2] = 0
    # Separate reduction would invent confidence*label=.25 at analysis size.
    assert torch.equal(encoder(image, image, labels, confidence), encoder(image, image))


def test_c_correction_fractional_gate_scales_ev_and_preserves_chromaticity():
    model = TMModel(ModelConfig(base_algorithm='baseline', correction_algorithm='gain_residual',
                               width=3, analysis_size=4)).eval()
    with torch.no_grad():
        model.correction_operator.ev_head.bias.fill_(.8)
    image = torch.rand(1, 3, 21, 23)
    with torch.no_grad():
        context = model.prepare_global(image)
        base = model.render(context, model.predict_controls(context))['image']
        controls = model.predict_controls(context, stage='correction', anchor=base)
        half = model.render(context, controls, stage='correction', anchor=base, correction_strength=.5)
        full = model.render(context, controls, stage='correction', anchor=base, correction_strength=1)
    torch.testing.assert_close(half['image'], base * full['maps']['gain'].sqrt(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(half['image'][:, :1] * base[:, 1:2],
                               half['image'][:, 1:2] * base[:, :1], rtol=1e-6, atol=1e-7)


def test_a_correction_respects_additive_delta_bound_and_manual_roi():
    model = TMModel(ModelConfig(base_algorithm='gtm', correction_algorithm='region_curves',
                               correction_max_delta=.1, width=3, analysis_size=4)).eval()
    image = torch.rand(1, 3, 9, 11)
    with torch.no_grad():
        context = model.prepare_global(image)
        anchor = context['base']
        controls = model.predict_controls(context, stage='correction', anchor=anchor)
        roi = torch.ones(1, 1, 9, 11)
        roi[..., :4, :] = 0
        result = model.render(context, controls, stage='correction', anchor=anchor,
                              correction_mask=roi, correction_strength=.5)
    assert torch.equal(result['image'][..., :4, :], anchor[..., :4, :])
    assert (result['image'] - anchor).abs().max() <= .0500001
