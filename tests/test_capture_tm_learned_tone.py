"""Mechanism tests for coordinated capture/render tone mapping."""
import importlib.util

import pytest
import torch


def make_tone(scheme="apple", trainable=True):
    # The first red run fails here because the complete algorithm is missing.
    assert importlib.util.find_spec("capture_tm.learned_tone") is not None, "learned TM algorithm is absent"
    from capture_tm.learned_tone import ConditionalToneMapper
    return ConditionalToneMapper(scheme, trainable=trainable)


def fixture_rgb():
    ramp = torch.linspace(.015, 3., 16 * 16).reshape(1, 1, 16, 16)
    return ramp * torch.tensor([.8, 1., .65]).reshape(1, 3, 1, 1)


def test_apple_compensates_physical_capture_ev_once():
    tone = make_tone(trainable=False)
    x = fixture_rgb()
    expected = tone(x, capture_ev=0.)
    actual = tone(4. * x, capture_ev=2.)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_samsung_radiance_is_not_divided_by_reference_exposure():
    tone = make_tone("samsung", trainable=False)
    x = fixture_rgb()
    torch.testing.assert_close(tone(x, capture_ev=-2.), tone(x, capture_ev=2.), atol=0, rtol=0)


@pytest.mark.parametrize("scheme", ["apple", "samsung"])
def test_hdr_values_and_render_intent_survive_tone_mapping(scheme):
    tone = make_tone(scheme, trainable=False)
    x = fixture_rgb()
    darker, brighter = tone(x, render_ev=-1.), tone(x, render_ev=1.)
    assert torch.isfinite(brighter).all() and 0 <= brighter.min() <= brighter.max() <= 1
    assert brighter.mean() > darker.mean() + .01
    # Clipping the HDR input to one before the tone curve loses these differences.
    assert not torch.allclose(tone(x), tone(x.clamp_max(1.)), atol=1e-4)


def test_frozen_tone_preserves_input_gradient_without_training_weights():
    tone = make_tone(trainable=False)
    x = fixture_rgb().requires_grad_()
    tone(x).square().mean().backward()
    assert not any(p.requires_grad for p in tone.parameters())
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_trainable_tone_receives_loss_gradients_and_changes_output():
    tone = make_tone("samsung")
    x = fixture_rgb()
    prepared = tone.prepare(x)
    optimizer = torch.optim.Adam([p for p in tone.parameters() if p.requires_grad], lr=.01)
    before = tone.render_prepared(prepared, render_ev=.3)
    (before.square().mean()).backward()
    grads = [p.grad for p in tone.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    optimizer.step()
    assert not torch.allclose(tone.render_prepared(prepared, render_ev=.3), before, atol=1e-5)
    assert all(p.grad is None for p in tone.model.parameters())


def test_prepared_coefficients_reproduce_forward_and_batch_ev():
    tone = make_tone("apple")
    x = torch.cat([fixture_rgb(), fixture_rgb() * 2.])
    ev = torch.tensor([0., 1.])
    prepared = tone.prepare(x, capture_ev=ev)
    actual = tone.render_prepared(prepared, render_ev=torch.tensor([-.5, .5]))
    torch.testing.assert_close(actual, tone(x, capture_ev=ev, render_ev=torch.tensor([-.5, .5])))
    assert actual[1].mean() > actual[0].mean()


def test_invalid_radiance_and_confidence_are_rejected():
    tone = make_tone()
    with pytest.raises(ValueError):
        tone(-fixture_rgb())
    with pytest.raises(ValueError):
        tone(fixture_rgb(), reliability=torch.full((1, 1, 16, 16), 2.))


def test_samsung_virtual_pdf_changes_curve_with_direct_gain_held_fixed(monkeypatch):
    """Ablate only target PDF; the same render EV still multiplies the image."""
    tone = make_tone("samsung", trainable=False)
    prepared = tone.prepare(fixture_rgb())
    prepared["gain"] = torch.full_like(prepared["gain"], 2.)
    full = {ev: tone.render_prepared(prepared, render_ev=ev) for ev in (-1., 0., 1.)}
    original_histogram = tone._soft_histogram
    source_pdf = None
    call_index = 0

    def source_pdf_for_both_inputs(y):
        nonlocal source_pdf, call_index
        # Samsung computes source, preliminary target, then final target PDF.
        # Replace both targets with this render's real source PDF, preserving
        # all pixels, frozen coefficients, adapter weights and total gain.
        if call_index % 3 == 0:
            source_pdf = original_histogram(y)
        call_index += 1
        return source_pdf

    with monkeypatch.context() as patch:
        patch.setattr(tone, "_soft_histogram", source_pdf_for_both_inputs)
        ablated = {ev: tone.render_prepared(prepared, render_ev=ev) for ev in (-1., 0., 1.)}

    # Stock gain 2 with explicit EV -1 means total virtual gain one.
    torch.testing.assert_close(full[-1.], ablated[-1.], atol=0, rtol=0)
    for ev in (0., 1.):
        # Disabling the histogram-derived knee would erase this difference,
        # even though the ordinary gain/brightness response would still pass.
        assert (full[ev] - ablated[ev]).abs().max() > 1e-5
        assert torch.isfinite(full[ev]).all()
    assert full[-1.].mean() < full[0.].mean() < full[1.].mean()
    assert (full[1.] >= full[0.] - 1e-5).float().mean() > .95
    assert (full[0.] >= full[-1.] - 1e-5).float().mean() > .95


def test_samsung_stock_gain_conditions_curve_at_zero_explicit_intent():
    samsung = make_tone("samsung", trainable=False)
    apple = make_tone("apple", trainable=False)
    prepared = samsung.prepare(fixture_rgb())
    # With identical coefficients and zero adapters, Apple supplies the exact
    # no-PDF-knee ablation while applying the same physical total render gain.
    prepared["gain"] = torch.full_like(prepared["gain"], 2.)
    full = samsung.render_prepared(prepared, render_ev=0.)
    no_pdf_knee = apple.render_prepared(prepared, render_ev=0.)
    assert (full - no_pdf_knee).abs().max() > 1e-5
    prepared["gain"] = torch.ones_like(prepared["gain"])
    torch.testing.assert_close(samsung.render_prepared(prepared), apple.render_prepared(prepared), atol=0, rtol=0)


def test_samsung_final_pdf_tracks_effective_gain_including_learned_residual():
    tone = make_tone("samsung")
    prepared = tone.prepare(fixture_rgb())
    prepared["gain"] = torch.full_like(prepared["gain"], 2.)
    # Two parameterizations produce the same total gain sqrt(8). A curve built
    # from the preliminary gain instead of the final one would distinguish them.
    with torch.no_grad():
        tone.conditioner[-1].bias[0] = torch.atanh(torch.tensor(.5))
    residual_result = tone.render_prepared(prepared)
    with torch.no_grad():
        tone.conditioner[-1].bias[0] = 0.
    reparameterized = dict(prepared, gain=prepared["gain"] * (2. ** .5))
    gain_result = tone.render_prepared(reparameterized)
    torch.testing.assert_close(residual_result, gain_result, atol=2e-6, rtol=2e-5)
    gain_result.square().mean().backward()
    gain_gradient = tone.conditioner[-1].bias.grad[0]
    assert torch.isfinite(gain_gradient) and gain_gradient.abs() > 1e-7
