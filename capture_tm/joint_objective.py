"""Independent display target and predeclared factorial study objective.

The reference is an analytic, max-channel Reinhard shoulder followed by the
sRGB OETF. It is deliberately independent of the current learned renderer and
capture plan. These transparent surrogates are not perceptual quality scores.
"""
import torch


def fixed_target(radiance, render_ev=0.):
    if radiance.ndim != 4 or radiance.shape[1] != 3:
        raise ValueError('radiance must be [B,3,H,W]')
    if not torch.isfinite(radiance).all() or (radiance < 0).any():
        raise ValueError('radiance must be finite and nonnegative')
    ev = torch.as_tensor(render_ev, device=radiance.device, dtype=radiance.dtype)
    if ev.numel() not in (1, len(radiance)) or not torch.isfinite(ev).all():
        raise ValueError('render_ev must be finite scalar or [B]')
    x = radiance * torch.exp2(ev.reshape(-1, 1, 1, 1))
    linear = x / (1 + x.amax(1, keepdim=True))
    return torch.where(linear <= .0031308, 12.92 * linear,
                       1.055 * linear.clamp_min(.0031308).pow(1 / 2.4) - .055)


def image_metrics(output, target, missing, subject_mask=None):
    """Differentiable [B] losses; missing means lost in every HDR exposure.

    J = display MSE + .2 subject-luma MAE + .05 gradient MAE + .02 missing.
    Missing is the observed reliability proxy from the capture backend: no
    usable measurement after saturation, a two-sigma signal threshold and HDR
    alignment/rejection. It is not ground-truth irreversible information loss.
    The definition and weights are fixed within each four-group experiment.
    """
    if output.ndim != 4 or output.shape[1] != 3 or min(output.shape[-2:]) < 2:
        raise ValueError('output must be [B,3,H,W] with H,W>=2')
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if missing.ndim == 3:
        missing = missing.unsqueeze(0)
    for name, tensor in (('target', target), ('missing', missing)):
        if tensor.shape[1:] != output.shape[1:] or len(tensor) not in (1, len(output)):
            raise ValueError(f'{name} must match output')
    if not torch.isfinite(output).all() or not torch.isfinite(target).all():
        raise ValueError('images must be finite')
    if missing.dtype != torch.bool:
        raise ValueError('missing must be a boolean information-loss mask')
    weights = output.new_tensor([.2126, .7152, .0722]).view(1, 3, 1, 1)
    delta = output - target
    luma_delta = (delta * weights).sum(1, keepdim=True).abs()
    if subject_mask is None:
        subject = luma_delta.mean((1, 2, 3))
    else:
        mask = subject_mask.to(output)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[1:] != (1, *output.shape[-2:]) or len(mask) not in (1, len(output)):
            raise ValueError('subject mask must be [B,1,H,W]')
        if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError('subject mask must be in [0,1]')
        count = mask.sum((1, 2, 3))
        subject = torch.where(count > 0, (mask * luma_delta).sum((1, 2, 3)) / count.clamp_min(1e-8), luma_delta.mean((1, 2, 3)))
    mse = delta.square().mean((1, 2, 3))
    detail = .5 * (torch.diff(delta, dim=-1).abs().mean((1, 2, 3)) +
                   torch.diff(delta, dim=-2).abs().mean((1, 2, 3)))
    lost = missing.to(output).mean((1, 2, 3)).expand_as(mse)
    target_y = (target * weights).sum(1, keepdim=True)
    dark, light = (target_y < .15).to(output), (target_y > .8).to(output)
    def region_mse(mask):
        return (delta.square() * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3))).clamp_min(1)
    return {'cost': mse + .2 * subject + .05 * detail + .02 * lost,
            'mse': mse, 'subject_luma_mae': subject, 'detail_mae': detail,
            'missing_fraction': lost, 'dark_mse': region_mse(dark),
            'highlight_mse': region_mse(light)}
