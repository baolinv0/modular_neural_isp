"""Fixed-appearance candidate costs. These are transparent research surrogates."""
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ScoreWeights:
    reconstruction: float = 1.0
    subject_luma: float = 0.2
    raw_saturation: float = 0.04
    rendered_noise: float = 0.1
    noiseless_capture_bias: float = 0.1

    def __post_init__(self):
        if any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in asdict(self).values()):
            raise ValueError('Score weights must be finite and nonnegative')


def luma(x):
    w = x.new_tensor([.2126, .7152, .0722]).reshape(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def quality_cost(output, target, saturation_mask, *, subject_mask=None,
                 clean_output=None, weights=None):
    """Return per-image costs without changing target brightness to hide noise.

    Saturation comes from capture provenance, never from tone-mapped highlights.
    `clean_output` is the noiseless *same candidate*, an offline teacher-only
    observation. It is never a deployed policy input.
    """
    weights = weights or ScoreWeights()
    if output.ndim != 4 or output.shape[1] != 3:
        raise ValueError('output must be B,3,H,W')
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if target.shape[1:] != output.shape[1:] or target.shape[0] not in (1, output.shape[0]):
        raise ValueError('target and output shapes do not match')
    if not torch.isfinite(output).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite rendered image')
    if saturation_mask.ndim == 3:
        saturation_mask = saturation_mask.unsqueeze(0)
    if saturation_mask.shape[1:] != output.shape[1:]:
        raise ValueError('Saturation mask must retain capture channel provenance')
    dims = (1, 2, 3)
    mse = (output - target).square().mean(dims)
    dy = (luma(output) - luma(target)).abs()
    if subject_mask is None:
        subject = dy.mean(dims)
    else:
        mask = subject_mask.to(output)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[1:] != (1, *output.shape[-2:]) or mask.shape[0] not in (1, output.shape[0]):
            raise ValueError('Invalid subject mask shape')
        if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError('Subject mask must be finite in [0,1]')
        denom = mask.sum(dims)
        subject = torch.where(denom > 0, (dy * mask).sum(dims) / denom.clamp_min(1e-8), dy.mean(dims))
    saturation = saturation_mask.to(output).mean(dims).expand_as(mse)
    if clean_output is None:
        noise, bias = torch.zeros_like(mse), torch.zeros_like(mse)
    else:
        if clean_output.ndim == 3:
            clean_output = clean_output.unsqueeze(0)
        if clean_output.shape[1:] != output.shape[1:] or clean_output.shape[0] not in (1, output.shape[0]):
            raise ValueError('Noiseless candidate must match rendered output')
        noise = (output - clean_output).square().mean(dims)
        bias = (clean_output - target).square().mean(dims).expand_as(mse)
    cost = (weights.reconstruction * mse + weights.subject_luma * subject
            + weights.raw_saturation * saturation + weights.rendered_noise * noise
            + weights.noiseless_capture_bias * bias)
    return {'cost': cost, 'mse': mse, 'subject_luma_mae': subject,
            'raw_saturation': saturation, 'rendered_noise_mse': noise, 'noiseless_capture_bias_mse': bias}
