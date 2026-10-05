"""Aligned sRGB supervision with explicit missing-annotation masking."""
from dataclasses import dataclass, fields
import math
import torch
import torch.nn.functional as F
from .common import luminance


@dataclass
class LossConfig:
    rgb: float = 1.0
    log_luma: float = 0.1
    gradient: float = 0.1
    region: float = 0.0
    semantic: float = 0.1

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'loss {field.name} must be a finite nonnegative number')
        if not any(getattr(self, f.name) for f in fields(self)):
            raise ValueError('at least one loss weight must be positive')


def gradient_error(output, target):
    value = output.sum() * 0
    for axis in (-2, -1):
        if output.shape[axis] > 1:
            value = value + (torch.diff(output, dim=axis) - torch.diff(target, dim=axis)).abs().mean()
    return value


def masked_semantic_bce(logits, labels, valid):
    """No labels means zero loss/gradient, rather than fabricated negatives."""
    if logits.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError('semantic logits, labels and validity shapes must agree')
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * valid).sum() / valid.sum().clamp_min(1)


def region_balanced_error(output, target, masks, valid, confidence):
    error = (output - target).abs().mean(1, keepdim=True)
    weights = masks * valid * confidence
    mass = weights.sum((-2, -1))
    region_error = (error * weights).sum((-2, -1)) / mass.clamp_min(1e-8)
    present = (mass > 0).to(error.dtype)
    return (region_error * present).sum() / present.sum().clamp_min(1)


class CompositeLoss:
    def __init__(self, config=None):
        self.config = config or LossConfig()

    def __call__(self, result, batch):
        output, target = result['output'], batch['target']
        terms = {'rgb': (output-target).abs().mean(),
                 'log_luma': (torch.log(luminance(output).clamp_min(1e-4)) - torch.log(luminance(target).clamp_min(1e-4))).abs().mean(),
                 'gradient': gradient_error(output, target),
                 'region': region_balanced_error(output, target, batch['semantics'], batch['semantic_valid'], batch['confidence']),
                 'semantic': output.sum() * 0}
        if 'semantic_logits' in result:
            terms['semantic'] = masked_semantic_bce(result['semantic_logits'], batch['semantics'], batch['semantic_valid'])
        total = sum(getattr(self.config, name) * value for name, value in terms.items())
        return total, terms
