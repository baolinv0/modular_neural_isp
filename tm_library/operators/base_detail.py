"""Edge-aware log-luminance decomposition with independent tone/detail controls.

This is a configurable fixed guided decomposition. Semantics condition the
predicted controls; the filter is not a learned semantic decomposition.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from tm_library.common import ConditionEncoder, apply_luminance, luminance


def _box_mean(value, radius):
    if radius == 0:
        return value
    padded = F.pad(value, (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=2 * radius + 1, stride=1)


def _guided_log_base(log_y, radius, eps):
    """Self-guided scalar filter, safe for singleton dimensions and batches."""
    mean = _box_mean(log_y, radius)
    variance = (_box_mean(log_y.square(), radius) - mean.square()).clamp_min(0)
    slope = variance / (variance + eps)
    intercept = mean * (1 - slope)
    return _box_mean(slope, radius) * log_y + _box_mean(intercept, radius)


class BaseDetailOperator(nn.Module):
    """Compress a smooth log base and amplify its residual independently.

    For source log luminance L, the guided base B and detail D=L-B are
    reconstructed as ``logsigmoid(contrast * B + shift) + detail_gain * D``.
    Contrast and detail gain lie in (0.5, 2); shift is bounded by max_ev stops.
    Thus detail gain cannot cancel out of reconstruction. A constant field has
    zero detail, while its brightness remains adjustable by the base curve.
    Input RGB chroma ratios are retained before subsequent pipeline gamut work.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = ConditionEncoder(config)
        self.radius = config.filter_radius
        self.filter_eps = config.filter_eps
        self.max_ev = config.max_ev
        self.base_head = nn.Conv2d(config.width, 2, kernel_size=1)
        self.detail_head = nn.Conv2d(config.width, 1, kernel_size=1)
        # Start near neutral controls, while keeping encoder gradients and
        # semantic/image conditioning active on the first optimization step.
        for head in (self.base_head, self.detail_head):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def forward(self, gain, base, semantics=None, confidence=None):
        features = self.encoder(gain, base, semantics, confidence)
        source_y = luminance(gain).clamp_min(0)
        source_log = (source_y + 1e-6).log()
        smooth_base = _guided_log_base(source_log, self.radius, self.filter_eps)
        detail = source_log - smooth_base

        controls = self.base_head(features).mean(dim=(-2, -1), keepdim=True)
        contrast = (math.log(2) * controls[:, :1].tanh()).exp()
        shift = math.log(2) * self.max_ev * controls[:, 1:].tanh()
        detail_logits = F.interpolate(
            self.detail_head(features), size=gain.shape[-2:], mode="bilinear", align_corners=False
        )
        detail_gain = (math.log(2) * detail_logits.tanh()).exp()
        base_toned = F.logsigmoid(contrast * smooth_base + shift)
        reconstructed_y = (base_toned + detail_gain * detail).clamp(-30, 20).exp()
        # The log floor protects gradients; it must not introduce light into
        # an exactly black pixel when the RGB-ratio helper handles zero luma.
        reconstructed_y = torch.where(source_y > 0, reconstructed_y, torch.zeros_like(reconstructed_y))
        image = apply_luminance(gain, reconstructed_y)
        return {
            "image": image,
            "features": features,
            "maps": {
                "source_log": source_log,
                "base": smooth_base,
                "detail": detail,
                "base_toned": base_toned,
                "base_contrast": contrast,
                "base_shift": shift,
                "detail_gain": detail_gain,
            },
        }
