"""Edge-aware log-luminance decomposition with independent tone/detail controls.

This is a configurable fixed guided decomposition. Semantics condition the
predicted controls; the filter is not a learned semantic decomposition.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from tm_library.common import ConditionEncoder, apply_luminance, guided_upsample, luminance


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
    Base contrast/shift are spatial analysis-resolution controls, edge-aware
    upsampled to the source. The fixed decomposition radius is in input pixels,
    so its effective scene scale depends on the input resolution.
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

    def predict(self, gain, base, semantics=None, confidence=None):
        features = self.encoder(gain, base, semantics, confidence)
        base_logits = self.base_head(features)
        return {"features": features,
                "base_contrast": (math.log(2) * base_logits[:, :1].tanh()).exp(),
                "base_shift": math.log(2) * self.max_ev * base_logits[:, 1:].tanh(),
                "detail_gain": (math.log(2) * self.detail_head(features).tanh()).exp()}

    def render(self, gain, base, controls, return_maps=True):
        source_y = luminance(gain).clamp_min(0)
        source_log = (source_y + 1e-6).log()
        smooth_base = _guided_log_base(source_log, self.radius, self.filter_eps)
        detail = source_log - smooth_base

        contrast = guided_upsample(controls["base_contrast"], source_y,
                                  radius=self.radius, eps=self.filter_eps).clamp(.5, 2.)
        shift_bound = math.log(2) * self.max_ev
        shift = guided_upsample(controls["base_shift"], source_y,
                               radius=self.radius, eps=self.filter_eps).clamp(-shift_bound, shift_bound)
        detail_gain = F.interpolate(controls["detail_gain"], size=gain.shape[-2:],
                                    mode="bilinear", align_corners=False)
        base_toned = F.logsigmoid(contrast * smooth_base + shift)
        reconstructed_y = (base_toned + detail_gain * detail).clamp(-30, 20).exp()
        # The log floor protects gradients; it must not introduce light into
        # an exactly black pixel when the RGB-ratio helper handles zero luma.
        reconstructed_y = torch.where(source_y > 0, reconstructed_y, torch.zeros_like(reconstructed_y))
        image = apply_luminance(gain, reconstructed_y)
        return {
            "image": image,
            "features": controls["features"],
            "maps": {
                "source_log": source_log,
                "base": smooth_base,
                "detail": detail,
                "base_toned": base_toned,
                "base_contrast": contrast,
                "base_shift": shift,
                "detail_gain": detail_gain,
            } if return_maps else {},
        }

    def forward(self, gain, base, semantics=None, confidence=None, *, return_maps=True):
        return self.render(gain, base, self.predict(gain, base, semantics, confidence),
                           return_maps=return_maps)
