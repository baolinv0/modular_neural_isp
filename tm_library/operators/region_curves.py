"""Learnable monotonic curve experts mixed with spatial conditioning.

Experts share one scalar curve across RGB channels. Their identities have no
semantic labels: a region can learn to select any convex combination of them.
"""

import torch
from torch import nn
from torch.nn import functional as F

from tm_library.common import ConditionEncoder, evaluate_curves, monotonic_curves
from tm_library.config import ModelConfig


class RegionCurveOperator(nn.Module):
    """Map gain-adjusted linear RGB with a per-pixel convex curve mixture.

    Each expert has ``curve_bins`` endpoint-including knots. The curve bank is
    learnable; local image and
    confidence-gated semantic features predict its mixture weights.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = ConditionEncoder(config)
        self.weight_head = nn.Conv2d(config.width, config.num_experts, 1)

        # Start with different generic smooth curves so the mixing head receives
        # a useful gradient immediately. These are initialization choices, not
        # fixed expert roles or brightness targets for semantic categories.
        x = torch.linspace(0.0, 1.0, config.curve_bins)
        exponents = torch.linspace(0.7, 1.3, config.num_experts)
        initial_knots = x.unsqueeze(0).pow(exponents.unsqueeze(1))
        increments = initial_knots.diff(dim=-1)
        self.curve_logits = nn.Parameter(torch.log(torch.expm1(increments)))

    def forward(self, gain, base, semantics=None, confidence=None):
        features = self.encoder(gain, base, semantics, confidence)
        logits = self.weight_head(features)
        logits = F.interpolate(logits, size=gain.shape[-2:], mode="bilinear",
                               align_corners=False)
        weights = logits.softmax(dim=1)
        curves = monotonic_curves(self.curve_logits).unsqueeze(0)
        curves = curves.expand(gain.shape[0], -1, -1)

        # This lookup has a normalized [0,1] domain. Pre-TM gain may exceed
        # one; such highlights saturate at the fixed white endpoint.
        expert_images = evaluate_curves(gain.clamp(0.0, 1.0), curves)
        image = (expert_images * weights.unsqueeze(2)).sum(dim=1)
        return {"image": image, "features": features,
                "maps": {"curves": curves, "weights": weights}}
