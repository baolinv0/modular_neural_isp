"""Learnable monotonic curve experts mixed with spatial conditioning.

Experts share one scalar curve across RGB channels. Their identities have no
semantic labels: a region can learn to select any convex combination of them.
"""

import torch
from torch import nn
from torch.nn import functional as F

from tm_library.common import ConditionEncoder, monotonic_curves
from tm_library.config import ModelConfig


class RegionCurveOperator(nn.Module):
    """Map gain-adjusted linear RGB with a per-pixel convex curve mixture.

    Each expert has ``curve_bins`` endpoint-including knots. This variant has
    one learned curve bank shared by all images; the predictor only estimates
    local mixture weights. A nonnegative shoulder coordinate x/(1+x) preserves
    available highlight distinctions above one before lookup. Composition on
    a selected anchor, bounded residuals and ROI blending belong to the model.
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

    def predict(self, gain, base, semantics=None, confidence=None):
        features = self.encoder(gain, base, semantics, confidence)
        return {"features": features, "weight_logits": self.weight_head(features),
                "curves": monotonic_curves(self.curve_logits)}

    def render(self, gain, base, controls, return_maps=True):
        logits = F.interpolate(controls["weight_logits"], size=gain.shape[-2:], mode="bilinear",
                               align_corners=False)
        weights = logits.softmax(dim=1)
        curves = controls["curves"]
        nonnegative = gain.clamp_min(0)
        coordinate = nonnegative / (1 + nonnegative)
        b = gain.shape[0]
        position = coordinate.reshape(b, -1) * (curves.shape[-1] - 1)
        # All experts reuse one RGB index field. Neither K RGB results nor K
        # copies of the int64 index field are materialized by the renderer.
        lower = position.floor().long().clamp(max=curves.shape[-1] - 2)
        upper = lower + 1
        fraction = position - lower.to(position.dtype)
        image = torch.zeros_like(gain)
        for expert in range(curves.shape[0]):
            knots = curves[expert].unsqueeze(0).expand(b, -1)
            y0, y1 = knots.gather(1, lower), knots.gather(1, upper)
            mapped = (y0 + fraction * (y1 - y0)).reshape_as(gain)
            image = image + mapped * weights[:, expert:expert + 1]
        maps = ({"curves": curves.unsqueeze(0).expand(b, -1, -1), "weights": weights}
                if return_maps else {})
        return {"image": image, "features": controls["features"], "maps": maps}

    def forward(self, gain, base, semantics=None, confidence=None, *, return_maps=True):
        return self.render(gain, base, self.predict(gain, base, semantics, confidence),
                           return_maps=return_maps)
