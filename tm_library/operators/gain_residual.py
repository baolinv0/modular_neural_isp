"""Bounded edge-aware local exposure correction of the GTM RGB image."""

import torch
from torch import nn

from tm_library.common import ConditionEncoder, guided_upsample, luminance
from tm_library.config import ModelConfig


class GainResidualOperator(nn.Module):
    """Predict a scalar log2 gain, shared by the three linear RGB channels.

    The low-resolution EV field and the full-resolution guided result both
    stay in ``[-max_ev, max_ev]``. Guided filtering can extrapolate outside
    its input range, so its result needs its own bound. A zero-initialized
    head starts at exact identity; image losses first update the head, then
    reach the conditioning encoder once its weights become nonzero.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_ev = config.max_ev
        self.filter_radius = config.filter_radius
        self.filter_eps = config.filter_eps
        self.encoder = ConditionEncoder(config)
        self.ev_head = nn.Conv2d(config.width, 1, 1)
        nn.init.zeros_(self.ev_head.weight)
        nn.init.zeros_(self.ev_head.bias)

    def predict(self, gain, base, semantics=None, confidence=None):
        features = self.encoder(gain, base, semantics, confidence)
        lowres_ev = self.max_ev * torch.tanh(self.ev_head(features))
        return {"features": features, "lowres_ev": lowres_ev}

    def render(self, gain, base, controls, return_maps=True):
        """Render on GTM for standalone use, or a model-selected anchor.

        Optional ``ev_scale`` is the model's full-resolution strength/ROI gate.
        Applying it before exp2 interpolates exposure in stops and retains
        exact identity for a zero gate, even with diagnostics disabled.
        """
        lowres_ev = controls["lowres_ev"]
        ev = guided_upsample(lowres_ev, luminance(base),
                             radius=self.filter_radius, eps=self.filter_eps)
        ev = ev.clamp(-self.max_ev, self.max_ev)
        if "ev_scale" in controls:
            ev = ev * controls["ev_scale"]
        local_gain = torch.exp2(ev)
        # Leave pre-chroma linear RGB unclipped. Multiplying every channel by
        # the same scalar preserves chromaticity, including amplified colors.
        image = base * local_gain
        maps = ({"lowres_ev": lowres_ev, "ev": ev, "gain": local_gain}
                if return_maps else {})
        return {"image": image, "features": controls["features"], "maps": maps}

    def forward(self, gain, base, semantics=None, confidence=None, *, return_maps=True):
        return self.render(gain, base, self.predict(gain, base, semantics, confidence),
                           return_maps=return_maps)
