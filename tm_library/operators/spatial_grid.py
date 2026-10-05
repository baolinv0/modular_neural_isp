"""Learned bilateral grid of local tone controls with trilinear slicing.

The grid axes are depth (learned guidance), vertical position, and horizontal
position. Every pixel receives all five controls from the same 3D lattice.
"""

import torch
from torch import nn
from torch.nn import functional as F

from tm_library.common import ConditionEncoder, tone_curve
from tm_library.config import ModelConfig


def slice_bilateral_grid(grid: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
    """Slice BCDHW coefficients at B1HW guidance in [0, 1].

    ``grid_sample`` takes coordinates in x/y/z order even though its input
    volume is stored in depth/height/width order. Spatial endpoints align with
    grid vertices; singleton output axes sample the center. ``bilinear`` on a
    5D input performs trilinear interpolation, including the guidance axis.
    """
    if grid.ndim != 5 or guidance.ndim != 4 or guidance.shape[1] != 1:
        raise ValueError("Expected BCDHW grid and B1HW guidance")
    if grid.shape[0] != guidance.shape[0]:
        raise ValueError("Grid and guidance batch sizes must match")
    batch, _, height, width = guidance.shape
    y = (torch.linspace(-1, 1, height, device=grid.device, dtype=grid.dtype)
         if height > 1 else grid.new_zeros(1))
    x = (torch.linspace(-1, 1, width, device=grid.device, dtype=grid.dtype)
         if width > 1 else grid.new_zeros(1))
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.expand(batch, -1, -1)
    yy = yy.expand(batch, -1, -1)
    zz = 2 * guidance[:, 0] - 1
    coordinates = torch.stack((xx, yy, zz), dim=-1).unsqueeze(1)
    return F.grid_sample(grid, coordinates, mode="bilinear",
                         padding_mode="border", align_corners=True).squeeze(2)


class SpatialGridOperator(nn.Module):
    """Mix GTM and a local re-tonemapped image using bilateral coefficients.

    Grid entries are unconstrained logits. After slicing, the gate is sigmoid,
    curve parameters are strictly positive, and local gain is bounded to
    [0.25, 4]. These constraints preserve finite, stable local rendering while
    allowing the gate to approach either original or local tone rendering.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.grid_size = config.grid_size
        self.grid_depth = config.grid_depth
        self.encoder = ConditionEncoder(config)
        self.grid_head = nn.Conv2d(config.width, 5 * config.grid_depth, 1)
        self.guidance_head = nn.Sequential(
            nn.Conv2d(config.width + 6, config.width, 1),
            nn.SiLU(),
            nn.Conv2d(config.width, 1, 1),
        )

    def forward(self, gain, base, semantics=None, confidence=None):
        features = self.encoder(gain, base, semantics, confidence)
        pooled = F.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
        grid = self.grid_head(pooled).reshape(
            gain.shape[0], 5, self.grid_depth, self.grid_size, self.grid_size)
        full_features = F.interpolate(features, size=gain.shape[-2:],
                                      mode="bilinear", align_corners=False)
        guide_input = torch.cat((gain, base, full_features), dim=1)
        guidance = torch.sigmoid(self.guidance_head(guide_input))
        raw = slice_bilateral_grid(grid, guidance)
        gate = torch.sigmoid(raw[:, :1])
        a, b, c = (F.softplus(raw[:, 1:4]) + 1e-4).split(1, dim=1)
        local_gain = .25 + 3.75 * torch.sigmoid(raw[:, 4:5])
        local = tone_curve(gain * local_gain, a, b, c)
        image = (1 - gate) * base + gate * local
        coefficients = torch.cat((gate, a, b, c, local_gain), dim=1)
        return {"image": image, "features": features,
                "maps": {"grid": grid, "guidance": guidance,
                         "coefficients": coefficients}}
