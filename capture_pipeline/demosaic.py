from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .types import RawFrame


_PATTERN_LAYOUTS = {
    "RGGB": (("R", "G"), ("G", "B")),
    "BGGR": (("B", "G"), ("G", "R")),
    "GRBG": (("G", "R"), ("B", "G")),
    "GBRG": (("G", "B"), ("R", "G")),
}


class BilinearBayerDemosaicer(nn.Module):
    """Differentiable normalized-neighborhood Bayer interpolation."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("kernel", torch.ones(1, 1, 3, 3), persistent=False)

    @staticmethod
    def _masks(raw: RawFrame) -> dict[str, torch.Tensor]:
        batch, _, height, width = raw.mosaic.shape
        masks = {
            "R": torch.zeros((batch, 1, height, width), device=raw.mosaic.device, dtype=raw.mosaic.dtype),
            "G": torch.zeros((batch, 1, height, width), device=raw.mosaic.device, dtype=raw.mosaic.dtype),
            "B": torch.zeros((batch, 1, height, width), device=raw.mosaic.device, dtype=raw.mosaic.dtype),
        }
        layout = _PATTERN_LAYOUTS[raw.cfa_pattern]
        for row in range(2):
            for col in range(2):
                masks[layout[row][col]][:, :, row::2, col::2] = 1.0
        return masks

    def forward(self, raw: RawFrame) -> torch.Tensor:
        frame = raw.normalized()
        masks = self._masks(frame)
        channels = []
        kernel = self.kernel.to(device=frame.mosaic.device, dtype=frame.mosaic.dtype)
        for name in ("R", "G", "B"):
            mask = masks[name]
            sampled = frame.mosaic * mask
            numerator = F.conv2d(sampled, kernel, padding=1)
            denominator = F.conv2d(mask, kernel, padding=1).clamp_min(self.eps)
            interpolated = numerator / denominator
            channel = torch.where(mask.bool(), sampled, interpolated)
            channels.append(channel)
        return torch.cat(channels, dim=1)
