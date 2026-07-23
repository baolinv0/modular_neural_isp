from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch

SUPPORTED_CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


def _to_level_tensor(value: torch.Tensor | float | int, *, batch: int, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.view(1, 1).expand(batch, 1)
    elif tensor.ndim == 1:
        if tensor.numel() in (1, 4):
            tensor = tensor.view(1, -1).expand(batch, -1)
        elif tensor.numel() == batch:
            tensor = tensor.view(batch, 1)
        else:
            raise ValueError(f"{name} must be scalar, [B], [1], [4], [B,1], or [B,4].")
    elif tensor.ndim == 2:
        if tensor.shape[0] == 1 and batch > 1:
            tensor = tensor.expand(batch, -1)
        if tensor.shape[0] != batch or tensor.shape[1] not in (1, 4):
            raise ValueError(f"{name} must have shape [B,1] or [B,4].")
    else:
        raise ValueError(f"{name} must be scalar, [B], [B,1], or [B,4].")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values.")
    return tensor


def _expand_cfa_levels(levels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if levels.shape[1] == 1:
        return levels[:, :, None, None]
    output = torch.empty((levels.shape[0], 1, height, width), device=levels.device, dtype=levels.dtype)
    output[:, :, 0::2, 0::2] = levels[:, 0].view(-1, 1, 1, 1)
    output[:, :, 0::2, 1::2] = levels[:, 1].view(-1, 1, 1, 1)
    output[:, :, 1::2, 0::2] = levels[:, 2].view(-1, 1, 1, 1)
    output[:, :, 1::2, 1::2] = levels[:, 3].view(-1, 1, 1, 1)
    return output


@dataclass
class RawFrame:
    mosaic: torch.Tensor
    black_level: torch.Tensor | float
    white_level: torch.Tensor | float
    cfa_pattern: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    is_normalized: bool = False

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.mosaic):
            raise TypeError("mosaic must be a torch.Tensor.")
        if self.mosaic.ndim != 4 or self.mosaic.shape[1] != 1:
            raise ValueError("mosaic must have shape [B, 1, H, W].")
        if self.mosaic.shape[2] < 2 or self.mosaic.shape[3] < 2:
            raise ValueError("mosaic height and width must both be at least 2.")
        if not torch.isfinite(self.mosaic).all():
            raise ValueError("mosaic must contain finite values.")
        pattern = str(self.cfa_pattern).upper()
        if pattern not in SUPPORTED_CFA_PATTERNS:
            raise ValueError(f"Unsupported CFA pattern: {self.cfa_pattern}.")
        self.cfa_pattern = pattern
        self.metadata = dict(self.metadata or {})
        batch = self.mosaic.shape[0]
        self.black_level = _to_level_tensor(
            self.black_level, batch=batch, device=self.mosaic.device, dtype=self.mosaic.dtype, name="black_level"
        )
        self.white_level = _to_level_tensor(
            self.white_level, batch=batch, device=self.mosaic.device, dtype=self.mosaic.dtype, name="white_level"
        )
        black_cmp = self.black_level.max(dim=1, keepdim=True).values
        white_cmp = self.white_level.min(dim=1, keepdim=True).values
        if not torch.all(white_cmp > black_cmp):
            raise ValueError("white_level must be greater than black_level for every batch item.")
        if self.is_normalized:
            tolerance = 1e-4
            if self.mosaic.min().item() < -tolerance or self.mosaic.max().item() > 1.0 + tolerance:
                raise ValueError("normalized mosaic values must lie in [0, 1] within tolerance.")

    @property
    def batch_size(self) -> int:
        return int(self.mosaic.shape[0])

    def normalized(self) -> "RawFrame":
        if self.is_normalized:
            normalized_mosaic = self.mosaic.clone()
        else:
            _, _, height, width = self.mosaic.shape
            black = _expand_cfa_levels(self.black_level, height, width)
            white = _expand_cfa_levels(self.white_level, height, width)
            normalized_mosaic = ((self.mosaic - black) / (white - black).clamp_min(torch.finfo(self.mosaic.dtype).eps)).clamp(0.0, 1.0)
        return RawFrame(
            mosaic=normalized_mosaic,
            black_level=torch.zeros_like(self.black_level),
            white_level=torch.ones_like(self.white_level),
            cfa_pattern=self.cfa_pattern,
            metadata=dict(self.metadata),
            is_normalized=True,
        )

    def with_mosaic(self, mosaic: torch.Tensor, *, is_normalized: Optional[bool] = None) -> "RawFrame":
        return RawFrame(
            mosaic=mosaic,
            black_level=self.black_level,
            white_level=self.white_level,
            cfa_pattern=self.cfa_pattern,
            metadata=dict(self.metadata),
            is_normalized=self.is_normalized if is_normalized is None else is_normalized,
        )


@dataclass
class AEOutput:
    ev: torch.Tensor
    confidence: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AWBOutput:
    illuminant: torch.Tensor
    ccm: torch.Tensor
    confidence: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToneMapOutput:
    output: torch.Tensor
    gain: Optional[torch.Tensor]
    gtm: Optional[torch.Tensor]
    ltm: Optional[torch.Tensor]
    parameters: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class CapturePipelineOutput:
    final_srgb: torch.Tensor
    stages: OrderedDict[str, torch.Tensor] | Dict[str, torch.Tensor]
    ae: AEOutput
    awb: AWBOutput
    tone: ToneMapOutput
    diagnostics: Dict[str, Any] = field(default_factory=dict)
