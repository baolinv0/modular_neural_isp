"""Low-capacity chroma adaptation heads for the unpaired Stage-2 experiment."""
from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn


class ChromaHead(str, Enum):
    FULL_LUT = "full_lut"
    AFFINE_RESIDUAL = "affine_residual"


class FrozenLUTAffineResidual(nn.Module):
    """Applies a bounded global CbCr affine residual after a frozen LuTNet.

    The wrapped Stage-1 LuTNet preserves the source model's image-adaptive,
    nonlinear CbCr mapping. Only a 2x2 residual matrix and two-dimensional
    bias are trainable, so the adaptation capacity is exactly six scalars.
    """

    def __init__(
        self,
        base_lut_net: nn.Module,
        matrix_limit: float = 0.15,
        bias_limit: float = 0.05,
    ) -> None:
        super().__init__()
        if matrix_limit <= 0 or bias_limit <= 0:
            raise ValueError("matrix_limit and bias_limit must be positive")
        if not hasattr(base_lut_net, "get_cbcr_lut_size"):
            raise TypeError("base_lut_net must expose get_cbcr_lut_size()")

        self.base_lut_net = base_lut_net
        for parameter in self.base_lut_net.parameters():
            parameter.requires_grad = False
        self.base_lut_net.eval()

        self.matrix_limit = float(matrix_limit)
        self.bias_limit = float(bias_limit)
        self.matrix_raw = nn.Parameter(torch.zeros(2, 2))
        self.bias_raw = nn.Parameter(torch.zeros(2))
        self.register_buffer("identity_matrix", torch.eye(2))

    def get_cbcr_lut_size(self) -> int:
        return int(self.base_lut_net.get_cbcr_lut_size())

    def get_num_of_params(self) -> int:
        return self.matrix_raw.numel() + self.bias_raw.numel()

    def effective_matrix(self) -> torch.Tensor:
        identity = self.identity_matrix.to(device=self.matrix_raw.device, dtype=self.matrix_raw.dtype)
        return identity + self.matrix_limit * torch.tanh(self.matrix_raw)

    def effective_bias(self) -> torch.Tensor:
        return self.bias_limit * torch.tanh(self.bias_raw)

    def train(self, mode: bool = True) -> "FrozenLUTAffineResidual":
        super().train(mode)
        self.base_lut_net.eval()
        return self

    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_lut = self.base_lut_net(ycbcr)
        if base_lut.ndim != 4 or base_lut.shape[1] != 2:
            raise ValueError(
                "base_lut_net must return [B, 2, LUT_SIZE, LUT_SIZE], "
                f"got {tuple(base_lut.shape)}"
            )
        matrix = self.effective_matrix().to(device=base_lut.device, dtype=base_lut.dtype)
        bias = self.effective_bias().to(device=base_lut.device, dtype=base_lut.dtype)
        adapted = torch.einsum("ij,bjhw->bihw", matrix, base_lut)
        adapted = adapted + bias.view(1, 2, 1, 1)
        return adapted.clamp(-0.5, 0.5)


def configure_chroma_head(
    model: nn.Module,
    head: ChromaHead | str,
    matrix_limit: float = 0.15,
    bias_limit: float = 0.05,
) -> nn.Module:
    """Configures the model's Stage-2 chroma head in place."""
    head = ChromaHead(head)
    current = getattr(model, "_lut_net", None)
    if current is None:
        raise AttributeError("Model missing required module _lut_net")
    if head is ChromaHead.FULL_LUT:
        if isinstance(current, FrozenLUTAffineResidual):
            raise ValueError("Cannot restore full_lut after affine wrapper construction")
        return current
    if isinstance(current, FrozenLUTAffineResidual):
        return current
    wrapped = FrozenLUTAffineResidual(current, matrix_limit=matrix_limit, bias_limit=bias_limit)
    model._lut_net = wrapped
    return wrapped
