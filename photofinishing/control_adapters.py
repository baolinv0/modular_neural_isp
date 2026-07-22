"""Zero-at-alpha brightness-control adapters for frozen parameter heads."""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

ControlMethod = Literal["param_residual", "parallel_adapter", "film", "dual_lora"]


def _alpha_parts(alpha: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    if alpha.shape != (batch, 1):
        raise ValueError(f"Expected alpha shape {(batch, 1)}, got {tuple(alpha.shape)}")
    return torch.relu(alpha), torch.relu(-alpha)


def _zero_last_linear(module: nn.Module) -> None:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("Module has no Linear layer")
    nn.init.zeros_(linears[-1].weight)
    if linears[-1].bias is not None:
        nn.init.zeros_(linears[-1].bias)


def _closest_hidden(target: int, per_hidden: int, fixed: int, minimum: int = 1) -> int:
    return max(minimum, int(round(max(target - fixed, per_hidden) / per_hidden)))


class ParamResidualControl(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, target_params: int):
        super().__init__()
        hidden = _closest_hidden(target_params, 2 * (feature_dim + output_dim + 1), 2 * output_dim)
        self.pos_head = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))
        self.neg_head = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))
        _zero_last_linear(self.pos_head)
        _zero_last_linear(self.neg_head)

    def forward(self, feature: torch.Tensor, alpha: torch.Tensor, base_head: nn.Linear) -> torch.Tensor:
        pos, neg = _alpha_parts(alpha, feature.shape[0])
        return pos * self.pos_head(feature) + neg * self.neg_head(feature)


class ParallelBottleneckControl(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, target_params: int):
        super().__init__()
        per_hidden = 2 * (2 * feature_dim + 1)
        fixed = 2 * feature_dim
        hidden = _closest_hidden(target_params, per_hidden, fixed)
        self.pos_adapter = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, feature_dim))
        self.neg_adapter = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, feature_dim))
        _zero_last_linear(self.pos_adapter)
        _zero_last_linear(self.neg_adapter)

    def forward(self, feature: torch.Tensor, alpha: torch.Tensor, base_head: nn.Linear) -> torch.Tensor:
        pos, neg = _alpha_parts(alpha, feature.shape[0])
        base = base_head(feature)
        pos_delta = base_head(feature + self.pos_adapter(feature)) - base
        neg_delta = base_head(feature + self.neg_adapter(feature)) - base
        return pos * pos_delta + neg * neg_delta


class FiLMControl(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, target_params: int):
        super().__init__()
        per_hidden = 2 * (2 * feature_dim + 2)
        fixed = 4 * feature_dim
        hidden = _closest_hidden(target_params, per_hidden, fixed)
        self.pos_film = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * feature_dim))
        self.neg_film = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * feature_dim))
        _zero_last_linear(self.pos_film)
        _zero_last_linear(self.neg_film)

    @staticmethod
    def _branch(feature: torch.Tensor, magnitude: torch.Tensor, network: nn.Module,
                base_head: nn.Linear) -> torch.Tensor:
        gamma, beta = network(magnitude).chunk(2, dim=-1)
        modulated = feature * (1.0 + gamma) + beta
        return base_head(modulated) - base_head(feature)

    def forward(self, feature: torch.Tensor, alpha: torch.Tensor, base_head: nn.Linear) -> torch.Tensor:
        pos, neg = _alpha_parts(alpha, feature.shape[0])
        pos_delta = self._branch(feature, pos, self.pos_film, base_head)
        neg_delta = self._branch(feature, neg, self.neg_film, base_head)
        pos_mask = (pos > 0).to(feature.dtype)
        neg_mask = (neg > 0).to(feature.dtype)
        return pos_mask * pos_delta + neg_mask * neg_delta


class DualLoRAControl(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, target_params: int):
        super().__init__()
        rank = max(1, int(round(target_params / max(2 * (feature_dim + output_dim), 1))))
        rank = min(rank, feature_dim)
        self.pos_a = nn.Linear(feature_dim, rank, bias=False)
        self.pos_b = nn.Linear(rank, output_dim, bias=False)
        self.neg_a = nn.Linear(feature_dim, rank, bias=False)
        self.neg_b = nn.Linear(rank, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.pos_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.neg_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.pos_b.weight)
        nn.init.zeros_(self.neg_b.weight)

    def forward(self, feature: torch.Tensor, alpha: torch.Tensor, base_head: nn.Linear) -> torch.Tensor:
        pos, neg = _alpha_parts(alpha, feature.shape[0])
        return pos * self.pos_b(self.pos_a(feature)) + neg * self.neg_b(self.neg_a(feature))


def build_control_adapter(method: ControlMethod, feature_dim: int, output_dim: int,
                          *, target_params: int) -> nn.Module:
    classes = {
        "param_residual": ParamResidualControl,
        "parallel_adapter": ParallelBottleneckControl,
        "film": FiLMControl,
        "dual_lora": DualLoRAControl,
    }
    if method not in classes:
        raise ValueError(f"Unknown control method: {method}")
    return classes[method](feature_dim, output_dim, target_params)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
