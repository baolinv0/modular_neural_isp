from __future__ import annotations

import weakref

import torch
from torch import nn
import torch.nn.functional as F

from .constants import (
    CONTROL_METHODS,
    DUAL_LORA_RANK,
    FILM_HIDDEN,
    PARALLEL_ADAPTER_BOTTLENECK,
    PARAM_RESIDUAL_HIDDEN,
)


def normalize_alpha(alpha: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
    if alpha.ndim == 0:
        alpha = alpha.expand(batch_size)
    if alpha.ndim == 2 and alpha.shape[1] == 1:
        alpha = alpha[:, 0]
    if alpha.ndim != 1 or alpha.shape[0] != batch_size:
        raise ValueError(f"alpha must have shape [B] or [B,1], got {tuple(alpha.shape)}")
    if torch.any(alpha < -1.0) or torch.any(alpha > 1.0):
        raise ValueError("alpha must be in [-1, 1]")
    return alpha.view(batch_size, 1)


def _zero_last(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("Expected final Linear layer")
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


class ControlledPredictor(nn.Module):
    """Controls the final predictor of a frozen GainNet or GTM network."""

    def __init__(
        self,
        base_module: nn.Module,
        sequential_attr: str,
        control_method: str,
        output_kind: str,
    ) -> None:
        super().__init__()
        if control_method not in CONTROL_METHODS:
            raise ValueError(f"Unknown control method: {control_method}")
        if output_kind not in {"gain", "gtm"}:
            raise ValueError(output_kind)
        object.__setattr__(self, "_base_ref", weakref.proxy(base_module))
        self.sequential_attr = sequential_attr
        self.control_method = control_method
        self.output_kind = output_kind

        sequence = self._sequence
        if len(sequence) < 2 or not isinstance(sequence[-2], nn.Linear):
            raise TypeError("Base predictor must end with Linear + activation")
        self.feature_dim = sequence[-2].in_features
        self.output_dim = sequence[-2].out_features

        if control_method == "param_residual":
            self.control_pos = nn.Sequential(
                nn.Linear(self.feature_dim, PARAM_RESIDUAL_HIDDEN),
                nn.SiLU(),
                nn.Linear(PARAM_RESIDUAL_HIDDEN, self.output_dim),
            )
            self.control_neg = nn.Sequential(
                nn.Linear(self.feature_dim, PARAM_RESIDUAL_HIDDEN),
                nn.SiLU(),
                nn.Linear(PARAM_RESIDUAL_HIDDEN, self.output_dim),
            )
            _zero_last(self.control_pos)
            _zero_last(self.control_neg)
        elif control_method == "parallel_adapter":
            self.control_pos = nn.Sequential(
                nn.Linear(self.feature_dim, PARALLEL_ADAPTER_BOTTLENECK),
                nn.SiLU(),
                nn.Linear(PARALLEL_ADAPTER_BOTTLENECK, self.feature_dim),
            )
            self.control_neg = nn.Sequential(
                nn.Linear(self.feature_dim, PARALLEL_ADAPTER_BOTTLENECK),
                nn.SiLU(),
                nn.Linear(PARALLEL_ADAPTER_BOTTLENECK, self.feature_dim),
            )
            _zero_last(self.control_pos)
            _zero_last(self.control_neg)
        elif control_method == "film":
            self.control_pos = nn.Sequential(
                nn.Linear(1, FILM_HIDDEN, bias=False),
                nn.SiLU(),
                nn.Linear(FILM_HIDDEN, 2 * self.feature_dim, bias=False),
            )
            self.control_neg = nn.Sequential(
                nn.Linear(1, FILM_HIDDEN, bias=False),
                nn.SiLU(),
                nn.Linear(FILM_HIDDEN, 2 * self.feature_dim, bias=False),
            )
            nn.init.zeros_(self.control_pos[-1].weight)
            nn.init.zeros_(self.control_neg[-1].weight)
        else:
            self.control_pos_a = nn.Linear(self.feature_dim, DUAL_LORA_RANK, bias=False)
            self.control_pos_b = nn.Linear(DUAL_LORA_RANK, self.output_dim, bias=False)
            self.control_neg_a = nn.Linear(self.feature_dim, DUAL_LORA_RANK, bias=False)
            self.control_neg_b = nn.Linear(DUAL_LORA_RANK, self.output_dim, bias=False)
            nn.init.kaiming_uniform_(self.control_pos_a.weight, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.control_neg_a.weight, a=5 ** 0.5)
            nn.init.zeros_(self.control_pos_b.weight)
            nn.init.zeros_(self.control_neg_b.weight)

    @property
    def _base(self) -> nn.Module:
        return object.__getattribute__(self, "_base_ref")

    @property
    def _sequence(self) -> nn.Sequential:
        if self.sequential_attr == "gain":
            return self._base._gain_net._net
        return self._base._gtm_net

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self._sequence
        size = self._base._input_size
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)
        for layer in list(sequence.children())[:-2]:
            x = layer(x)
        return x

    def _activate(self, logits: torch.Tensor) -> torch.Tensor:
        return self._sequence[-1](logits)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(x)
        batch = features.shape[0]
        alpha = normalize_alpha(alpha, batch, features.device, features.dtype)
        alpha_pos = torch.relu(alpha)
        alpha_neg = torch.relu(-alpha)
        head = self._sequence[-2]
        base_logits = head(features)

        if self.control_method == "param_residual":
            logits = base_logits + alpha_pos * self.control_pos(features) + alpha_neg * self.control_neg(features)
        elif self.control_method == "parallel_adapter":
            controlled_features = features + alpha_pos * self.control_pos(features) + alpha_neg * self.control_neg(features)
            logits = head(controlled_features)
        elif self.control_method == "film":
            film_pos = self.control_pos(alpha_pos)
            film_neg = self.control_neg(alpha_neg)
            gamma_pos, beta_pos = film_pos.chunk(2, dim=1)
            gamma_neg, beta_neg = film_neg.chunk(2, dim=1)
            controlled_features = features * (1.0 + gamma_pos + gamma_neg) + beta_pos + beta_neg
            logits = head(controlled_features)
        else:
            logits = (
                base_logits
                + alpha_pos * self.control_pos_b(self.control_pos_a(features))
                + alpha_neg * self.control_neg_b(self.control_neg_a(features))
            )

        activated = self._activate(logits)
        if self.output_kind == "gain":
            gain = self._base._gain_min + (self._base._gain_max - self._base._gain_min) * activated
            return gain.view(-1, 1, 1, 1)
        return activated
