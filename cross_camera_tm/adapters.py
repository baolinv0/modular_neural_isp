from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


LUMA_WEIGHTS = (0.2989, 0.5870, 0.1140)


@dataclass(frozen=True)
class PairTransformParameters:
    gains: torch.Tensor
    matrix: torch.Tensor
    curve_y: torch.Tensor

    def __post_init__(self) -> None:
        if self.gains.ndim != 2 or self.gains.shape[1] != 3:
            raise ValueError("gains must have shape [B,3]")
        if self.matrix.shape != (self.gains.shape[0], 3, 3):
            raise ValueError("matrix must have shape [B,3,3]")
        if self.curve_y.ndim != 2 or self.curve_y.shape[0] != self.gains.shape[0]:
            raise ValueError("curve_y must have shape [B,K]")
        if self.curve_y.shape[1] < 3:
            raise ValueError("curve_y requires at least three control points")
        tensors = (self.gains, self.matrix, self.curve_y)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("transform parameters must be finite")

    @classmethod
    def identity(
        cls,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        curve_points: int = 6,
    ) -> "PairTransformParameters":
        if batch < 1 or curve_points < 3:
            raise ValueError("batch and curve_points must be positive")
        return cls(
            gains=torch.ones(batch, 3, device=device, dtype=dtype),
            matrix=torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).clone(),
            curve_y=torch.linspace(0.0, 1.0, curve_points, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch, -1)
            .clone(),
        )


@dataclass(frozen=True)
class AdapterOutput:
    image: torch.Tensor
    gains: torch.Tensor
    matrix: torch.Tensor
    curve_y: torch.Tensor


def _validate_image(image: torch.Tensor) -> None:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B,3,H,W]")
    if not torch.isfinite(image).all():
        raise ValueError("image must contain finite values")


def _piecewise_curve(luma: torch.Tensor, curve_y: torch.Tensor) -> torch.Tensor:
    points = curve_y.shape[1]
    position = luma.clamp(0.0, 1.0) * (points - 1)
    lower = position.floor().long().clamp(0, points - 2)
    upper = lower + 1
    fraction = position - lower.to(position.dtype)
    expanded_curve = curve_y[:, :, None, None].expand(-1, -1, luma.shape[2], luma.shape[3])
    y0 = torch.gather(expanded_curve, 1, lower)
    y1 = torch.gather(expanded_curve, 1, upper)
    return y0 + fraction * (y1 - y0)


class TargetCameraAdapter(nn.Module):
    """Low-capacity, metadata-conditioned input-domain residual transform."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 4,
        *,
        curve_points: int = 6,
        max_log_gain: float = 0.35,
        max_matrix_delta: float = 0.10,
    ):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or curve_points < 3:
            raise ValueError("adapter dimensions must be positive")
        self.curve_points = curve_points
        self.max_log_gain = float(max_log_gain)
        self.max_matrix_delta = float(max_matrix_delta)
        self.predictor = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, 3 + 9 + curve_points - 1)
        nn.init.xavier_uniform_(self.predictor[0].weight)
        nn.init.zeros_(self.predictor[0].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def predict_parameters(
        self, features: torch.Tensor, *, confidence: torch.Tensor
    ) -> PairTransformParameters:
        if features.ndim != 2:
            raise ValueError("features must have shape [B,F]")
        raw = self.head(self.predictor(features))
        raw_gain, raw_matrix, raw_curve = torch.split(
            raw, (3, 9, self.curve_points - 1), dim=1
        )
        gate = confidence.to(device=features.device, dtype=features.dtype).reshape(features.shape[0], 1).clamp(0.0, 1.0)
        gains = torch.exp(torch.tanh(raw_gain) * self.max_log_gain * gate)
        identity_matrix = torch.eye(3, device=features.device, dtype=features.dtype).unsqueeze(0)
        matrix = identity_matrix + torch.tanh(raw_matrix).reshape(-1, 3, 3) * self.max_matrix_delta * gate[:, :, None]
        increments = functional.softplus(raw_curve * gate)
        curve_y = torch.cat(
            (torch.zeros_like(increments[:, :1]), torch.cumsum(increments, dim=1)), dim=1
        )
        curve_y = curve_y / curve_y[:, -1:].clamp_min(1e-8)
        return PairTransformParameters(gains=gains, matrix=matrix, curve_y=curve_y)

    @staticmethod
    def apply_explicit(
        image: torch.Tensor,
        parameters: PairTransformParameters,
        *,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        _validate_image(image)
        batch = image.shape[0]
        if parameters.gains.shape[0] != batch:
            raise ValueError("parameter batch must match image batch")
        if confidence.shape not in ((batch,), (batch, 1)):
            raise ValueError("confidence must have shape [B] or [B,1]")
        confidence = confidence.to(device=image.device, dtype=image.dtype).reshape(batch, 1).clamp(0.0, 1.0)
        identity = PairTransformParameters.identity(
            batch=batch,
            device=image.device,
            dtype=image.dtype,
            curve_points=parameters.curve_y.shape[1],
        )
        gains = identity.gains + confidence * (parameters.gains.to(image) - identity.gains)
        matrix = identity.matrix + confidence[:, :, None] * (parameters.matrix.to(image) - identity.matrix)
        curve_y = identity.curve_y + confidence * (parameters.curve_y.to(image) - identity.curve_y)

        gained = image * gains[:, :, None, None]
        flat = gained.permute(0, 2, 3, 1).reshape(batch, -1, 3)
        mixed = torch.bmm(flat, matrix.transpose(1, 2)).reshape(
            batch, image.shape[2], image.shape[3], 3
        ).permute(0, 3, 1, 2)
        weights = image.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        luma = (mixed * weights).sum(dim=1, keepdim=True)
        projected_luma = _piecewise_curve(luma, curve_y)
        scale = projected_luma / luma.clamp_min(1e-6)
        return (mixed * scale).clamp(0.0, 1.0)

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
        *,
        confidence: torch.Tensor,
    ) -> AdapterOutput:
        _validate_image(image)
        if features.ndim != 2 or features.shape[0] != image.shape[0]:
            raise ValueError("features must have shape [B,F]")
        parameters = self.predict_parameters(features, confidence=confidence)
        transformed = self.apply_explicit(image, parameters, confidence=torch.ones_like(confidence))
        return AdapterOutput(
            image=transformed,
            gains=parameters.gains,
            matrix=parameters.matrix,
            curve_y=parameters.curve_y,
        )
