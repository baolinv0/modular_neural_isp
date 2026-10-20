"""Frozen shipped photofinishing renderer and an explicit analytic fallback.

The patent-inspired pre-render operator owns HDR compression. These backends
only accept bounded linear RGB; silently clamping HDR here would lose detail.
"""
import hashlib
import io
from pathlib import Path

import torch
from torch import nn


def _validate_linear_rgb(x: torch.Tensor, *, bounded: bool) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("expected channels-first RGB [B,3,H,W]")
    if any(n == 0 for n in x.shape) or x.dtype != torch.float32:
        raise ValueError("RGB must be a nonempty float32 tensor")
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError("sensor-linear RGB must be finite and nonnegative")
    if bounded and (x > 1).any():
        raise ValueError("backend requires bounded linear RGB; map HDR first")


class AnalyticBackend(nn.Module):
    """Deterministic sRGB OETF for explicit tests/smoke, not a learned ISP."""

    renderer_identity = "analytic-srgb-oetf-v1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_linear_rgb(x, bounded=True)
        return torch.where(x <= .0031308, 12.92 * x,
                           1.055 * torch.pow(x, 1 / 2.4) - .055).clamp(0, 1)


class ModularPhotofinishingBackend(nn.Module):
    """Strictly load a real checkpoint and keep every learned stage frozen.

    ``None`` selects the repository's shipped S24 style-0 checkpoint by an
    absolute module-relative path. Missing/incompatible weights are errors;
    there is no random-weight or analytic fallback.
    """

    def __init__(self, checkpoint: str | Path | None = None, *, device="cpu"):
        super().__init__()
        root = Path(__file__).resolve().parents[1]
        path = Path(checkpoint) if checkpoint is not None else (
            root / "photofinishing/models/photofinishing_s24-style-0.pth")
        self.checkpoint = path.expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"photofinishing checkpoint missing: {self.checkpoint}")
        # Hash and load the same byte snapshot: same-named replaced weights
        # must not share a cache identity, including during concurrent edits.
        checkpoint_bytes = self.checkpoint.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint must contain a model state_dict")
        from photofinishing.photofinishing_model import PhotofinishingModule
        use_lut = any(key.startswith("_3d_lut.") for key in state)
        self.model = PhotofinishingModule(device=torch.device(device), use_3d_lut=use_lut)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.renderer_identity = (
            f"modular-photofinishing:{self.checkpoint.name}:sha256={self.checkpoint_sha256}")
        self.train(False)

    def train(self, mode: bool = True):
        # A parent's .train() must never switch this inference renderer.
        super().train(False)
        return self

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Upstream matrices are plain attributes, not registered buffers.
        if hasattr(self, "model"):
            self.model._rgb_to_ycbcr_matrix = fn(self.model._rgb_to_ycbcr_matrix)
            self.model._ycbcr_to_rgb_matrix = fn(self.model._ycbcr_to_rgb_matrix)
            self.model._device = next(self.model.parameters()).device
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_linear_rgb(x, bounded=True)
        # Upstream guidance pools by four, then reflection-pads by two.
        # Each pooled dimension must therefore exceed two: H,W >= 12.
        if min(x.shape[-2:]) < 12:
            raise ValueError(
                f"Modular photofinishing requires H and W >= 12; got {x.shape[-2]}x{x.shape[-1]}")
        if x.device != next(self.model.parameters()).device:
            raise ValueError("input and frozen renderer must be on the same device")
        self.model.eval()
        output = self.model(x, post_process_ltm=False, return_params=False)
        if output.shape != x.shape or not torch.isfinite(output).all():
            raise RuntimeError("frozen modular renderer returned invalid RGB")
        return output.clamp(0, 1)
