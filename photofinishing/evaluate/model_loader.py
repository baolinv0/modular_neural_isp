"""Strict model reconstruction and inference for evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch

from .config import ModelSpec

try:
    from ..eval_unpaired_style import _load_adapted_model, _read_adapted_run_config
    from ..train_unpaired_style import _load_model
except ImportError:  # direct execution
    from photofinishing.eval_unpaired_style import _load_adapted_model, _read_adapted_run_config
    from photofinishing.train_unpaired_style import _load_model


ModelRole = Literal["pretrained", "stage1", "stage2"]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return device


def load_model_from_spec(
    spec: ModelSpec,
    *,
    role: ModelRole,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict[str, object]]:
    """Loads a model while preserving the existing Stage-2 fail-closed contract."""

    if role == "stage2":
        configured_path = str(spec.run_config) if spec.run_config is not None else None
        run_config, run_config_path = _read_adapted_run_config(str(spec.checkpoint), configured_path)
        model = _load_adapted_model(
            str(spec.checkpoint), device, spec.use_3d_lut, run_config
        ).eval()
        metadata = {
            "role": role,
            "label": spec.name,
            "checkpoint": str(spec.checkpoint),
            "run_config": str(run_config_path) if run_config_path else None,
            "chroma_head": run_config.get("chroma_head", "full_lut"),
            "use_3d_lut": spec.use_3d_lut,
        }
        return model, metadata
    model = _load_model(str(spec.checkpoint), device=device, use_3d_lut=spec.use_3d_lut).eval()
    return model, {
        "role": role,
        "label": spec.name,
        "checkpoint": str(spec.checkpoint),
        "run_config": str(spec.run_config) if spec.run_config else None,
        "chroma_head": None,
        "use_3d_lut": spec.use_3d_lut,
    }


@torch.no_grad()
def infer_rgb(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Runs one linear-sRGB target-camera input and returns product RGB [0,1]."""

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Inference image must have shape [H,W,3], got {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to(device)
    result = model(tensor)
    if isinstance(result, dict):
        if "output" not in result:
            raise RuntimeError("Photofinishing model dictionary output lacks 'output'")
        result = result["output"]
    if not isinstance(result, torch.Tensor) or result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 3:
        raise RuntimeError(f"Unexpected Photofinishing output shape/type: {type(result)}")
    output = result[0].detach().float().cpu().permute(1, 2, 0).numpy()
    if not np.isfinite(output).all():
        raise ValueError("Model output contains non-finite values")
    return np.clip(output, 0.0, 1.0).astype(np.float32)
