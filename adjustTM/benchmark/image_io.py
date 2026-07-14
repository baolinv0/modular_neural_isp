from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected three-channel image, got {image.shape}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_linear_png16(path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    image = _read_rgb(path)
    if image.dtype != np.uint16:
        raise TypeError(f"Linear input must be uint16, got {image.dtype}: {path}")
    tensor = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32) / 65535.0)).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def read_srgb_png(path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    image = _read_rgb(path)
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"sRGB image must be uint8 or uint16, got {image.dtype}: {path}")
    tensor = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32) / scale)).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def write_srgb_png16(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Only single-image tensors can be written")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB tensor, got {tuple(image.shape)}")
    array = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = np.round(array * 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write image: {path}")


def fit_pad_tensor(
    tensor: torch.Tensor,
    *,
    max_side: int | None,
    multiple: int = 16,
    pad_mode: str = "replicate",
) -> tuple[torch.Tensor, dict[str, Any]]:
    if tensor.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(tensor.shape)}")
    height, width = tensor.shape[-2:]
    if max_side is None or max(height, width) <= max_side:
        scale = 1.0
        resized_height, resized_width = height, width
        resized = tensor
    else:
        scale = max_side / max(height, width)
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        resized = F.interpolate(tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = (multiple - resized_height % multiple) % multiple
    pad_width = (multiple - resized_width % multiple) % multiple
    padded = F.pad(resized, (0, pad_width, 0, pad_height), mode=pad_mode) if pad_height or pad_width else resized
    geometry = {
        "original_height": height,
        "original_width": width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_bottom": pad_height,
        "pad_right": pad_width,
        "scale": float(scale),
        "multiple": multiple,
    }
    return padded, geometry


def unpad_tensor(tensor: torch.Tensor, geometry: dict[str, Any]) -> torch.Tensor:
    return tensor[..., : int(geometry["resized_height"]), : int(geometry["resized_width"])]


def resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)
