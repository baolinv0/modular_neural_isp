from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from .types import AWBOutput, RawFrame, ToneMapOutput


def _as_batch_vector(value: Any, *, batch: int, width: int, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 1 and tensor.numel() == width:
        tensor = tensor.view(1, width).expand(batch, width)
    elif tensor.ndim == 2 and tensor.shape == (1, width) and batch > 1:
        tensor = tensor.expand(batch, width)
    if tensor.shape != (batch, width):
        raise ValueError(f"{name} must have shape [{batch}, {width}] or [{width}]; got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values.")
    return tensor


def _as_batch_matrix(value: Any, *, batch: int, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 2 and tensor.shape == (3, 3):
        tensor = tensor.unsqueeze(0).expand(batch, 3, 3)
    elif tensor.ndim == 3 and tensor.shape == (1, 3, 3) and batch > 1:
        tensor = tensor.expand(batch, 3, 3)
    if tensor.shape != (batch, 3, 3):
        raise ValueError(f"{name} must have shape [{batch}, 3, 3] or [3, 3]; got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values.")
    return tensor


class IdentityRawDenoiser(nn.Module):
    def forward(self, camera_rgb: torch.Tensor, metadata: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        return camera_rgb


class ModuleDenoiserAdapter(nn.Module):
    def __init__(self, model: nn.Module, *, pass_metadata: bool = False):
        super().__init__()
        self.model = model
        self.pass_metadata = bool(pass_metadata)

    def forward(self, camera_rgb: torch.Tensor, metadata: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        output = self.model(camera_rgb, metadata) if self.pass_metadata else self.model(camera_rgb)
        if isinstance(output, Mapping):
            for key in ("output", "denoised", "image"):
                if key in output:
                    output = output[key]
                    break
        if not torch.is_tensor(output) or output.shape != camera_rgb.shape:
            raise ValueError("Denoiser output must be a tensor with the same shape as its input.")
        return output


class MetadataAWB(nn.Module):
    def forward(self, raw: RawFrame, camera_rgb: torch.Tensor) -> AWBOutput:
        metadata = raw.metadata
        illum_key = "cam_illum" if "cam_illum" in metadata else "illum_color" if "illum_color" in metadata else None
        if illum_key is None:
            raise KeyError("Metadata does not contain an illuminant key ('cam_illum' or 'illum_color').")
        ccm_key = "color_matrix" if "color_matrix" in metadata else "ccm" if "ccm" in metadata else None
        if ccm_key is None:
            raise KeyError("Metadata does not contain a CCM key ('color_matrix' or 'ccm').")
        batch = camera_rgb.shape[0]
        illuminant = _as_batch_vector(
            metadata[illum_key], batch=batch, width=3, device=camera_rgb.device, dtype=camera_rgb.dtype, name="illuminant"
        )
        ccm = _as_batch_matrix(metadata[ccm_key], batch=batch, device=camera_rgb.device, dtype=camera_rgb.dtype, name="ccm")
        confidence = torch.ones(batch, device=camera_rgb.device, dtype=camera_rgb.dtype)
        return AWBOutput(illuminant, ccm, confidence, {"source": "metadata", "illum_key": illum_key, "ccm_key": ccm_key})


class ModuleAWBAdapter(nn.Module):
    def __init__(self, model: nn.Module, *, input_mode: str = "mosaic"):
        super().__init__()
        if input_mode not in {"mosaic", "rgb"}:
            raise ValueError("input_mode must be 'mosaic' or 'rgb'.")
        self.model = model
        self.input_mode = input_mode

    def forward(self, raw: RawFrame, camera_rgb: torch.Tensor) -> AWBOutput:
        model_input = raw.normalized().mosaic if self.input_mode == "mosaic" else camera_rgb
        result = self.model(model_input)
        if isinstance(result, AWBOutput):
            return result
        if torch.is_tensor(result):
            illuminant, ccm, confidence, diagnostics = result, torch.eye(3, device=result.device, dtype=result.dtype), None, {}
        elif isinstance(result, Mapping):
            illum_key = "illuminant" if "illuminant" in result else "illum" if "illum" in result else None
            if illum_key is None:
                raise KeyError("AWB model dictionary output must contain 'illuminant' or 'illum'.")
            illuminant = result[illum_key]
            ccm = result.get("ccm", torch.eye(3, device=model_input.device, dtype=model_input.dtype))
            confidence = result.get("confidence")
            diagnostics = {k: v for k, v in result.items() if k not in {illum_key, "ccm", "confidence"}}
        else:
            raise TypeError("AWB model output must be AWBOutput, tensor, or mapping.")
        batch = camera_rgb.shape[0]
        illuminant = _as_batch_vector(
            illuminant, batch=batch, width=3, device=camera_rgb.device, dtype=camera_rgb.dtype, name="illuminant"
        )
        ccm = _as_batch_matrix(ccm, batch=batch, device=camera_rgb.device, dtype=camera_rgb.dtype, name="ccm")
        if confidence is None:
            confidence_tensor = torch.ones(batch, device=camera_rgb.device, dtype=camera_rgb.dtype)
        else:
            confidence_tensor = confidence if torch.is_tensor(confidence) else torch.as_tensor(confidence)
            confidence_tensor = confidence_tensor.to(device=camera_rgb.device, dtype=camera_rgb.dtype).reshape(-1)
            if confidence_tensor.numel() == 1 and batch > 1:
                confidence_tensor = confidence_tensor.expand(batch)
            if confidence_tensor.shape != (batch,):
                raise ValueError("AWB confidence must be scalar or shape [B].")
            confidence_tensor = confidence_tensor.clamp(0.0, 1.0)
        return AWBOutput(illuminant, ccm, confidence_tensor, diagnostics)


class LinearColorTransform(nn.Module):
    def __init__(self, *, clamp_output: bool = True, eps: float = 1e-8):
        super().__init__()
        self.clamp_output = bool(clamp_output)
        self.eps = float(eps)

    def forward(self, camera_rgb: torch.Tensor, awb: AWBOutput) -> torch.Tensor:
        if camera_rgb.ndim != 4 or camera_rgb.shape[1] != 3:
            raise ValueError("camera_rgb must have shape [B,3,H,W].")
        batch, _, height, width = camera_rgb.shape
        illum = _as_batch_vector(
            awb.illuminant, batch=batch, width=3, device=camera_rgb.device, dtype=camera_rgb.dtype, name="illuminant"
        )
        ccm = _as_batch_matrix(awb.ccm, batch=batch, device=camera_rgb.device, dtype=camera_rgb.dtype, name="ccm")
        wb_gain = illum[:, 1:2] / illum.clamp_min(self.eps)
        white_balanced = camera_rgb * wb_gain.view(batch, 3, 1, 1)
        flat = white_balanced.permute(0, 2, 3, 1).reshape(batch, -1, 3)
        corrected = torch.bmm(flat, ccm.transpose(1, 2)).reshape(batch, height, width, 3).permute(0, 3, 1, 2)
        return corrected.clamp(0.0, 1.0) if self.clamp_output else corrected


class IdentityToneMapper(nn.Module):
    def forward(self, linear_rgb: torch.Tensor, context: Optional[Mapping[str, Any]] = None) -> ToneMapOutput:
        return ToneMapOutput(linear_rgb, None, None, None, {}, {"tone": linear_rgb})


class ModuleToneAdapter(nn.Module):
    """Wrap a generic tone model returning a tensor, mapping, or ToneMapOutput."""

    def __init__(self, model: nn.Module, *, pass_context: bool = False):
        super().__init__()
        self.model = model
        self.pass_context = bool(pass_context)

    def forward(self, linear_rgb: torch.Tensor, context: Optional[Mapping[str, Any]] = None) -> ToneMapOutput:
        result = self.model(linear_rgb, context) if self.pass_context else self.model(linear_rgb)
        if isinstance(result, ToneMapOutput):
            return result
        if torch.is_tensor(result):
            return ToneMapOutput(result, None, None, None, {}, {"tone": result})
        if not isinstance(result, Mapping) or "output" not in result:
            raise TypeError("Generic tone model must return ToneMapOutput, a tensor, or a mapping containing 'output'.")
        output = result["output"]
        if not torch.is_tensor(output):
            raise TypeError("Generic tone mapping 'output' must be a tensor.")
        stages = {}
        for key in ("gain", "gtm", "ltm", "chroma", "gamma"):
            value = result.get(key)
            if torch.is_tensor(value):
                stages[key] = value
        parameters = dict(result.get("parameters", {}))
        return ToneMapOutput(
            output=output,
            gain=result.get("gain_param"),
            gtm=result.get("gtm_param"),
            ltm=result.get("ltm_param"),
            parameters=parameters,
            stages=stages,
        )


class PhotofinishingToneAdapter(nn.Module):
    def __init__(self, model: nn.Module, **forward_kwargs: Any):
        super().__init__()
        self.model = model
        self.forward_kwargs = dict(forward_kwargs)

    def forward(self, linear_rgb: torch.Tensor, context: Optional[Mapping[str, Any]] = None) -> ToneMapOutput:
        kwargs = dict(self.forward_kwargs)
        kwargs.update({"return_intermediate": True, "return_params": True})
        result = self.model(linear_rgb, **kwargs)
        if torch.is_tensor(result):
            return ToneMapOutput(result, None, None, None, {}, {"tone": result})
        if not isinstance(result, Mapping):
            raise TypeError("Photofinishing model must return a tensor or mapping.")
        if "output" not in result:
            raise KeyError("Photofinishing mapping output must contain 'output'.")
        gain = result.get("pred_gain", result.get("gain_param"))
        gtm = result.get("pred_gtm", result.get("gtm_param"))
        ltm = result.get("pred_ltm", result.get("ltm_param"))
        stages: Dict[str, torch.Tensor] = {}
        for name, keys in {
            "gain": ("lsrgb_gain", "gain"),
            "gtm": ("lsrgb_gtm", "gtm"),
            "ltm": ("lsrgb_ltm", "ltm"),
            "chroma": ("processed_lsrgb", "chroma"),
            "gamma": ("gamma",),
        }.items():
            for key in keys:
                value = result.get(key)
                if torch.is_tensor(value):
                    stages[name] = value
                    break
        parameters = {
            "gain": gain,
            "gtm": gtm,
            "ltm": ltm,
            "chroma_lut": result.get("pred_lut", result.get("chroma_lut_param")),
            "gamma": result.get("pred_gamma", result.get("gamma_param")),
        }
        return ToneMapOutput(result["output"], gain, gtm, ltm, parameters, stages)


class IdentityEnhancer(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


class ModuleEnhancerAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        if isinstance(output, Mapping):
            for key in ("output", "enhanced", "image"):
                if key in output:
                    output = output[key]
                    break
        if not torch.is_tensor(output) or output.shape != image.shape:
            raise ValueError("Enhancer output must be a tensor with the same shape as its input.")
        return output


def load_module_checkpoint(module: nn.Module, path: str | Path, *, strict: bool = True, map_location: Any = "cpu") -> nn.Module:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, Mapping) and "model" in payload and isinstance(payload["model"], Mapping):
        state_dict = payload["model"]
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint must contain a state dictionary mapping.")
    module.load_state_dict(state_dict, strict=strict)
    return module
