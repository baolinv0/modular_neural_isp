from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from .types import AEOutput, AWBOutput, CapturePipelineOutput, RawFrame, ToneMapOutput


def _batch_scalar(value: torch.Tensor | float, raw: RawFrame, name: str) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=raw.mosaic.device, dtype=raw.mosaic.dtype)
    tensor = tensor.to(device=raw.mosaic.device, dtype=raw.mosaic.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(raw.batch_size)
    elif tensor.numel() == raw.batch_size:
        tensor = tensor.reshape(raw.batch_size)
    else:
        raise ValueError(f"{name} must be scalar or shape [B].")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values.")
    return tensor


def _batch_illuminant(value: torch.Tensor | Sequence[float], *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 1 and tensor.numel() == 3:
        tensor = tensor.view(1, 3).expand(batch, 3)
    elif tensor.ndim == 2 and tensor.shape == (1, 3) and batch > 1:
        tensor = tensor.expand(batch, 3)
    if tensor.shape != (batch, 3):
        raise ValueError("override_illuminant must have shape [3] or [B,3].")
    if not torch.isfinite(tensor).all() or (tensor <= 0).any():
        raise ValueError("override_illuminant must contain finite positive values.")
    return tensor


def _batch_ccm(value: torch.Tensor | Sequence[Sequence[float]], *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 2 and tensor.shape == (3, 3):
        tensor = tensor.unsqueeze(0).expand(batch, 3, 3)
    elif tensor.ndim == 3 and tensor.shape == (1, 3, 3) and batch > 1:
        tensor = tensor.expand(batch, 3, 3)
    if tensor.shape != (batch, 3, 3):
        raise ValueError("override_ccm must have shape [3,3] or [B,3,3].")
    if not torch.isfinite(tensor).all():
        raise ValueError("override_ccm must contain finite values.")
    return tensor


class ModularCapturePipeline(nn.Module):
    """Physical-order, replaceable single-frame RAW capture pipeline."""

    MODULE_NAMES = ("ae", "exposure", "demosaic", "denoiser", "awb", "color", "tone", "enhancement")

    def __init__(
        self,
        *,
        ae_estimator: nn.Module,
        exposure_synthesizer: nn.Module,
        demosaicer: nn.Module,
        denoiser: nn.Module,
        awb_estimator: nn.Module,
        color_transform: nn.Module,
        tone_mapper: nn.Module,
        enhancer: nn.Module,
    ):
        super().__init__()
        self.ae_estimator = ae_estimator
        self.exposure_synthesizer = exposure_synthesizer
        self.demosaicer = demosaicer
        self.denoiser = denoiser
        self.awb_estimator = awb_estimator
        self.color_transform = color_transform
        self.tone_mapper = tone_mapper
        self.enhancer = enhancer

    def _named_stage_modules(self) -> Dict[str, nn.Module]:
        return {
            "ae": self.ae_estimator,
            "exposure": self.exposure_synthesizer,
            "demosaic": self.demosaicer,
            "denoiser": self.denoiser,
            "awb": self.awb_estimator,
            "color": self.color_transform,
            "tone": self.tone_mapper,
            "enhancement": self.enhancer,
        }

    def set_trainable_modules(self, names: Iterable[str]) -> None:
        requested = set(names)
        unknown = requested.difference(self.MODULE_NAMES)
        if unknown:
            raise ValueError(f"Unknown capture module name(s): {', '.join(sorted(unknown))}.")
        for name, module in self._named_stage_modules().items():
            enabled = name in requested
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)

    def module_trainability(self) -> Dict[str, bool]:
        result: Dict[str, bool] = {}
        for name, module in self._named_stage_modules().items():
            params = list(module.parameters())
            result[name] = any(param.requires_grad for param in params) if params else False
        return result

    def forward(
        self,
        raw_frame: RawFrame,
        *,
        override_ev: Optional[torch.Tensor | float] = None,
        override_illuminant: Optional[torch.Tensor | Sequence[float]] = None,
        override_ccm: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
        return_stages: bool = True,
    ) -> CapturePipelineOutput:
        if not isinstance(raw_frame, RawFrame):
            raise TypeError("raw_frame must be a RawFrame.")
        normalized = raw_frame.normalized()

        if override_ev is None:
            ae_output = self.ae_estimator(normalized)
            if not isinstance(ae_output, AEOutput):
                raise TypeError("AE estimator must return AEOutput.")
        else:
            ev = _batch_scalar(override_ev, normalized, "override_ev")
            ae_output = AEOutput(
                ev=ev,
                confidence=torch.ones_like(ev),
                diagnostics={"source": "override"},
            )

        exposed_result = self.exposure_synthesizer(normalized, ae_output.ev)
        if not isinstance(exposed_result, tuple) or len(exposed_result) != 2:
            raise TypeError("Exposure synthesizer must return (RawFrame, diagnostics).")
        exposed_raw, exposure_diagnostics = exposed_result
        if not isinstance(exposed_raw, RawFrame):
            raise TypeError("Exposure synthesizer first output must be RawFrame.")

        demosaiced = self.demosaicer(exposed_raw)
        denoised = self.denoiser(demosaiced, exposed_raw.metadata)

        need_awb_model = override_illuminant is None or override_ccm is None
        estimated_awb: Optional[AWBOutput]
        if need_awb_model:
            estimated_awb = self.awb_estimator(exposed_raw, denoised)
            if not isinstance(estimated_awb, AWBOutput):
                raise TypeError("AWB estimator must return AWBOutput.")
        else:
            estimated_awb = None

        illuminant = (
            _batch_illuminant(
                override_illuminant,
                batch=denoised.shape[0],
                device=denoised.device,
                dtype=denoised.dtype,
            )
            if override_illuminant is not None
            else estimated_awb.illuminant
        )
        ccm = (
            _batch_ccm(override_ccm, batch=denoised.shape[0], device=denoised.device, dtype=denoised.dtype)
            if override_ccm is not None
            else estimated_awb.ccm
        )
        if estimated_awb is None:
            confidence = torch.ones(denoised.shape[0], device=denoised.device, dtype=denoised.dtype)
            awb_output = AWBOutput(illuminant, ccm, confidence, {"source": "override"})
        else:
            diagnostics = dict(estimated_awb.diagnostics)
            if override_illuminant is not None or override_ccm is not None:
                diagnostics["source"] = "partial_override"
            awb_output = AWBOutput(illuminant, ccm, estimated_awb.confidence, diagnostics)

        linear_awb = self.color_transform(denoised, awb_output)
        context: Mapping[str, object] = {
            "metadata": exposed_raw.metadata,
            "ae": ae_output,
            "awb": awb_output,
            "exposure_diagnostics": exposure_diagnostics,
        }
        tone_output = self.tone_mapper(linear_awb, context)
        if not isinstance(tone_output, ToneMapOutput):
            raise TypeError("Tone mapper must return ToneMapOutput.")
        enhanced = self.enhancer(tone_output.output)
        if not torch.is_tensor(enhanced) or enhanced.shape != tone_output.output.shape:
            raise ValueError("Enhancer output must match tone output shape.")
        final_srgb = enhanced.clamp(0.0, 1.0)

        stages: OrderedDict[str, torch.Tensor] = OrderedDict()
        if return_stages:
            stages["raw_input"] = raw_frame.mosaic
            stages["raw_normalized"] = normalized.mosaic
            stages["raw_exposed"] = exposed_raw.mosaic
            stages["demosaiced_raw"] = demosaiced
            stages["denoised_raw"] = denoised
            stages["linear_awb"] = linear_awb
            for name, value in tone_output.stages.items():
                if torch.is_tensor(value):
                    stages[f"tone_{name}"] = value
            stages["tone_output"] = tone_output.output
            stages["enhanced_output"] = enhanced
            stages["final_srgb"] = final_srgb

        diagnostics = {
            "module_order": list(self.MODULE_NAMES),
            "exposure": exposure_diagnostics,
            "trainability": self.module_trainability(),
        }
        return CapturePipelineOutput(final_srgb, stages, ae_output, awb_output, tone_output, diagnostics)
