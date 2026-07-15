from __future__ import annotations

import importlib
from typing import Any, Mapping, Optional

import torch
from torch import nn

from .adapters import (
    IdentityEnhancer,
    IdentityRawDenoiser,
    IdentityToneMapper,
    LinearColorTransform,
    MetadataAWB,
    ModuleAWBAdapter,
    ModuleDenoiserAdapter,
    ModuleEnhancerAdapter,
    ModuleToneAdapter,
    PhotofinishingToneAdapter,
    load_module_checkpoint,
)
from .demosaic import BilinearBayerDemosaicer
from .exposure import HistogramRawAE, LearnedAEAdapter, RawExposureSynthesizer
from .pipeline import ModularCapturePipeline


def import_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError("class_path must use 'module:object' syntax.")
    module_name, object_name = path.split(":", 1)
    if not module_name or not object_name:
        raise ValueError("class_path must use 'module:object' syntax.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(f"Object '{object_name}' was not found in module '{module_name}'.") from exc


def _instantiate_model(spec: Mapping[str, Any], device: torch.device) -> nn.Module:
    if "module" in spec:
        model = spec["module"]
        if not isinstance(model, nn.Module):
            raise TypeError("spec['module'] must be an nn.Module.")
    else:
        class_path = spec.get("class_path")
        if not class_path:
            raise ValueError("A learned/module spec requires 'class_path' or 'module'.")
        cls = import_object(str(class_path))
        model = cls(**dict(spec.get("kwargs", {})))
        if not isinstance(model, nn.Module):
            raise TypeError(f"{class_path} did not construct an nn.Module.")
    checkpoint = spec.get("checkpoint")
    if checkpoint:
        load_module_checkpoint(model, checkpoint, strict=bool(spec.get("strict", True)), map_location=device)
    return model.to(device)


def build_capture_pipeline(config: Optional[Mapping[str, Any]], device: torch.device) -> ModularCapturePipeline:
    cfg = dict(config or {})
    ev_min = float(cfg.get("ev_min", -4.0))
    ev_max = float(cfg.get("ev_max", 4.0))
    clipping_mode = str(cfg.get("clipping_mode", "hard"))
    soft_beta = float(cfg.get("soft_beta", 12.0))

    ae_spec = dict(cfg.get("ae", {"type": "histogram"}))
    ae_type = ae_spec.get("type", "histogram")
    if ae_type == "histogram":
        ae = HistogramRawAE(
            target_gray=float(ae_spec.get("target_gray", 0.18)),
            ev_min=float(ae_spec.get("ev_min", ev_min)),
            ev_max=float(ae_spec.get("ev_max", ev_max)),
        )
    elif ae_type == "learned":
        ae = LearnedAEAdapter(
            _instantiate_model(ae_spec, device),
            ev_min=float(ae_spec.get("ev_min", ev_min)),
            ev_max=float(ae_spec.get("ev_max", ev_max)),
        )
    else:
        raise ValueError(f"Unsupported AE type: {ae_type}.")

    denoise_spec = dict(cfg.get("denoiser", {"type": "identity"}))
    if denoise_spec.get("type", "identity") == "identity":
        denoiser = IdentityRawDenoiser()
    elif denoise_spec.get("type") == "module":
        denoiser = ModuleDenoiserAdapter(
            _instantiate_model(denoise_spec, device), pass_metadata=bool(denoise_spec.get("pass_metadata", False))
        )
    else:
        raise ValueError(f"Unsupported denoiser type: {denoise_spec.get('type')}.")

    awb_spec = dict(cfg.get("awb", {"type": "metadata"}))
    if awb_spec.get("type", "metadata") == "metadata":
        awb = MetadataAWB()
    elif awb_spec.get("type") == "module":
        awb = ModuleAWBAdapter(
            _instantiate_model(awb_spec, device), input_mode=str(awb_spec.get("input_mode", "mosaic"))
        )
    else:
        raise ValueError(f"Unsupported AWB type: {awb_spec.get('type')}.")

    tone_spec = dict(cfg.get("tone", {"type": "identity"}))
    if tone_spec.get("type", "identity") == "identity":
        tone = IdentityToneMapper()
    elif tone_spec.get("type") in {"photofinishing", "module"}:
        model = _instantiate_model(tone_spec, device)
        adapter = tone_spec.get("adapter", "photofinishing" if tone_spec.get("type") == "photofinishing" else "direct")
        if adapter == "photofinishing":
            tone = PhotofinishingToneAdapter(model, **dict(tone_spec.get("forward_kwargs", {})))
        elif adapter == "direct":
            tone = ModuleToneAdapter(model, pass_context=bool(tone_spec.get("pass_context", False)))
        else:
            raise ValueError(f"Unsupported tone adapter: {adapter}.")
    else:
        raise ValueError(f"Unsupported tone type: {tone_spec.get('type')}.")

    enhancement_spec = dict(cfg.get("enhancement", {"type": "identity"}))
    if enhancement_spec.get("type", "identity") == "identity":
        enhancement = IdentityEnhancer()
    elif enhancement_spec.get("type") == "module":
        enhancement = ModuleEnhancerAdapter(_instantiate_model(enhancement_spec, device))
    else:
        raise ValueError(f"Unsupported enhancement type: {enhancement_spec.get('type')}.")

    pipeline = ModularCapturePipeline(
        ae_estimator=ae,
        exposure_synthesizer=RawExposureSynthesizer(
            ev_min=ev_min, ev_max=ev_max, clipping_mode=clipping_mode, soft_beta=soft_beta
        ),
        demosaicer=BilinearBayerDemosaicer(),
        denoiser=denoiser,
        awb_estimator=awb,
        color_transform=LinearColorTransform(clamp_output=bool(cfg.get("clamp_linear_rgb", True))),
        tone_mapper=tone,
        enhancer=enhancement,
    ).to(device)
    trainable = cfg.get("trainable_modules")
    if trainable is not None:
        pipeline.set_trainable_modules(trainable)
    return pipeline
