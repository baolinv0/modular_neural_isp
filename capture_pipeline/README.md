# Modular RAW Capture Pipeline

This package implements a single-frame camera simulation and joint-tuning pipeline:

```text
Bayer RAW
→ RAW-domain AE estimation
→ deterministic EV synthesis
→ differentiable demosaic
→ RAW RGB denoise
→ AWB / CCM estimation
→ linear color transform
→ tone mapping
→ image enhancement
→ final sRGB
```

Version 1 changes RAW signal magnitude only:

```text
raw_exposed = clip(raw_normalized * 2 ** EV, 0, 1)
```

It does not simulate shot noise, read noise, motion blur, or exposure-time-dependent blur.

## Baseline inference

A config is optional. Without one, the factory uses:

- `HistogramRawAE`
- hard RAW exposure clipping
- differentiable bilinear Bayer demosaic
- identity denoiser
- metadata AWB/CCM
- identity tone mapping
- identity enhancement

```bash
python main/run_modular_capture.py \
  --input-file /data/scene.dng \
  --output-dir results \
  --clipping-mode hard
```

For a single-channel 16-bit Bayer PNG, provide a sidecar JSON:

```json
{
  "black_level": [64, 64, 64, 64],
  "white_level": 1023,
  "pattern": "RGGB",
  "illum_color": [0.52, 1.0, 0.76],
  "ccm": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
}
```

```bash
python main/run_modular_capture.py \
  --input-file /data/scene.png \
  --metadata-json /data/scene.json \
  --override-ev 1.0 \
  --output-dir results
```

## Plug in separately trained modules

Every learned component remains an ordinary `torch.nn.Module`. Adapters do not rename checkpoint keys.

```python
import torch

from capture_pipeline import (
    BilinearBayerDemosaicer,
    LearnedAEAdapter,
    LinearColorTransform,
    MetadataAWB,
    ModularCapturePipeline,
    ModuleDenoiserAdapter,
    ModuleEnhancerAdapter,
    ModuleToneAdapter,
    RawExposureSynthesizer,
    load_module_checkpoint,
)

raw_ae = MyAEModel()
raw_denoiser = MyDenoiser()
tone_model = MyToneMapper()
enhancer = MyEnhancer()

load_module_checkpoint(raw_ae, "checkpoints/ae.pth")
load_module_checkpoint(raw_denoiser, "checkpoints/denoiser.pth")
load_module_checkpoint(tone_model, "checkpoints/tm.pth")
load_module_checkpoint(enhancer, "checkpoints/enhancement.pth")

pipeline = ModularCapturePipeline(
    ae_estimator=LearnedAEAdapter(raw_ae, ev_min=-4, ev_max=4),
    exposure_synthesizer=RawExposureSynthesizer(clipping_mode="soft"),
    demosaicer=BilinearBayerDemosaicer(),
    denoiser=ModuleDenoiserAdapter(raw_denoiser),
    awb_estimator=MetadataAWB(),
    color_transform=LinearColorTransform(),
    tone_mapper=ModuleToneAdapter(tone_model),
    enhancer=ModuleEnhancerAdapter(enhancer),
)
```

A custom AWB model can return:

```python
{
    "illuminant": illuminant,  # [B, 3]
    "ccm": ccm,                # [B, 3, 3] or [3, 3]
    "confidence": confidence,  # optional [B]
}
```

Wrap it with `ModuleAWBAdapter(model, input_mode="mosaic")` or `input_mode="rgb"`.

For the repository photofinishing model, use `PhotofinishingToneAdapter`. It requests intermediate Gain/GTM/LTM outputs and parameters without changing the original model.

## Freeze modules for isolated testing

```python
pipeline.set_trainable_modules(["tone"])
```

Supported names are:

```text
ae, exposure, demosaic, denoiser, awb, color, tone, enhancement
```

Examples:

```python
# Evaluate a new AE with every downstream component frozen.
pipeline.set_trainable_modules(["ae"])

# Joint-tune tone mapping and enhancement.
pipeline.set_trainable_modules(["tone", "enhancement"])

# Joint-tune the complete learned chain.
pipeline.set_trainable_modules(["ae", "denoiser", "awb", "tone", "enhancement"])
```

## End-to-end optimization

Use soft clipping when gradients must reach the AE output near saturation:

```python
pipeline.exposure_synthesizer.clipping_mode = "soft"
pipeline.set_trainable_modules(["ae", "awb", "tone", "enhancement"])

optimizer = torch.optim.AdamW(
    [parameter for parameter in pipeline.parameters() if parameter.requires_grad],
    lr=1e-4,
)

result = pipeline(raw_frame)
loss = reconstruction_loss(result.final_srgb, target_srgb)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

The forward result contains:

- estimated EV and confidence;
- exposed Bayer RAW;
- demosaiced and denoised camera RGB;
- estimated illuminant and CCM;
- linear AWB output;
- tone mapping stages and parameters;
- enhanced and final sRGB;
- exposure clipping diagnostics.

Use `override_ev`, `override_illuminant`, and `override_ccm` to isolate modules during debugging.

## Cross-camera Samsung-style adaptation

The optional `cross_camera_tm` package adapts already black-level-corrected and white-balanced iPhone linear RGB to the input domain expected by a frozen Samsung-trained photofinishing model. It does not replace the capture pipeline, estimate AWB from scratch, or retrain the Samsung backbone.

Run the deterministic mechanical canary:

```bash
PYTHONPATH=/tmp/cross-camera-torch:. \
  python main/run_cross_camera_adaptation.py synthetic-canary \
  --config configs/cross_camera_tm_v2.yaml \
  --output-dir /tmp/cross-camera-canary
```

Run the unified verification, including all pre-existing capture tests and the included Samsung checkpoint interface canary:

```bash
scripts/run_cross_camera_domain_adaptation_verification.sh
```

The default configuration is intentionally synthetic and sets `pixel_route_enabled: false`. Synthetic output proves deterministic mechanics only; it does not establish improvement on real iPhone data. `real-run` fails closed unless local calibration, trained-adapter, input, metadata, and non-synthetic profile artifacts are supplied, and no remote or closed-model fallback is provided.
