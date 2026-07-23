# Modular RAW Capture Pipeline Design

## 1. Goal

Build a modular, testable, and trainable camera pipeline that starts from one Bayer RAW frame and executes the following order:

```text
Bayer RAW
→ AE estimation
→ RAW-domain exposure synthesis
→ demosaicing
→ RAW denoising
→ AWB / CCM estimation
→ linear RGB color correction
→ tone mapping
→ image enhancement
→ final sRGB
```

Each major module must support three uses:

1. independent testing with a separately trained checkpoint;
2. partial joint tuning while other modules are frozen;
3. full end-to-end joint tuning through differentiable stages.

This implementation is independent from the existing stage-analysis PR and must not depend on its two-pass analyzer.

## 2. Scope and non-goals

### In scope

- Single-frame Bayer RAW input.
- AE prediction in RAW domain.
- Deterministic exposure simulation over a configurable EV range.
- Differentiable Bayer demosaicing.
- Pluggable denoising, AWB/CCM, tone-mapping, and enhancement modules.
- Module overrides for controlled experiments.
- Per-module freezing and joint optimization.
- Stage outputs and diagnostics.
- Compatibility adapters for existing repository models where their tensor contracts permit reuse.

### Out of scope for the first version

- Shot-noise or read-noise resampling.
- Motion blur or exposure-time-dependent blur.
- Saturated-highlight reconstruction.
- Multi-frame fusion.
- Sensor rolling-shutter simulation.
- Lens shading, defective-pixel correction, or geometric correction.
- Automatic camera-control decomposition into shutter, ISO, and aperture. The AE output is one scalar EV offset.

The first version models only exposure-dependent signal scaling and clipping.

## 3. Exposure model

The default supported range is configurable and initialized to:

```text
EV ∈ [-4, +4]
```

For normalized, black-level-corrected Bayer data `R ∈ [0, 1]`, the target exposure is:

```text
scale = 2 ** EV
R_exposed = clip(R * scale, 0, 1)
```

If the input has not yet been normalized, normalization is:

```text
R_norm = (R_sensor - black_level) / (white_level - black_level)
```

The first version has two clipping modes:

- `hard`: exact `[0, 1]` clamp, used for inference and physical reporting;
- `soft`: differentiable saturation approximation, optional during joint training.

Exposure synthesis must preserve image dimensions, Bayer phase, CFA pattern, metadata, and dtype semantics.

Diagnostics must include:

- predicted EV;
- applied EV after range validation;
- exposure scale;
- saturation ratio;
- deep-shadow ratio;
- per-CFA-channel saturation ratios.

No noise or blur is added when EV changes.

## 4. Core data contracts

### 4.1 RawFrame

```python
@dataclass
class RawFrame:
    mosaic: torch.Tensor          # [B, 1, H, W], normalized or sensor-domain
    black_level: torch.Tensor     # [B, 1] or [B, 4]
    white_level: torch.Tensor     # [B, 1]
    cfa_pattern: str              # RGGB, BGGR, GRBG, or GBRG
    metadata: dict
    is_normalized: bool
```

Validation rules:

- `mosaic` must be finite and four-dimensional;
- batch and metadata-dependent parameter sizes must be compatible;
- height and width must be at least 2;
- CFA pattern must be one of the four supported Bayer patterns;
- `white_level > black_level`;
- normalized values outside a small tolerance around `[0, 1]` must fail explicitly.

### 4.2 AEOutput

```python
@dataclass
class AEOutput:
    ev: torch.Tensor              # [B]
    confidence: torch.Tensor      # [B]
    diagnostics: dict
```

### 4.3 AWBOutput

```python
@dataclass
class AWBOutput:
    illuminant: torch.Tensor      # [B, 3]
    ccm: torch.Tensor             # [B, 3, 3]
    confidence: torch.Tensor      # [B]
    diagnostics: dict
```

### 4.4 ToneMapOutput

```python
@dataclass
class ToneMapOutput:
    output: torch.Tensor
    gain: torch.Tensor | None
    gtm: torch.Tensor | None
    ltm: torch.Tensor | None
    parameters: dict
```

### 4.5 CapturePipelineOutput

```python
@dataclass
class CapturePipelineOutput:
    final_srgb: torch.Tensor
    stages: OrderedDict[str, torch.Tensor]
    ae: AEOutput
    awb: AWBOutput
    tone: ToneMapOutput
    diagnostics: dict
```

## 5. Module interfaces

All trainable components inherit `torch.nn.Module`.

### 5.1 AE estimator

```python
class AEEstimator(nn.Module):
    def forward(self, raw: RawFrame) -> AEOutput:
        ...
```

Required implementations:

- `HistogramRawAE`: deterministic baseline with no checkpoint;
- `LearnedAEAdapter`: wrapper around a separately trained AE model.

The pipeline accepts `override_ev`. When supplied, the AE estimator is not called.

### 5.2 Exposure synthesizer

```python
class RawExposureSynthesizer(nn.Module):
    def forward(self, raw: RawFrame, ev: torch.Tensor) -> RawFrame:
        ...
```

The exposure synthesizer is parameter-free in the first version.

### 5.3 Demosaicer

```python
class BayerDemosaicer(nn.Module):
    def forward(self, raw: RawFrame) -> torch.Tensor:
        ...
```

Required implementation:

- differentiable PyTorch bilinear demosaicing for RGGB, BGGR, GRBG, and GBRG.

Optional inference-only adapter:

- existing Menon demosaicer wrapper for quality comparison. It is not used in end-to-end gradient tests.

Output shape is `[B, 3, H, W]` in camera RGB space.

### 5.4 RAW denoiser

```python
class RawDenoiser(nn.Module):
    def forward(self, camera_rgb: torch.Tensor, metadata: dict) -> torch.Tensor:
        ...
```

Required implementations:

- `IdentityRawDenoiser`;
- adapter for compatible repository denoising checkpoints;
- generic adapter base class for separately trained models.

The first version follows the requested order: exposure synthesis → demosaic → denoise.

### 5.5 AWB / CCM estimator

```python
class AWBEstimator(nn.Module):
    def forward(
        self,
        raw: RawFrame,
        camera_rgb: torch.Tensor,
    ) -> AWBOutput:
        ...
```

Required implementations:

- metadata-based AWB/CCM;
- adapter for current repository AWB models;
- adapter base class for separately trained AWB models.

The pipeline accepts `override_illuminant` and `override_ccm`. If both are supplied, the AWB estimator is not called.

### 5.6 Color transform

```python
class RawToLinearRGB(nn.Module):
    def forward(
        self,
        camera_rgb: torch.Tensor,
        illuminant: torch.Tensor,
        ccm: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

The transform applies white-balance gains followed by the CCM and returns linear RGB.

### 5.7 Tone mapper

```python
class ToneMapper(nn.Module):
    def forward(
        self,
        linear_rgb: torch.Tensor,
        context: dict,
    ) -> ToneMapOutput:
        ...
```

Required implementations:

- identity tone mapper;
- adapter for the repository photofinishing module;
- adapter base class for separately trained TM models.

The repository adapter should expose Gain, GTM, and LTM stages when available. Color LUT and learned gamma are optional adapter-specific behavior, not required by the core interface.

### 5.8 Image enhancer

```python
class ImageEnhancer(nn.Module):
    def forward(self, display_rgb: torch.Tensor, context: dict) -> torch.Tensor:
        ...
```

Required implementations:

- identity enhancer;
- adapter for compatible repository enhancement checkpoints;
- adapter base class for separately trained enhancement models.

### 5.9 Output transfer function

The core pipeline applies a fixed sRGB OETF unless a tone-mapper adapter explicitly returns display-referred sRGB and marks that fact in `ToneMapOutput.parameters`.

The fixed OETF is:

```text
C_sRGB = 12.92 * C_linear                          if C_linear <= 0.0031308
C_sRGB = 1.055 * C_linear ** (1 / 2.4) - 0.055    otherwise
```

## 6. CapturePipeline orchestration

```python
class CapturePipeline(nn.Module):
    def __init__(
        self,
        ae_estimator: AEEstimator,
        exposure_synthesizer: RawExposureSynthesizer,
        demosaicer: BayerDemosaicer,
        denoiser: RawDenoiser,
        awb_estimator: AWBEstimator,
        color_transform: RawToLinearRGB,
        tone_mapper: ToneMapper,
        enhancer: ImageEnhancer,
    ):
        ...

    def forward(
        self,
        raw: RawFrame,
        *,
        override_ev: torch.Tensor | float | None = None,
        override_illuminant: torch.Tensor | None = None,
        override_ccm: torch.Tensor | None = None,
        return_stages: bool = True,
    ) -> CapturePipelineOutput:
        ...
```

Execution order is fixed:

1. validate and normalize input RAW;
2. estimate or override EV;
3. synthesize target-exposure Bayer RAW;
4. demosaic;
5. denoise;
6. estimate or override illuminant and CCM;
7. apply AWB and CCM to obtain linear RGB;
8. tone map;
9. enhance;
10. apply fixed sRGB OETF when needed;
11. clamp final output and assemble diagnostics.

Required stage names:

```text
raw_input
raw_normalized
raw_exposed
demosaiced_raw
denoised_raw
linear_awb
tone_gain
tone_gtm
tone_ltm
tone_output
enhanced_output
final_srgb
```

Unavailable optional tone stages remain absent rather than being filled with duplicate tensors.

## 7. Module replacement and checkpoint loading

A registry maps module roles to constructors:

```text
ae
denoiser
awb
tone_mapper
enhancer
```

Checkpoint loading is role-specific and strict by default:

```python
load_module_checkpoint(
    module,
    checkpoint_path,
    state_dict_key=None,
    strict=True,
)
```

Adapters are responsible for reconciling existing model-specific inputs and outputs with core interfaces. Core pipeline code must not inspect checkpoint-specific keys.

A configuration file or CLI can select:

- baseline module implementation;
- checkpoint path;
- frozen/trainable state;
- device;
- exposure range and clipping mode.

## 8. Training and joint tuning

The pipeline exposes:

```python
pipeline.set_trainable_modules(["tone_mapper", "enhancer"])
```

Rules:

- modules not listed have `requires_grad=False` and are set to evaluation mode;
- listed modules have `requires_grad=True` and are set to training mode;
- parameter-free modules remain unaffected;
- overrides may intentionally bypass a module during ablation testing.

Supported modes:

### Independent module test

Replace one adapter/checkpoint, freeze all other trainable modules, and evaluate stage and final outputs.

### Partial joint tuning

Examples:

```text
AE + AWB frozen; TM + enhancement trainable
AE frozen; AWB + TM trainable
```

### Full joint tuning

AE, denoiser, AWB, TM, and enhancement may all be trainable. The differentiable demosaicer, exposure synthesizer, color transform, and OETF preserve the gradient path from final loss to AE EV prediction, except at hard-clipping saturation points. Soft clipping is available to improve gradient flow in training experiments.

## 9. CLI and configuration

A new entry point will be added without changing existing `main/demo.py`:

```text
main/run_capture_pipeline.py
```

It supports:

- DNG Bayer RAW;
- one-channel 16-bit Bayer PNG plus JSON metadata;
- module configuration file;
- separate checkpoint paths for AE, denoiser, AWB, TM, and enhancement;
- EV override;
- illuminant and CCM override;
- stage image saving;
- JSON diagnostics saving;
- inference mode and joint-training smoke-test mode.

Three-channel linear PNG is not treated as a full-capture input because demosaicing and sensor-domain AE have already been bypassed. It may be supported later through an explicit `linear_rgb_bypass` mode, outside this first version.

## 10. Error handling

The implementation fails explicitly for:

- unsupported CFA pattern;
- non-finite input or module output;
- illegal RAW dimensions;
- invalid black/white levels;
- EV outside configured range when range policy is `error`;
- missing AWB metadata in metadata mode;
- missing or incompatible checkpoint;
- malformed illuminant or CCM override;
- incompatible module output shapes;
- a display-referred tone output being passed through sRGB OETF twice.

The configurable EV range policy supports:

- `error`: reject out-of-range EV;
- `clamp`: clamp EV and record both requested and applied values.

Default is `error` for development and tests.

## 11. Test strategy

Tests are written before implementation.

### Exposure tests

1. `EV=0` preserves normalized RAW exactly.
2. `EV=+1` doubles unsaturated values.
3. `EV=-1` halves values.
4. Hard clipping reports correct saturation ratios.
5. CFA pattern and dimensions are preserved.
6. Out-of-range EV follows the configured range policy.

### Demosaicing tests

7. Each supported CFA pattern produces `[B, 3, H, W]`.
8. Known constant-color mosaics reconstruct expected channel values.
9. Gradients propagate from RGB output to Bayer input.

### Pipeline-contract tests

10. Module invocation order is exact.
11. `override_ev` bypasses AE.
12. AWB overrides bypass the AWB estimator.
13. Stage names and tensor shapes are correct.
14. Non-finite intermediate outputs fail immediately.
15. Fixed sRGB OETF is applied exactly once.

### Modularity and training tests

16. A checkpoint is loaded through its adapter without core-pipeline key handling.
17. Frozen modules receive no gradients.
18. Selected trainable modules receive gradients.
19. In full-joint soft-clipping mode, a final-image loss reaches the AE estimator.
20. Identity adapters produce a deterministic baseline chain.

### CLI tests

21. DNG and Bayer PNG metadata paths are parsed into `RawFrame`.
22. Module configuration selects the correct adapters.
23. All available stages and diagnostics are written.
24. Invalid checkpoint or metadata paths fail with actionable messages.

## 12. Acceptance criteria

The feature is complete when:

- the full ordered pipeline runs from Bayer RAW to final sRGB;
- EV is estimated before exposure synthesis;
- target exposure is synthesized in Bayer RAW domain;
- no noise or blur is introduced by exposure synthesis;
- all major learned modules can be replaced by separately trained checkpoints;
- modules can be independently frozen or jointly trained;
- the final loss can backpropagate to AE in soft-clipping mode;
- model-free unit tests and CLI contract tests pass;
- existing `main/demo.py`, existing model definitions, and existing checkpoint files remain unchanged;
- documentation includes independent-module testing and joint-tuning examples.

## 13. Deferred extensions

The following are explicitly deferred:

- exposure-dependent Poisson-Gaussian noise;
- motion blur and exposure-time simulation;
- shutter/ISO decomposition;
- multi-frame RAW synthesis and fusion;
- learned exposure renderer;
- sensor-specific saturation recovery.
