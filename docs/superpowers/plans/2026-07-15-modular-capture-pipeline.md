# Modular RAW Capture Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a modular, differentiable single-frame Bayer RAW capture pipeline with RAW-domain AE estimation, deterministic EV synthesis, demosaicing, denoising, AWB/CCM, tone mapping, enhancement, module replacement, freezing, and joint tuning support.

**Architecture:** Add a new `capture_pipeline` package independent from the existing `main/pipeline.py`. The new orchestrator owns the physical processing order and communicates with every stage through typed dataclasses and small `nn.Module` interfaces. Existing repository models are connected through adapters without modifying their checkpoint keys or model definitions.

**Tech Stack:** Python 3.9, PyTorch 2.5.1, NumPy 1.26.4, standard-library `unittest`.

## Global Constraints

- Input is one Bayer mosaic RAW frame with shape `[B, 1, H, W]`.
- Processing order is AE → RAW exposure synthesis → demosaic → denoise → AWB/CCM → color transform → tone map → enhancement → final sRGB.
- Default EV range is `[-4, +4]`.
- Version 1 simulates signal scaling and clipping only; it does not add shot noise, read noise, motion blur, or exposure-time blur.
- Hard clipping is the inference default; soft clipping is optional for gradient experiments.
- Support RGGB, BGGR, GRBG, and GBRG CFA patterns.
- Existing model definitions and checkpoint keys must not be modified.
- `main/demo.py` and `main/pipeline.py` behavior must remain unchanged.
- Production behavior must be introduced test-first.
- All invalid shapes, non-finite values, unsupported CFA patterns, and incompatible overrides fail explicitly.

---

### Task 1: Define RAW and pipeline data contracts

**Files:**
- Create: `capture_pipeline/types.py`
- Create: `capture_pipeline/__init__.py`
- Test: `tests/test_capture_types.py`

**Interfaces:**
- Produces `RawFrame`, `AEOutput`, `AWBOutput`, `ToneMapOutput`, and `CapturePipelineOutput`.
- `RawFrame.normalized()` returns a new normalized frame while preserving CFA and metadata.

- [ ] **Step 1: Write failing validation tests**

Test finite `[B,1,H,W]` mosaics, all four CFA patterns, scalar/four-channel black levels, white-level ordering, normalized-range enforcement, and metadata preservation.

- [ ] **Step 2: Run RED verification**

Run: `python -m unittest tests.test_capture_types -v`
Expected: import failure because `capture_pipeline.types` does not exist.

- [ ] **Step 3: Implement dataclasses and validation**

Implement immutable-style `RawFrame` transformations, batch broadcasting for black/white levels, and explicit error messages.

- [ ] **Step 4: Run GREEN verification**

Run: `python -m unittest tests.test_capture_types -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add capture_pipeline/types.py capture_pipeline/__init__.py tests/test_capture_types.py
git commit -m "feat: add capture pipeline data contracts"
```

---

### Task 2: Implement AE baseline and RAW exposure synthesis

**Files:**
- Create: `capture_pipeline/exposure.py`
- Test: `tests/test_capture_exposure.py`

**Interfaces:**
- Produces `HistogramRawAE.forward(raw: RawFrame) -> AEOutput`.
- Produces `LearnedAEAdapter.forward(raw: RawFrame) -> AEOutput`.
- Produces `RawExposureSynthesizer.forward(raw: RawFrame, ev: Tensor) -> tuple[RawFrame, dict]`.

- [ ] **Step 1: Write failing exposure tests**

Verify EV `+1` doubles unsaturated samples, EV `-1` halves them, hard clipping saturates exactly, soft clipping remains differentiable, EV bounds fail, CFA phase is unchanged, and per-CFA clipping diagnostics are reported.

- [ ] **Step 2: Write failing AE tests**

Verify the histogram baseline outputs one finite EV and confidence per batch item, clamps to the configured EV range, and a learned model returning `[B]` or `[B,1]` is accepted by `LearnedAEAdapter`.

- [ ] **Step 3: Run RED verification**

Run: `python -m unittest tests.test_capture_exposure -v`
Expected: import failure because `capture_pipeline.exposure` does not exist.

- [ ] **Step 4: Implement AE and exposure modules**

Use RAW luminance statistics from the Bayer green samples for the baseline AE. Use `scale = 2 ** ev`; hard mode uses `clamp(0,1)`, soft mode uses a smooth upper saturation function and preserves nonzero gradients.

- [ ] **Step 5: Run GREEN verification**

Run: `python -m unittest tests.test_capture_exposure -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add capture_pipeline/exposure.py tests/test_capture_exposure.py
git commit -m "feat: add raw AE and exposure synthesis"
```

---

### Task 3: Implement differentiable Bayer demosaicing

**Files:**
- Create: `capture_pipeline/demosaic.py`
- Test: `tests/test_capture_demosaic.py`

**Interfaces:**
- Produces `BilinearBayerDemosaicer.forward(raw: RawFrame) -> Tensor[B,3,H,W]`.

- [ ] **Step 1: Write failing CFA tests**

Construct constant-color synthetic mosaics for RGGB, BGGR, GRBG, and GBRG. Verify output shape, sampled pixel preservation, finite interpolation, and gradient flow to every input mosaic.

- [ ] **Step 2: Run RED verification**

Run: `python -m unittest tests.test_capture_demosaic -v`
Expected: import failure because `capture_pipeline.demosaic` does not exist.

- [ ] **Step 3: Implement bilinear demosaicing**

Create CFA masks from pixel parity, place known samples, and fill missing values with normalized 3×3 convolution kernels. Avoid NumPy conversion so gradients remain in PyTorch.

- [ ] **Step 4: Run GREEN verification**

Run: `python -m unittest tests.test_capture_demosaic -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add capture_pipeline/demosaic.py tests/test_capture_demosaic.py
git commit -m "feat: add differentiable Bayer demosaicing"
```

---

### Task 4: Implement pluggable denoise, AWB, color, tone, and enhancement adapters

**Files:**
- Create: `capture_pipeline/adapters.py`
- Test: `tests/test_capture_adapters.py`

**Interfaces:**
- Produces `IdentityRawDenoiser`, `ModuleDenoiserAdapter`, `MetadataAWB`, `ModuleAWBAdapter`, `LinearColorTransform`, `IdentityToneMapper`, `PhotofinishingToneAdapter`, `IdentityEnhancer`, and `ModuleEnhancerAdapter`.
- Produces `load_module_checkpoint(module, path, strict=True)`.

- [ ] **Step 1: Write failing identity and override tests**

Verify identity modules preserve tensors, metadata AWB reads `illum_color`/`cam_illum` and `ccm`/`color_matrix`, color correction applies WB gains and batched CCM, and malformed metadata fails.

- [ ] **Step 2: Write failing generic-module tests**

Use tiny `nn.Module` fixtures to verify adapters accept common tensor/dictionary output forms, preserve gradients, and checkpoint loading uses unchanged state-dict keys.

- [ ] **Step 3: Write failing photofinishing adapter tests**

Use a fake repository-style photofinishing module returning `output`, `pred_gain`, `pred_gtm`, and `pred_ltm`; verify conversion to `ToneMapOutput` and intermediate stage extraction.

- [ ] **Step 4: Run RED verification**

Run: `python -m unittest tests.test_capture_adapters -v`
Expected: import failure because `capture_pipeline.adapters` does not exist.

- [ ] **Step 5: Implement adapters**

Keep adapters thin: no architecture rewriting, no checkpoint-key remapping, and explicit shape validation at boundaries.

- [ ] **Step 6: Run GREEN verification**

Run: `python -m unittest tests.test_capture_adapters -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add capture_pipeline/adapters.py tests/test_capture_adapters.py
git commit -m "feat: add capture module adapters"
```

---

### Task 5: Implement the modular capture orchestrator

**Files:**
- Create: `capture_pipeline/pipeline.py`
- Test: `tests/test_modular_capture_pipeline.py`

**Interfaces:**
- Produces `ModularCapturePipeline.forward(raw_frame, override_ev=None, override_illuminant=None, override_ccm=None, return_stages=True) -> CapturePipelineOutput`.
- Produces `set_trainable_modules(names: Sequence[str])` and `module_trainability() -> dict[str,bool]`.

- [ ] **Step 1: Write failing order and override tests**

Use recording fake modules to verify exact call order. Verify `override_ev` bypasses AE, AWB overrides bypass the AWB estimator, and every required stage is returned in order.

- [ ] **Step 2: Write failing trainability tests**

Verify `set_trainable_modules(["tone", "enhancement"])` freezes AE/AWB/denoise and enables only tone/enhancement parameters. Unknown module names must fail.

- [ ] **Step 3: Write failing joint-gradient test**

Use a trainable scalar AE, differentiable exposure/demosaic, and simple downstream modules. Backpropagate final image mean and assert the AE parameter receives a finite nonzero gradient in soft-clipping mode.

- [ ] **Step 4: Run RED verification**

Run: `python -m unittest tests.test_modular_capture_pipeline -v`
Expected: import failure because `capture_pipeline.pipeline` does not exist.

- [ ] **Step 5: Implement orchestrator**

Normalize RAW, estimate/override EV, synthesize exposed RAW, demosaic, denoise, estimate/override AWB, color-correct, tone-map, enhance, clamp final sRGB, and assemble diagnostics.

- [ ] **Step 6: Run GREEN verification**

Run: `python -m unittest tests.test_modular_capture_pipeline -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add capture_pipeline/pipeline.py tests/test_modular_capture_pipeline.py
git commit -m "feat: add modular raw capture orchestrator"
```

---

### Task 6: Add repository model factory and CLI

**Files:**
- Create: `capture_pipeline/factory.py`
- Create: `main/run_modular_capture.py`
- Test: `tests/test_capture_factory_cli.py`

**Interfaces:**
- Produces `build_capture_pipeline(config: dict, device: torch.device) -> ModularCapturePipeline`.
- CLI accepts Bayer PNG/DNG input, metadata JSON, module checkpoint paths, EV override, clipping mode, trainable-module list, and output directory.

- [ ] **Step 1: Write failing factory tests**

Verify identity defaults build without checkpoints, requested checkpoint paths are passed to the proper adapter, and incompatible configurations fail before inference.

- [ ] **Step 2: Write failing CLI helper tests**

Verify CFA/black/white-level metadata parsing, output stage naming, JSON diagnostics serialization, and `--override-ev` argument validation.

- [ ] **Step 3: Run RED verification**

Run: `python -m unittest tests.test_capture_factory_cli -v`
Expected: import failure because factory/CLI modules do not exist.

- [ ] **Step 4: Implement factory and CLI**

Reuse repository DNG utilities for decoding only, but retain the Bayer mosaic before demosaicing. Save Bayer/exposed RAW as PNG-16, RGB stages as PNG-16, final image as JPEG, and diagnostics as JSON.

- [ ] **Step 5: Run GREEN verification**

Run: `python -m unittest tests.test_capture_factory_cli -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add capture_pipeline/factory.py main/run_modular_capture.py tests/test_capture_factory_cli.py
git commit -m "feat: add modular capture factory and CLI"
```

---

### Task 7: Document independent testing and joint tuning

**Files:**
- Create: `capture_pipeline/README.md`
- Modify: `capture_pipeline/__init__.py`

**Interfaces:**
- Documents baseline run, AE-only replacement, AWB-only replacement, TM/enhancement replacement, partial freezing, and full end-to-end optimization.

- [ ] **Step 1: Add executable inference example**

Document a command using identity/default modules plus `HistogramRawAE`.

- [ ] **Step 2: Add checkpoint replacement example**

Show Python code constructing adapters around separately trained AE, AWB, TM, and enhancement modules without changing their state-dict keys.

- [ ] **Step 3: Add joint-tuning example**

Show `set_trainable_modules`, optimizer construction from `requires_grad=True` parameters, forward, loss, backward, and optimizer step.

- [ ] **Step 4: Commit**

```bash
git add capture_pipeline/README.md capture_pipeline/__init__.py
git commit -m "docs: explain modular capture testing and tuning"
```

---

### Task 8: Full verification and draft PR

**Files:**
- Review every file changed on `feature/modular-capture-pipeline`.

- [ ] **Step 1: Run all new tests**

```bash
python -m unittest \
  tests.test_capture_types \
  tests.test_capture_exposure \
  tests.test_capture_demosaic \
  tests.test_capture_adapters \
  tests.test_modular_capture_pipeline \
  tests.test_capture_factory_cli -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Compile all new Python files**

```bash
python -m py_compile \
  capture_pipeline/__init__.py \
  capture_pipeline/types.py \
  capture_pipeline/exposure.py \
  capture_pipeline/demosaic.py \
  capture_pipeline/adapters.py \
  capture_pipeline/pipeline.py \
  capture_pipeline/factory.py \
  main/run_modular_capture.py
```

Expected: exit code 0.

- [ ] **Step 3: Inspect branch diff**

Confirm no modifications to `main/demo.py`, `main/pipeline.py`, trained model definitions, or existing checkpoint loaders.

- [ ] **Step 4: Create a draft pull request**

Use title `Add modular RAW capture and joint-tuning pipeline`. Include architecture, module interfaces, tests, and the explicit limitation that version 1 simulates exposure scaling/clipping without noise or blur.
