# Brightness Control Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four fair, CLI-selectable brightness-control adapter variants to the luminance-only photofinishing baseline, modifying only Gain and Global Tone Mapping while freezing the baseline and LTM.

**Architecture:** Training samples are scene-wise ordered level pairs from nine balanced brightness levels. A frozen luminance-only baseline consumes 16-bit linear RGB PNGs and renders with fixed sRGB OETF. Each control variant adds zero-at-alpha control parameters only to Gain and Global Tone Mapping. The same data split, pair schedule, losses, optimizer settings, and parameter budget are shared across variants.

**Tech Stack:** Python, PyTorch, Pillow/OpenCV, pytest, existing modular_neural_isp photofinishing code.

## Global Constraints

- Input images are 16-bit linear RGB PNGs already processed by white balance and CCM; normalize by 65535.
- GT images are 8-bit sRGB PNGs stored in nine directories with identical filenames.
- Levels: a_m100=-1.0, a_m075=-0.75, a_m050=-0.5, a_m025=-0.25, a_000=0.0, a_p025=0.25, a_p050=0.5, a_p075=0.75, a_p100=1.0.
- Load one common luminance-only baseline checkpoint and freeze all baseline parameters.
- LTM remains present and frozen; only Gain and GTM receive control adapters.
- Color modules and learned gamma are absent; use fixed standard sRGB OETF.
- Four variants share the same CLI and are selected by `--control-method`: `param_residual`, `parallel_adapter`, `film`, `dual_lora`.
- Added trainable parameter counts should be matched within approximately ±10%.
- Use brightness-only losses: log-luminance reconstruction, luminance gradient, pairwise monotonicity, and alpha-zero anchor.
- Use scene-wise ordered level pairs. All 36 unordered pairs are balanced per epoch, so each level has equal marginal frequency.

---

### Task 1: Dataset and balanced pair schedule

**Files:**
- Create: `photofinishing/control_data.py`
- Test: `tests/test_control_data.py`

**Interfaces:**
- Produces: `LEVELS`, `BrightnessPairDataset`, `BalancedLevelPairSampler`, `read_linear_png`, `read_srgb_png`.

- [ ] Write tests for strict filename matching, uint16 normalization, all 36 level pairs, equal level frequency, and deterministic epoch schedules.
- [ ] Run `pytest tests/test_control_data.py -v` and verify the tests fail before implementation.
- [ ] Implement the minimal dataset and sampler.
- [ ] Run the tests and verify they pass.

### Task 2: Fixed luminance rendering utilities

**Files:**
- Create: `photofinishing/brightness_ops.py`
- Test: `tests/test_brightness_ops.py`

**Interfaces:**
- Produces: `linear_rgb_to_luminance`, `apply_luminance_scale`, `srgb_oetf`, `log_luminance`, `luminance_gradient`.

- [ ] Write tests for exact sRGB break point behavior, monotonic OETF, chromaticity-preserving scaling, and finite log luminance.
- [ ] Verify tests fail.
- [ ] Implement utilities.
- [ ] Verify tests pass.

### Task 3: Control adapter modules

**Files:**
- Create: `photofinishing/control_adapters.py`
- Test: `tests/test_control_adapters.py`

**Interfaces:**
- Produces: `ParamResidualControl`, `ParallelBottleneckControl`, `FiLMControl`, `DualLoRAControl`, `build_control_adapter`, `count_trainable_parameters`.

- [ ] Write tests that every adapter is exactly zero at alpha=0, separates positive/negative branches, accepts batch alpha tensors, and exposes only adapter parameters as trainable.
- [ ] Verify tests fail.
- [ ] Implement modules with zero initialization.
- [ ] Verify tests pass.

### Task 4: Gain/GTM controlled wrapper

**Files:**
- Create: `photofinishing/controlled_photofinishing.py`
- Test: `tests/test_controlled_photofinishing.py`

**Interfaces:**
- Produces: `ControlledLuminancePhotofinishing`, `load_frozen_baseline_checkpoint`, `freeze_baseline`.

- [ ] Write tests using a tiny fake baseline to verify alpha=0 exact equivalence, frozen LTM, no trainable color/gamma parameters, and adapter-only gradients.
- [ ] Verify tests fail.
- [ ] Implement wrapper and method-specific Gain/GTM injection.
- [ ] Verify tests pass.

### Task 5: Brightness-only losses

**Files:**
- Create: `photofinishing/control_losses.py`
- Test: `tests/test_control_losses.py`

**Interfaces:**
- Produces: `BrightnessControlLoss`, `pairwise_monotonic_loss`.

- [ ] Write tests for zero loss on exact targets, monotonic penalty only for ordered violations, and alpha-zero anchoring.
- [ ] Verify tests fail.
- [ ] Implement the loss.
- [ ] Verify tests pass.

### Task 6: Unified training and evaluation CLI

**Files:**
- Create: `photofinishing/train_brightness_control.py`
- Create: `photofinishing/eval_brightness_control.py`
- Modify: `README.md`
- Test: `tests/test_control_cli.py`

**Interfaces:**
- Consumes all prior components.
- Produces reproducible four-variant runs and common metrics/log format.

- [ ] Write parser/config tests for all four methods and invalid settings.
- [ ] Verify tests fail.
- [ ] Implement CLI, checkpoint loading, frozen-parameter assertions, deterministic pair schedules, logging, and evaluation.
- [ ] Document commands and directory layout.
- [ ] Run targeted tests and full test suite.

### Task 7: Parameter-budget audit and smoke verification

**Files:**
- Create: `photofinishing/audit_control_variants.py`
- Test: `tests/test_parameter_budget.py`

**Interfaces:**
- Produces a JSON report containing trainable parameter counts and relative deviations for all variants.

- [ ] Write a test requiring configured variants to remain within ±10% or explicitly fail with actionable output.
- [ ] Verify the test fails.
- [ ] Implement audit utility and tune hidden dimensions/ranks.
- [ ] Run compile, unit tests, CLI smoke tests, and audit.
