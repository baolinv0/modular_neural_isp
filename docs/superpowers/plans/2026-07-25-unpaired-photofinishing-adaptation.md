# Unpaired Photofinishing Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimum experiment that adapts an existing Photofinishing checkpoint from same-scene, non-pixel-aligned references in two strictly separated stages.

**Architecture:** Stage 1 freezes LTM, Chroma, Gamma, and optional 3D LUT and trains only GainNet and GlobalToneMappingNet with position-independent luminance statistics. Stage 2 loads the accepted Stage-1 checkpoint, freezes the luminance chain, and trains only LuTNet with position-independent chroma statistics plus frozen Stage-1 luminance/LUT preservation.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, existing `PhotofinishingModule`, pytest.

## Global Constraints

- Do not modify the existing paired `photofinishing/train.py` path.
- Do not use pixelwise loss between the B output and A reference.
- Stage 1 trainable scope is exactly `_gain_net` and `_gtm_net`.
- Stage 2 trainable scope is exactly `_lut_net`.
- LTM, Gamma, and optional 3D LUT remain frozen in both stages.
- Stage 2 must compare final luminance and predicted LUT against a frozen copy of the Stage-1 checkpoint.
- Same-scene pairs are supplied by an explicit CSV manifest; no filename-order pairing is allowed.

---

### Task 1: Manifest-driven non-aligned dataset

**Files:** `photofinishing/unpaired_reference_data.py`, `tests/test_unpaired_reference_data.py`

- [x] Validate explicit sample IDs, paths, and split membership.
- [x] Resize input/reference independently without registration or shared crops.
- [x] Support precomputed linear-sRGB inputs and existing raw+metadata conversion.

### Task 2: Position-independent losses

**Files:** `photofinishing/unpaired_style_losses.py`, `tests/test_unpaired_style_losses.py`

- [x] Implement log exposure, luminance percentile, and luminance quantile-Wasserstein losses.
- [x] Implement differentiable CbCr histogram, chroma moments, saturation distribution, Y preservation, and LUT-delta regularization.
- [x] Verify spatial shuffling leaves distribution losses unchanged.

### Task 3: Strict module freezing

**Files:** `photofinishing/unpaired_stage_control.py`, `tests/test_unpaired_stage_control.py`

- [x] Configure Stage-1 and Stage-2 trainable scopes.
- [x] Fail when required modules are absent or unexpected parameters remain trainable.
- [x] Anchor Stage-1 trainable parameters to the source checkpoint.

### Task 4: Training and validation CLI

**Files:** `photofinishing/train_unpaired_style.py`, `tests/test_unpaired_training.py`, `tests/test_unpaired_cli.py`

- [x] Load a strict source/Stage-1 checkpoint.
- [x] Train and validate each stage using only its allowed losses and parameters.
- [x] Save best/last checkpoints, loss history, provenance, weights, and trainable parameter names.

### Task 5: Evaluation and documentation

**Files:** `photofinishing/eval_unpaired_style.py`, `photofinishing/README_UNPAIRED_STYLE.md`

- [x] Report independent luminance and chroma style distances on a held-out split.
- [x] Compare baseline and adapted checkpoints per scene.
- [x] Document manifest schema, commands, claims, and non-claims.
