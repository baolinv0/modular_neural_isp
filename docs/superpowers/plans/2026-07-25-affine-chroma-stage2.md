# Affine Chroma Stage-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-capacity Stage-2 mode that freezes the Stage-1 adaptive LuTNet and trains only a six-parameter global CbCr affine residual, while preserving the existing full-LUT mode as the default control.

**Architecture:** A `FrozenLUTAffineResidual` wrapper owns the loaded Stage-1 `LuTNet`, keeps it frozen, and transforms its predicted `[B,2,H,W]` LUT with a bounded residual matrix and bias. Training and evaluation construct either the original full LUT path or the affine wrapper from an explicit `chroma_head` field, so both experiments share the same Stage-1 checkpoint, losses, data, and metrics. The wrapper does not add output clamping because the original LuTNet may produce values outside the nominal CbCr interval; identity initialization must preserve those values exactly.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, pytest, existing `PhotofinishingModule` and unpaired-style training code.

## Global Constraints

- `full_lut` remains the default and must preserve current behavior.
- `affine_residual` starts exactly at the Stage-1 output: zero matrix residual and zero bias.
- The frozen base LuTNet retains the source model's scene-adaptive nonlinear color mapping.
- Only six affine parameters may receive gradients in `affine_residual` mode.
- Both modes use the same Stage-2 losses and evaluation metrics.
- Stage-2 run configuration records head type and trainable parameter count.
- Evaluation must reconstruct the adapted head from its run configuration and fail closed on incompatible checkpoints.

---

### Task 1: Low-capacity Chroma Wrapper

**Files:**
- Create: `photofinishing/unpaired_chroma_heads.py`
- Test: `tests/test_unpaired_chroma_heads.py`

**Interfaces:**
- Consumes: an existing `nn.Module` with `forward(ycbcr) -> Tensor[B,2,L,L]` and `get_cbcr_lut_size()`.
- Produces: `FrozenLUTAffineResidual(base_lut_net, matrix_limit=0.15, bias_limit=0.05)` with the same forward and LUT-size interface.

- [ ] Write tests asserting exact identity at initialization even for out-of-range base LUT values, exactly six trainable scalars, frozen base parameters, bounded affine parameters, and gradient flow only to affine parameters.
- [ ] Run `pytest -q tests/test_unpaired_chroma_heads.py` and verify failure because the module does not exist.
- [ ] Implement the wrapper using `A = I + matrix_limit * tanh(matrix_raw)` and `b = bias_limit * tanh(bias_raw)`, applying `A` and `b` to every point of the frozen base LUT without adding output clamping.
- [ ] Re-run the focused tests and commit.

### Task 2: Stage-2 Construction and Trainable Scope

**Files:**
- Modify: `photofinishing/train_unpaired_style.py`
- Modify: `photofinishing/unpaired_stage_control.py`
- Test: `tests/test_unpaired_chroma_stage2.py`

**Interfaces:**
- Produces: `ChromaHead` enum values `full_lut` and `affine_residual`.
- Produces: `configure_chroma_head(model, head)` that wraps `_lut_net` only for affine mode.
- Extends: `configure_trainable_scope(model, stage, chroma_head)` and `assert_trainable_scope(...)`.

- [ ] Add failing tests for CLI selection, default full-LUT behavior, affine wrapper construction, six trainable parameters, and frozen base LuTNet.
- [ ] Run the focused tests and verify expected failures.
- [ ] Add `--chroma-head` to Stage 2, construct the wrapper after loading the Stage-1 checkpoint, preserve full-LUT behavior, and record head type plus trainable parameter count in `run_config.json`.
- [ ] Ensure affine mode calls `.eval()` on the frozen base LuTNet while the residual parameters remain trainable.
- [ ] Re-run focused tests and commit.

### Task 3: Checkpoint Loading and Evaluation

**Files:**
- Modify: `photofinishing/eval_unpaired_style.py`
- Test: `tests/test_unpaired_eval_heads.py`

**Interfaces:**
- Produces: `_read_adapted_run_config(checkpoint, configured_path)`.
- Produces: `_load_adapted_model(checkpoint, device, use_3d_lut, run_config)`.
- Evaluation consumes an adapted `run_config.json` and reconstructs the correct head before strict checkpoint loading.

- [ ] Add failing tests that affine adapted checkpoints require a run config, that full-LUT checkpoints remain loadable without one, and that invalid head metadata fails closed.
- [ ] Verify failures.
- [ ] Load Stage-2 affine checkpoints by constructing the wrapper before strict state-dict loading; retain the original loader for full-LUT checkpoints.
- [ ] Add `--adapted-run-config`, defaulting to the adapted checkpoint directory, and record the evaluated head in output JSON.
- [ ] Re-run tests and commit.

### Task 4: Real Model Canary, Documentation, and CI

**Files:**
- Modify: `tests/test_unpaired_real_interface.py`
- Modify: `photofinishing/README_UNPAIRED_STYLE.md`
- Modify: `.github/workflows/unpaired-photofinishing.yml`

**Interfaces:**
- Real Canary validates Stage-2 forward/backward behavior on `PhotofinishingModule` for both head modes.

- [ ] Add a real-model test asserting affine iteration-0 output matches the Stage-1 model, only six affine parameters receive gradients, and the base LuTNet receives none.
- [ ] Document paired commands for `full_lut` and `affine_residual`, the fairness constraints, and result interpretation.
- [ ] Run `python -m compileall -q photofinishing` and all `tests/test_unpaired_*.py`.
- [ ] Push, create a Draft PR, and require GitHub Actions to pass both focused tests and the real-interface Canary.
