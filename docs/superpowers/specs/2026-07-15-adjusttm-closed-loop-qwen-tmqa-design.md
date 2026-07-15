# adjustTM Closed Loop with Qwen-TMQA — Design Specification

## Goal

Add an isolated `adjustTM_closed_loop/` package that turns the existing controllable-brightness TM experiment and the independently maintained Qwen-TMQA evaluator into one reproducible V1 engineering loop.

## Scope

The V1 system validates nine-level GT data, prepares only the `param_residual` training handoff, evaluates Dataset V1 or generated model trajectories with Qwen-TMQA, normalizes failures into F1–F8, and emits a training-data prescription. It must not modify the existing `adjustTM/` model, training, benchmark, or test code.

## Architecture

`adjustTM_closed_loop` is a thin orchestration and diagnosis layer. Original training remains under `adjustTM`; Qwen-TMQA remains an independently versioned CLI installed from the verified archive in `baolinv0/IQA`. Integration occurs through explicit commands, JSON/YAML artifacts, and normalized scene-level results.

## Components

1. Dataset Gate: complete nine-level layout, uint16 linear source, spatial consistency, monotonicity, endpoint range, clipping, shadow, and chroma checks; outputs clean/boundary/invalid.
2. adjustTM Handoff: frozen `param_residual` command with batch-size and scope constraints.
3. Qwen-TMQA Adapter: runtime config with source conditioning, optional Qwen3-VL primary and InternVL arbiter, CLI execution, result normalization.
4. Failure Taxonomy: F1 brightness-under, F2 brightness-over, F3 clipping, F4 shadow crush, F5 chroma drift, F6 control curve, F7 regional inconsistency, F8 structural artifacts.
5. Data Compiler: positive, boundary, hard-negative, and regression-anchor scene sets plus supervision and acceptance gates.

## Safety Boundaries

- No automatic model-weight update.
- No opaque combined score is used to approve training changes.
- Evaluator and model remain independently versioned.
- Invalid data never enters supervised training.
- Target-slice gains must be checked against KEEP-scene regression anchors.
- The system does not claim that image metadata proves the linear source is physically calibrated; it only enforces the declared uint16 interface.
