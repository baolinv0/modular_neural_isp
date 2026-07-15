# adjustTM Closed Loop with Qwen-TMQA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Dataset V1 → param-residual handoff → Qwen-TMQA → failure taxonomy → training-data prescription loop.

**Architecture:** Keep `adjustTM/` and Qwen-TMQA independently versioned. Add a thin Python package that validates data, builds commands, normalizes evaluation artifacts, and compiles actionable data tasks.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, PyYAML, pytest, existing `adjustTM`, external `qwen-tmqa` CLI.

## Global Constraints

- Do not modify existing `adjustTM/` production or benchmark files.
- First training method is exactly `param_residual`.
- Batch size is a positive multiple of 18.
- Qwen-TMQA primary and arbiter are optional and use OpenAI-compatible endpoints.
- Invalid scenes are excluded; KEEP scenes become regression anchors.

---

### Task 1: Dataset Gate

Create `schemas.py` and `dataset_gate.py`; test complete layouts, uint16 source, monotonicity, clipping, and missing files.

### Task 2: Qwen-TMQA Integration

Create `qwen_tmqa_adapter.py` and `bootstrap_qwen_tmqa.py`; test command construction, source-conditioned runtime config, archive extraction, and scene-result normalization.

### Task 3: Failure Taxonomy and Data Compiler

Create `failure_taxonomy.py` and `data_prescription.py`; test F1–F8 mapping, positive/boundary/hard-negative sets, regression anchors, supervision, and acceptance gates.

### Task 4: Closed-Loop CLI and Handoff

Create `runner.py`, `cli.py`, and `__main__.py`; test frozen param-residual command and dry-run artifact creation.

### Task 5: Documentation and Verification

Add default Qwen-TMQA config, README, focused CI, compile checks, full tests, and a synthetic file-level canary.
