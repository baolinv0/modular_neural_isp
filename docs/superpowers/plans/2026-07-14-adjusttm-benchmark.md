# adjustTM Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible, cache-driven benchmark for learned brightness controllers, simple baselines, VLM review, blinded human review, statistics, and reporting.

**Architecture:** Pure, independently tested core modules implement manifest validation, transforms, calibration, metrics, statistics, VLM schemas, and human-study design. Thin CLI modules compose these cores and lazily import model code so analytical tests do not require real checkpoints.

**Tech Stack:** Python 3.11, PyTorch, NumPy, OpenCV, pytest; optional openpyxl for XLSX.

## Global Constraints

- Only `a_000` is real-camera GT.
- Flux endpoints and Retinex intermediate levels are reported as teacher fidelity.
- Global exposure/gamma parameters use calibration scenes only.
- Oracle parameters may use test GT but never enter main ranking.
- All statistical inference uses scene as the independent unit.
- VLM target-match and naturalness are separate outputs.
- No single weighted overall score.

---

### Task 1: Protocol, manifest, and cache identities

**Files:** create `adjustTM/benchmark/schemas.py`, `adjustTM/benchmark/build_manifest.py`; test `adjustTM/tests/test_benchmark.py`.

- [ ] Write failing tests for level provenance, file validation, SHA-256 stability, and protocol hash mismatch.
- [ ] Run the focused tests and confirm missing-symbol failures.
- [ ] Implement immutable protocol/manifest helpers and CLI.
- [ ] Re-run focused and package tests.
- [ ] Commit.

### Task 2: Exposure and luminance-gamma baselines

**Files:** create `adjustTM/benchmark/transforms.py`, `adjustTM/benchmark/baselines.py`, `adjustTM/benchmark/calibrate_baselines.py`, `adjustTM/benchmark/generate_oracles.py`.

- [ ] Write failing tests for identity anchors, chromaticity-preserving gamma, grid refinement, monotonic parameter projection, and no test leakage.
- [ ] Confirm failures.
- [ ] Implement transforms, one-dimensional search, global calibration, and per-image oracle search.
- [ ] Re-run tests.
- [ ] Commit.

### Task 3: Reference and dense-control metrics

**Files:** create `adjustTM/benchmark/metrics.py`, `adjustTM/benchmark/evaluate_reference.py`, `adjustTM/benchmark/evaluate_control.py`.

- [ ] Write failing tests for GT semantic grouping, scene-first aggregation, monotonic violations, dead zones, jumps, smoothness, and range balance.
- [ ] Confirm failures.
- [ ] Implement metrics and JSONL CLIs.
- [ ] Re-run tests.
- [ ] Commit.

### Task 4: Cached output generation

**Files:** create `adjustTM/benchmark/methods.py`, `adjustTM/benchmark/generate_outputs.py`.

- [ ] Write failing tests using synthetic method runners for output paths, cache hash rejection, alpha-zero checks, and dense records.
- [ ] Confirm failures.
- [ ] Implement runner registry, lazy learned-model loading, lossless PNG writing, and dense scalar records.
- [ ] Re-run tests.
- [ ] Commit.

### Task 5: VLM task protocol

**Files:** create `adjustTM/benchmark/vlm.py`, `adjustTM/benchmark/evaluate_vlm.py`.

- [ ] Write failing tests for separate prompts, strict score schema, retries, median aggregation, and unstable-judgment flags.
- [ ] Confirm failures.
- [ ] Implement task export and command-backend protocol without hard-coding one vendor model.
- [ ] Re-run tests.
- [ ] Commit.

### Task 6: Blinded human study

**Files:** create `adjustTM/benchmark/human_study.py`, `adjustTM/benchmark/build_human_study.py`, `adjustTM/benchmark/analyze_human_study.py`.

- [ ] Write failing tests for deterministic blinding, balanced candidate exposure, response QC, pairwise conversion, and Bradley-Terry scores.
- [ ] Confirm failures.
- [ ] Implement builders and analysis CLIs.
- [ ] Re-run tests.
- [ ] Commit.

### Task 7: Statistical comparison and reports

**Files:** create `adjustTM/benchmark/statistics.py`, `adjustTM/benchmark/compare_methods.py`, `adjustTM/benchmark/report.py`, `adjustTM/benchmark/run.py`.

- [ ] Write failing tests for scene bootstrap, CVaR, paired permutation, Holm correction, main/diagnostic separation, and report tables.
- [ ] Confirm failures.
- [ ] Implement comparison engine, HTML/CSV/XLSX reporting, and resumable stage runner.
- [ ] Re-run tests.
- [ ] Commit.

### Task 8: Documentation and CI

**Files:** modify `adjustTM/README.md`, `.github/workflows/adjusttm-tests.yml`; create `adjustTM/benchmark/README.md`.

- [ ] Add CLI smoke tests and documentation checks.
- [ ] Run compileall and all adjustTM tests.
- [ ] Verify the requirement checklist against the design.
- [ ] Commit and upload all files to the feature branch.

## Execution Status — 2026-07-14

Implemented all planned benchmark stages, focused tests, documentation, sample configurations, cache identity checks, continuous simple-baseline interpolation, alpha-zero gates, semantic GT grouping, dense-control metrics, command-based VLM integration, blinded static human-study UI, scene-level statistics, paired baseline comparisons, HTML/CSV/XLSX reporting, and resumable orchestration.
