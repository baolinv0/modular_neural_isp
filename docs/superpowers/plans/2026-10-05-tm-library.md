# TM Algorithm Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. User explicitly selected separate implementers and review before delivery; implementation is authorized.

**Goal:** Runnable, reviewed A–E TM library integrated with original Modular Neural ISP, including data loading, training, evaluation and inference.

**Architecture:** Preserve original pipeline and checkpoints. Add interchangeable local renderers, a shared input/conditioning contract, and isolated experiment tooling. Keep scientific claims separate from execution evidence.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV/Pillow, YAML, pytest.

**Spec:** docs/superpowers/specs/2026-10-05-tm-library-design.md

## Global Constraints

- Base commit e0f82e9932b560a53a63d5eaf1a85ceefe4c91d3; branch feat/tm-algorithm-library. Preserve original files unless a small integration/documentation link is necessary.
- Shared worktree: /workspace/scratch/91d9efd2196d/tm-library-worktree. Python: /workspace/scratch/91d9efd2196d/tm-library-venv/bin/python. Use torch.set_num_threads(1) in CPU tests.
- Workers own disjoint files, never git add/commit others' changes, never push or merge. Root coordinates commits to avoid shared-index races.
- Source license remains applicable. No downloaded training datasets/weights beyond original repository contents, and no external paid API calls.
- Use specification interfaces exactly; propose interface changes to coordinator before editing shared contracts.

## Review Focus

- Scene/camera or reference leakage: separate paired supervision and unaligned reference roles, validate train/val separation.
- Missing semantics vs labelled background: loss masks and zero-confidence controls must be correct.
- Color and precision: no accidental gamma/uint16 conversion or Bayer-as-RGB interpretation.
- Baseline parity and checkpoints: original path exact, resume reproduces uninterrupted CPU training.
- Algorithm fidelity: D really uses pyramids, E separately controls detail, C gain bounded, B bilateral slicing, A monotonic expert curves.

## Task 1 — Foundation and upstream integration

Files: tm_library/{__init__,config,common,model}.py; tm_library/operators/__init__.py; tests/tm_library/test_core.py. Implement validated ModelConfig, ConditionEncoder/helpers, deferred operator registry, upstream wrapper/loading and semantic modes. Tests first for config, upstream equality, mask gating and devices. Operator imports may be lazy until tasks 3–7 land.

## Task 2 — Data and reproducible fixtures

Files: tm_library/{data,prepare_data,synthetic}.py; tests/tm_library/test_data.py. Implement spec dataset and CLI builders, precision-safe IO, synchronized transformations and metadata conversion. Tests first for explicit path pairs, uint16, camera RGB, invalid labels/ranges, tiny/odd images and deterministic fixtures. Synthetic output must contain usable train/val manifests.

## Tasks 3–7 — Independent operators A, B, C, D, E

Each task owns its operator module and tests/tm_library/test_<algorithm>.py only; uses common/config produced by Task 1. Implement exact signature and mechanism specified in design. Include differentiability, shape, edge cases, meaningful parameter sensitivity, and defining invariant tests. Files: region_curves.py; spatial_grid.py; gain_residual.py; exposure_fusion.py; base_detail.py. Do not write substitute placeholders while common helpers are pending.

## Task 8 — Training, evaluation and inference

Files: tm_library/{train,evaluate,infer,losses,metrics,engine}.py; tests/tm_library/test_training.py; configs/tm_library/*.yaml; requirements-tm-library.txt. Consume ModelConfig/TMModel/PairedImageDataset; implement actual optimization, sem auxiliary supervision, resume, validation selection, metrics and saved outputs. Smoke all five algorithms plus upstream baseline. Inference uses stored config and never needs GT. Independent post-implementation review before completion.

## Task 9 — Integration docs, review and delivery

Files: tm_library/README.md; docs/tm_library_review.md; root README.md link; .github/workflows/tm-library-tests.yml if suitable. Document exact commands, IO domains, A–E mechanisms and limitations, semantic modes, sources, hardware caveats, and evidence. Run complete focused tests and command-level smoke. Assign independent algorithm and training/data reviews, fix actionable findings and scoped re-review. Commit and push feature branch, open draft PR; no merge.

## Acceptance commands

- Python -m pytest tests/tm_library -q
- Python -m tm_library.synthetic --output /tmp/tm-fixture (exact implemented CLI documented)
- Python -m tm_library.train --config configs/tm_library/<algorithm>.yaml with fixture paths/output overrides.
- Python -m tm_library.evaluate --checkpoint ... --manifest ... --output ...
- Python -m tm_library.infer --checkpoint ... --manifest ... --output ...

Passing synthetic tests verifies executable behavior and differentiability, not real-camera IQ or SoC throughput.
