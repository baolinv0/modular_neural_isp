# TM Library v2 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. The user explicitly authorizes implementation, separate reviews and updating the existing GitHub delivery. Do not pause for another approval.

**Goal:** Correct the original architecture gaps while preserving a runnable research library, with real loading hooks for the user's segmentation model.

**Architecture:** Optional A/C correction over selectable full-scene base; staged prediction/rendering APIs; frozen segmentation provider outside the TM optimizer; matched and explicit experiment protocols.

**Tech Stack:** Existing Python/PyTorch/NumPy/OpenCV/YAML/pytest, no new large external model dependency.

**Spec:** docs/superpowers/specs/2026-10-05-tm-library-v2-design.md

## Global constraints

- Shared isolated worktree /workspace/scratch/91d9efd2196d/tm-library-worktree, branch feat/tm-algorithm-library, base866c873. All agents use disjoint ownership, never commit/push; root handles GitHub delivery.
- Python /workspace/scratch/91d9efd2196d/tm-library-venv/bin/python; compatibility Python /workspace/scratch/91d9efd2196d/tm-library-torch25-venv/bin/python. CPU threads1. No real data/GPU/user segmenter checkpoint supplied.
- Keep original photofinishing files/weights intact, do not alter unrelated feature branches. Code version2 rejects incompatible v1 library checkpoints explicitly; original weights remain supported.

## Review focus

- Correction strength/ROI zero must preserve selected anchor, including absent semantic labels.
- C zero initialization over original LTM must preserve original baseline; no accidental GTM-only substitution in composed mode.
- Staged/direct outputs and gradients agree; no full-resolution width feature tensor in B or conditioning concat.
- S1 infer must succeed after deleting segmentation weight file; S2 must actually load/run it when requested.
- Split policy, target alignment, model preprocessing and class mapping errors fail clearly; resume model/source identity cannot silently change.

## Tasks and ownership

1. Foundation: config/common/model and test_core + new test_composition. Implement configv2, staged APIs, composition, ROI/strength, fixed-reference post controls, small-first encoder, legacy guards.
2. A/C/E: three operator modules and their tests. Add predict/render APIs; A highlight shoulder/sequential expert rendering; C preserve EV semantics; E spatial base controls.
3. B/D: two operator modules and their tests. Efficient affine guidance, staged API, diagnostics switch, maintain grid/pyramid invariants.
4. Semantics: new semantics.py and test_semantics. Real TorchScript/factory loading, preprocessing/class groups, provider modes and frozen behavior.
5. Data/metrics: data/prepare_data/synthetic/metrics and test_data + new test_metrics. Scene metadata, meaningful semantic fixtures, extra IQ diagnostics, preserve IO invariants.
6. Training/integration: engine/train/evaluate/infer/losses, test_training, configs and requirements if needed. Wire v2+provider, strict splits, matched objectives, resume, CLI and matrix validation. Coordinate signatures early.
7. Root documentation/evidence and fresh independent reviewers after implementations. Implementers fix findings; scoped re-review; full tests on both torchversions, actual CLI fixture runs with loaded semantic model, commit/update PR14.

For each worker: write meaningful failing regression before correction, implement within ownership, run scoped suite, report assumptions/changed interfaces/results. Full suite is root's integration gate. Do not rewrite another owner's file without coordination.
