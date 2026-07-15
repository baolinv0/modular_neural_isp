# TM Baseline Evolution MVP — Implementation Plan

**Goal:** Add the candidate selection, teacher manifest, policy/fixed-baseline distillation, rendering and held-out adjudication missing from the existing adjustTM closed loop.

## Tasks

- [x] Define candidate, teacher and distribution contracts.
- [x] Merge adjustTM inference provenance with candidate-level IQA scores.
- [x] Implement fail-closed teacher selection with baseline anchors.
- [x] Freeze deterministic scene-disjoint train/validation/test splits.
- [x] Implement interpretable weighted-ridge alpha policy with OOD fallback.
- [x] Integrate policy rendering with existing `MethodRunner` interfaces.
- [x] Implement fixed Gain/GTM baseline checkpoint distillation.
- [x] Verify frozen parameter zero-drift and checkpoint reloadability.
- [x] Implement held-out target/anchor/hard-failure adjudication.
- [x] Track teacher and final evaluator provenance.
- [x] Add standalone CLI, documentation and focused tests.
