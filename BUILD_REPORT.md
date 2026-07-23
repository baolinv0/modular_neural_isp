# BUILD_REPORT

## Status

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
FULL_PGT_NOT_IMPLEMENTED
PHASE2_BLOCKED
```

The Phase-1 implementation includes both rounds of external fail-closed, provenance and holdout-independence remediation.

Real cross-device effectiveness is not declared because the captured 50-pair dataset and independent source/target holdouts were not executed in this repository session.

## Repository

```text
Repository: baolinv0/modular_neural_isp
Branch: feature/cross-camera-domain-adaptation-v2
Code verification HEAD: 29e7dcb30af855433171cbecc103cbb523a45e0c
Base: feature/modular-capture-pipeline@efbfdfea87385254cb36e23354fa9b7f1ae2e4ce
Draft PR: #6
```

## Frozen objective

```text
Samsung teacher: O_s = F_s(C_s(X_s))
iPhone student:  O_i = F_s(A_t(C_i(X_i)))
Training target:  O_i approximately equals O_s
```

The Adapter does not reconstruct an unobservable Samsung sensor-linear target. Samsung GT qualifies the Samsung teacher and calibrates source residuals.

## Second-review remediation

- `CrossCameraPipeline.run()` rejects non-synthetic Phase 2 before candidate generation.
- Support and minimum Adapter-margin thresholds use the 40 development pairs only.
- Artifact, policy and runtime thresholds reject NaN/Inf and invalid negatives.
- `evaluate-phase1` requires a real artifact and real calibration acceptance.
- Formal loaders enforce device roles, metadata/tensor binding, non-negative tensors, unique content and development/locked disjointness.
- The obsolete leaking `phase1_training.train_phase1` function is removed from source; `phase1_protocol.train_phase1` is the sole implementation.
- CI runs both focused cross-camera and full repository regression.

## Stable canary correction

The original synthetic integration test mixed protocol validation with randomly optimized pair targets and could flip near acceptance boundaries across CI hardware. Production thresholds were not changed. The test now supplies the analytically correct fixed 1.25x pair transform while retaining fold-local call inspection, artifact serialization, locked evaluation and inference checks. Pair-solver optimization remains covered by independent unit tests.

## Verification

Exact verified head:

```text
29e7dcb30af855433171cbecc103cbb523a45e0c
```

GitHub Actions run:

```text
30011756321: PASS
```

Evidence:

```text
compileall: PASS
focused cross-camera tests: 66 PASS
full repository tests: 97 PASS
synthetic/default config validation: PASS
real example config validation: PASS
Samsung checkpoint strict load/forward/frozen-gradient/input-gradient canary: PASS
```

The workflow retains both complete unittest logs as an artifact.

## Evidence boundary

The repository is ready for independent Reviewer re-verification, but it has not established real 40+10 effectiveness, independent source replay, independent target good-case/failure/OOD safety, valid real Phase-2 activation, or complete IQA-PGT/VLM/pixel pseudo-supervision delivery.

Only after Reviewer PASS should the frozen real-data experiment or final Evaluator begin.
