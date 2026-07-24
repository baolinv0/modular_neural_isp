# BUILD_REPORT

## Status

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
FULL_PGT_NOT_IMPLEMENTED
PHASE2_BLOCKED
```

The Phase-1 implementation includes all three rounds of external fail-closed, provenance, holdout-independence and validation-evidence remediation.

Real cross-device effectiveness is not declared because the captured 50-pair dataset and independent source/target holdouts were not executed in this repository session.

## Repository

```text
Repository: baolinv0/modular_neural_isp
Branch: feature/cross-camera-domain-adaptation-v2
Verified product-code HEAD: 719192750bb72c8d4c8d0447f29eea6d61208b30
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

## Remediation summary

- `CrossCameraPipeline.run()` rejects non-synthetic Phase 2 before candidate generation.
- Support and minimum Adapter-margin thresholds use the 40 development pairs only.
- Artifact, policy and runtime thresholds reject NaN/Inf and invalid negatives.
- `evaluate-phase1` requires a real artifact and real calibration acceptance.
- Formal loaders enforce device roles, metadata/tensor binding, non-negative tensors, unique content and development/locked disjointness.
- The obsolete leaking `phase1_training.train_phase1` function is removed from source; `phase1_protocol.train_phase1` is the sole implementation.
- Phase-1 pair targets, prepared inputs, predictor state, fold evidence, bootstrap evidence, locked evidence and artifact support geometry now fail closed on every non-finite value.
- The production `phase1_training.ObservablePairSolver` has a direct deterministic test for finite bounded parameters, monotonic curves and reduced observable teacher error.
- CI runs both focused cross-camera and full repository regression.

## Stable canary correction

The protocol integration test uses the analytically correct fixed 1.25x pair transform so protocol acceptance is hardware-stable. Production thresholds and production training behavior were not weakened. The production `ObservablePairSolver` optimization path is now covered directly by its own test.

## Verification

The successful workflow tested the PR merge candidate whose head component was:

```text
719192750bb72c8d4c8d0447f29eea6d61208b30
```

GitHub Actions run:

```text
30060023501: PASS
```

Evidence:

```text
compileall: PASS
focused cross-camera tests: 70 PASS
full repository tests: 101 PASS
synthetic/default config validation: PASS
real example config validation: PASS
Samsung checkpoint strict load/forward/frozen-gradient/input-gradient canary: PASS
```

The workflow retains both complete unittest logs as an artifact. Because pull-request workflows check out `refs/pull/6/merge`, this evidence is described as PR merge-candidate CI rather than an exact-head checkout.

## Evidence boundary

The repository is ready for independent Reviewer re-verification and, after Reviewer PASS, the frozen real 40+10 Phase-1 experiment. It has not established real 40+10 effectiveness, independent source replay, independent target good-case/failure/OOD safety, valid real Phase-2 activation, or complete IQA-PGT/VLM/pixel pseudo-supervision delivery.

The first locked-set result is one-shot evidence. If method capacity, features, losses, canonicalization, alignment policy or acceptance thresholds change after the locked set is opened, those ten pairs become development data and a new locked set is required.
