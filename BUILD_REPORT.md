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
Verified HEAD: 4af77ceaea5e38d1b42ac828d61de1472e2146ab
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

## Implemented Phase-1 chain

```text
strict source/calibration manifests
→ numeric alignment-evidence downgrade
→ config-bound device canonicalization
→ Samsung teacher qualification
→ fold-local pair refinement through frozen Samsung TM
→ fold-local PCA + teacher-weighted ridge predictor
→ locked-holdout acceptance
→ schema-2 hardened artifact
→ evaluate-phase1
→ fail-closed real-run
```

## Second-review remediation

### API-level Phase-2 isolation

`CrossCameraPipeline.run()` rejects every non-synthetic Phase-2 request before candidate generation:

```text
phase2_enabled=true + synthetic=false
→ PHASE2_NOT_IMPLEMENTED
```

Real config and pixel-routing guards remain active. Experimental Phase-2 interfaces cannot emit real parameter, range, preference or pixel supervision.

### Holdout-independent runtime policy

Both runtime thresholds are calibrated from the 40 development pairs only:

- maximum calibration-support distance;
- minimum Adapter parameter-bound margin.

The ten locked pairs evaluate the frozen system but do not set deployment policy.

### Finite-value enforcement

Artifact thresholds, alignment thresholds and runtime support/margin measurements reject NaN, Inf and invalid negatives.

### Real evaluation identity

`evaluate-phase1` requires a real artifact and successful real Phase-1 calibration acceptance. A sealed synthetic artifact cannot be presented as a real evaluation result.

### Strict formal data contracts

Source and calibration loaders enforce device roles, metadata/tensor binding, non-empty and unique IDs, finite non-negative tensors, source uniqueness, pair uniqueness and development/locked exact-content disjointness.

### Single training implementation

The obsolete leaking `phase1_training.train_phase1` function has been removed from source. `phase1_training.py` contains shared primitives and artifact operations only. The sole authoritative implementation is `phase1_protocol.train_phase1`.

## Verification

Exact verified head:

```text
4af77ceaea5e38d1b42ac828d61de1472e2146ab
```

GitHub Actions run:

```text
30010973315: PASS
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
