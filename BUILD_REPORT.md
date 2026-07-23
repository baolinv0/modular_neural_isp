# BUILD_REPORT

## Status

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
FULL_PGT_NOT_IMPLEMENTED
PHASE2_BLOCKED
```

The Phase-1 implementation now includes both rounds of external fail-closed, provenance and holdout-independence remediation.

Real cross-device effectiveness is not declared because the captured 50-pair dataset and independent source/target holdouts were not executed in this repository session.

## Repository

```text
Repository: baolinv0/modular_neural_isp
Branch: feature/cross-camera-domain-adaptation-v2
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

The following reject NaN/Inf and invalid negatives:

- artifact support threshold;
- artifact minimum parameter margin;
- alignment displacement threshold;
- runtime support distance;
- runtime Adapter margin.

### Real evaluation identity

`evaluate-phase1` requires:

```text
artifact.data_mode == real
real_phase1_calibration_accepted == true
```

A sealed synthetic artifact cannot be presented as a real evaluation result.

### Strict formal data contracts

Source loading enforces:

- Samsung device role;
- metadata/sample/tensor binding;
- non-empty and unique IDs;
- finite non-negative tensors;
- unique Samsung source content;
- minimum source scene diversity.

Calibration loading enforces:

- iPhone/Samsung device roles;
- metadata/tensor binding;
- unique pair and metadata IDs;
- finite non-negative tensors;
- unique pair content signatures;
- development/locked scene and exact-content disjointness.

### Single training implementation

The obsolete leaking `phase1_training.train_phase1` function has been removed from source. `phase1_training.py` contains shared primitives and artifact operations only. The sole authoritative training implementation is:

```text
phase1_protocol.train_phase1
```

Reload, IDE discovery and static analysis can no longer recover a second training path.

## Verification

Code verification head:

```text
f247f0c5e6968a001b3bd407bcfe955d1580441d
```

GitHub Actions run:

```text
30010621410: PASS
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

Workflow artifact `cross-camera-test-logs` retains both complete unittest logs.

## Evidence boundary

The repository is ready to execute the frozen real-data protocol, but it has not established:

- real 40-pair out-of-fold prevalence;
- real ten-pair locked global/ROI/highlight improvement;
- independent Samsung source replay metrics;
- independent iPhone good-case/failure/OOD behavior;
- valid activation of real Phase 2;
- complete IQA-PGT/VLM/pixel pseudo-supervision delivery.

The next valid action is an independent Reviewer re-verification of the exact branch head. Only after Reviewer PASS should the frozen 40+10 experiment or final Evaluator begin.
