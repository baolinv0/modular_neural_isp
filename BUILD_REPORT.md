# BUILD_REPORT

## Status

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
FULL_PGT_NOT_IMPLEMENTED
PHASE2_BLOCKED
```

The Phase-1 implementation now includes the fail-closed and provenance remediation identified by the external review of head `89e0aa942ec5d41e4a4a950001c5400cb5964c1a`.

Real cross-device effectiveness is not declared because the actual captured 50-pair calibration set and independent source/target holdouts were not available in this repository session.

## Repository baseline

- Repository: `baolinv0/modular_neural_isp`
- Branch: `feature/cross-camera-domain-adaptation-v2`
- Integration base: `feature/modular-capture-pipeline@efbfdfea87385254cb36e23354fa9b7f1ae2e4ce`
- Draft PR: `#6 Cross-camera Phase 1 observable-output remediation`

## Frozen objective

```text
Samsung teacher: O_s = F_s(C_s(X_s))
iPhone student:  O_i = F_s(A_t(C_i(X_i)))
Training target:  O_i approximately equals O_s
```

The Adapter does not reconstruct an unobservable Samsung sensor-linear target. Samsung GT qualifies the Samsung teacher and calibrates source residuals.

## Implemented Phase-1 chain

```text
source/calibration manifest loading
→ numeric alignment-evidence downgrade
→ config-bound device canonicalization
→ Samsung teacher qualification
→ fold-local pair-parameter refinement through frozen Samsung TM
→ fold-local PCA + teacher-weighted ridge predictor
→ locked-holdout acceptance
→ schema-2 hardened artifact
→ evaluate-phase1
→ fail-closed real-run
```

## External review remediation

### 1. Phase-2 scope isolation

Real configuration now rejects:

```text
phase2.enabled=true       → PHASE2_NOT_IMPLEMENTED
pixel_route_enabled=true  → PIXEL_ROUTING_NOT_IMPLEMENTED
```

Existing L1/L2/L3/VLM interfaces remain experimental synthetic-only interfaces. They are not treated as a delivered Open-Source Pseudo-Supervision Pipeline.

### 2. Honest evidence labels

The hardened run manifest no longer derives a broad real-effectiveness claim from calibration acceptance.

It records independent fields:

```text
real_phase1_calibration_accepted
real_source_replay_verified
real_target_effectiveness_verified
```

Only the first can be set by the frozen 40+10 calibration protocol. The other two remain false until their corresponding evaluations are actually executed.

### 3. Canonicalization provenance

Train, evaluate and real inference all construct:

```python
DeviceCanonicalizer(config.canonicalization)
```

Artifact schema 2 stores the canonicalization payload and SHA. Loading and inference reject a mismatch between runtime config and the artifact.

### 4. Real artifact identity

- `train-phase1` in real mode always writes `data_mode=real`.
- The old CLI data-mode override was removed.
- `real-run` rejects synthetic artifacts.
- Samsung model and calibration manifest identities remain fail-closed.

### 5. Frozen calibration support

The maximum support distance is calibrated from leave-one-scene-group-out development feature distances and stored in the artifact.

The old runtime `--max-support-distance` override was removed. Operators cannot widen the frozen support boundary.

### 6. Adapter parameter support

The artifact stores a calibrated minimum parameter-bound margin. Inference rejects:

```text
margin <= 0
or
margin < artifact.minimum_parameter_bound_margin
```

A saturated prediction cannot silently produce a formal output.

### 7. Alignment evidence

The manifest quality label is only an upper bound. A frozen numeric policy checks overlap, valid ROI coverage, forward-backward consistency and residual displacement.

Unsupported low-frequency claims are downgraded to ROI or scene-only supervision. Numeric evidence never upgrades a weaker declaration.

### 8. Single supported training entry

`phase1_protocol.train_phase1` is the authoritative supported training path. The obsolete pre-fold target-fitting symbol is removed from the public package module namespace so callers cannot silently select the leaking implementation.

## Hardened artifact contents

Schema 2 binds:

- Samsung checkpoint SHA;
- source manifest SHA;
- calibration manifest SHA;
- feature schema and normalization;
- Adapter state;
- training and locked validation evidence;
- canonicalization payload and SHA;
- alignment policy and SHA;
- calibration support geometry;
- maximum support distance;
- minimum Adapter parameter-bound margin;
- separated real-evidence labels.

## Tests added

New regression coverage verifies:

- real Phase 2 and pixel routing are rejected;
- weak numeric alignment evidence is downgraded;
- synthetic artifacts cannot enter real inference;
- calibration acceptance does not claim target effectiveness;
- canonicalization mismatch is rejected;
- support boundary cannot be overridden;
- Adapter saturation blocks output;
- the obsolete public training entry is unavailable;
- schema-1 training output is sealed, reloaded and executed as a hash-bound schema-2 artifact.

## Verification

Latest implementation verification before this report update:

```text
GitHub Actions run: 29985795562
Head: a8b8fb2bfdc3bdcd224a9130d8e784ac202ef1e3
compileall: PASS
cross-camera tests: 57 PASS
configuration validation: PASS
Samsung checkpoint strict load/forward/frozen-gradient/input-gradient canary: PASS
```

The workflow retained the complete unittest log as an artifact.

## Remaining evidence limitation

The repository is code-ready for the frozen real-data protocol, but it has not executed the actual captured data. Therefore none of the following are claimed:

- real target improvement;
- independent source replay verification;
- target good-case/OOD safety;
- Phase-2 activation validity;
- complete IQA-PGT/VLM/pixel pseudo-supervision delivery.

The next valid action is to prepare the actual source/calibration manifests, run the frozen 40+10 protocol once, and preserve all outputs without changing capacity, features, losses or thresholds after the locked set is opened.
