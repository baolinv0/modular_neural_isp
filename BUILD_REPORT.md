# BUILD_REPORT

## Status

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
PHASE2_BLOCKED
```

The Phase-1 architecture and executable code now match the frozen requirements in `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`.

Real cross-device effectiveness is not declared because the actual 50-pair calibration manifests and independent target holdouts were not available in this repository session.

## Baseline

- Repository: `baolinv0/modular_neural_isp`
- Branch: `feature/cross-camera-domain-adaptation-v2`
- Original branch head before remediation: `240c21b3e0069a2e9c1d85491d10816801ecae69`
- Integration base: `feature/modular-capture-pipeline@efbfdfea87385254cb36e23354fa9b7f1ae2e4ce`
- Draft PR: `#6 Cross-camera Phase 1 observable-output remediation`

## Corrected requirement

The Adapter is not trained to reconstruct an unobservable Samsung linear input.

```text
Samsung teacher: O_s = F_s(C_s(X_s))
iPhone student:  O_i = F_s(A_t(C_i(X_i)))
Training target:  O_i approximately equals O_s
```

Samsung GT is used to qualify the Samsung teacher and calibrate the source residual distribution. A Samsung sample on which the frozen model clearly fails cannot supervise the iPhone Adapter.

## Implemented

### 1. Frozen requirement and data contracts

- Frozen Phase-1 baseline: `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`.
- Strict Samsung source and 50-pair calibration manifests.
- Exact 40-development/10-locked split.
- Scene-group separation between development and locked data.
- Canonical tensor hashes and strict metadata JSON booleans.
- Per-pair overlap, forward-backward consistency, valid ROI fraction and residual displacement.

### 2. Device canonicalization

- White-level normalization.
- Exposure-scale normalization.
- Reliable-range and highlight masks.
- Metadata-supported applied-AWB alignment only when coordinate definitions are comparable.
- Optional declared CCM into a common linear color space.
- Missing or unreliable metadata lowers confidence rather than inventing a correction.

### 3. Low-capacity TargetCameraAdapter

```text
bounded residual channel gain
+ near-identity 3x3 transform
+ six-point monotonic luminance curve
```

No Phase-1 U-Net, high-resolution gain map, adversarial alignment or arbitrary local color transform was introduced.

### 4. Observable-output pair fitting

- Input-side weighted least-squares gain and near-identity matrix initialization.
- Low-dimensional monotonic curve initialization.
- Joint refinement through the frozen Samsung TM.
- Mandatory global tone loss including log-luminance quantiles, highlight headroom, clipping and contrast.
- ROI and low-frequency losses enabled only when alignment evidence permits them.
- Teacher P75/P90 qualification and downweight/reject behavior.

### 5. True group-aware cross-validation

Each of five folds independently computes:

```text
fold training pair targets
→ fold normalization
→ fold PCA basis
→ teacher-weighted ridge z-to-theta predictor
→ unseen scene-group validation
```

Validation groups do not participate in pair-target fitting, normalization, PCA or ridge fitting for that fold.

The final artifact is trained on all 40 development pairs only after out-of-fold evaluation has been computed.

### 6. Acceptance evidence

Phase-1 acceptance requires:

- at least four positive folds;
- at least 30/40 development pairs improved out of fold;
- bootstrap lower bound above zero;
- positive locked median improvement;
- positive locked global-tone and ROI medians;
- no locked highlight regression;
- no Adapter boundary saturation;
- sufficient qualified-teacher coverage;
- unchanged Samsung backbone state hash.

Because the TargetCameraAdapter is not inserted in the Samsung source branch, an unchanged frozen backbone means Samsung source replay output is unchanged by Phase 1.

### 7. Artifact and real execution chain

The versioned artifact binds:

- Samsung checkpoint SHA;
- source manifest SHA;
- calibration manifest SHA;
- feature schema and normalization;
- calibration-support geometry;
- Adapter state;
- training configuration;
- locked validation evidence.

Executable commands:

```text
train-phase1
→ evaluate-phase1
→ real-run
```

`evaluate-phase1` rejects a calibration manifest that differs from the artifact's training manifest. `real-run` rejects a failed artifact, model mismatch, invalid input/metadata or input outside calibrated support.

### 8. Phase-2 boundary

- Phase 2 is disabled in the default configuration and real example configuration.
- Phase-1 inference records `phase2_status=blocked_until_separate_activation`.
- Existing Phase-2 components remain non-blocking interfaces and are not used to claim a complete self-evolution loop.

## Files added or materially changed

- `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`
- `docs/PHASE1_REQUIREMENT_AUDIT.md`
- `docs/PHASE1_DATA_AND_RUNBOOK.md`
- `cross_camera_tm/phase1_data.py`
- `cross_camera_tm/phase1_training.py`
- `cross_camera_tm/phase1_protocol.py`
- `cross_camera_tm/cli.py`
- `cross_camera_tm/config.py`
- `configs/cross_camera_tm_v2.yaml`
- `configs/cross_camera_tm_v2.real.example.yaml`
- `tests/test_cross_camera_phase1_real_mvp.py`
- `tests/test_cross_camera_pipeline_cli.py`
- `.github/workflows/cross-camera-phase1.yml`

## Verification

GitHub Actions workflow: `Cross-camera Phase 1`.

The latest code verification before this report-only update completed successfully with:

```text
compileall: PASS
cross-camera tests: 49 PASS
synthetic/default config validation: PASS
real example config validation: PASS
included Samsung checkpoint strict load: PASS
Samsung checkpoint finite forward: PASS
frozen model parameter-gradient check: PASS
input-gradient canary: PASS
```

The workflow retains the complete unittest log as an artifact.

## Requirement audit

The full requirement-by-requirement result is recorded in:

```text
docs/PHASE1_REQUIREMENT_AUDIT.md
```

All architecture and executable-code requirements are marked PASS.

## Known evidence limitation

Repository tests use deterministic synthetic calibration data to prove mechanics, leakage prevention, artifact behavior and fail-closed paths. They do not prove the result on the actual captured 50 pairs.

The next valid action is to prepare the real source/calibration manifests using `docs/PHASE1_DATA_AND_RUNBOOK.md`, run the frozen 40+10 protocol once, and retain the resulting reports without changing capacity, features, losses or thresholds after opening the locked set.
