# Cross-Camera TM Requirement Baseline

Status: **FROZEN**

This document is the delivery baseline for `feature/cross-camera-domain-adaptation-v2`. When implementation notes or older reports conflict with it, this document takes precedence.

## 1. Purpose

Use a low-capacity `TargetCameraAdapter` to make an iPhone BLC+WB linear-RGB input reproduce the **observable output behavior** of the frozen Samsung photofinishing model on a same-scene Samsung input.

The target is not an unobservable Samsung-equivalent linear image.

For a calibration pair:

```text
Samsung teacher: O_s = F_s(C_s(X_s))
iPhone student:  O_i = F_s(A_t(C_i(X_i)))
Phase-1 target:  O_i ≈ O_s
```

`F_s` is frozen. Samsung GT qualifies `O_s` and measures the source model residual; it does not replace the teacher output.

## 2. Available data

- Samsung paired source data: Samsung linear input and Samsung GT.
- Cross-device calibration data `D_c`: exactly 50 balanced same-scene iPhone/Samsung pairs, approximately aligned, with Samsung GT.
- Target data `D_t`: unpaired iPhone inputs and metadata. It cannot replace `D_c` for Phase-1 training.

Calibration split:

```text
40 development pairs
10 locked holdout pairs
```

The locked scene groups must not appear in development.

## 3. Canonicalization boundary

Canonicalization handles only deterministic or metadata-supported operations:

1. white-level normalization;
2. exposure-scale normalization;
3. reliable-range estimation;
4. alignment/undoing of known applied AWB operations when coordinates are comparable;
5. conversion through a reliable CCM to a declared common linear color space.

Do not directly apply Samsung/iPhone WB-gain ratios when the two devices remain in different sensor-RGB coordinates. Remaining spectral/color-response differences belong to a bounded near-identity Adapter.

## 4. TargetCameraAdapter capacity

The Phase-1 Adapter is limited to:

```text
bounded residual per-channel gain
+ near-identity 3x3 transform
+ six-point monotonic luminance curve
```

All parameters use identity initialization and explicit bounds. Metadata uncertainty moves the output toward identity.

The following are outside the Phase-1 MVP:

- U-Net or other high-capacity spatial adapters;
- high-resolution gain maps;
- arbitrary local color transforms;
- adversarial domain alignment;
- Samsung backbone fine-tuning.

## 5. Approximate-pair supervision

Every calibration pair records:

- overlap;
- forward-backward alignment consistency;
- valid ROI fraction;
- residual displacement;
- maximum legal loss level.

Loss use is monotonic with alignment quality:

| Alignment quality | Legal losses |
|---|---|
| Same scene only | global tone/statistics |
| Reliable ROI correspondence | tone + ROI statistics |
| Valid low-frequency registration | tone + ROI + masked low-frequency map |
| Approximate pairs | no full-resolution pixel loss |

The mandatory tone loss includes log-luminance quantiles, highlight headroom, clipping and global/local contrast. Adapter regularization is always enabled.

## 6. Teacher qualification

First evaluate:

```text
O_s = F_s(C_s(X_s)) against Samsung GT
```

- all critical metrics at or below source P75 and no hard defect: qualified;
- P75-P90 residual: downweighted;
- any metric above P90 or hard defect: rejected from Adapter distillation.

Rejected samples remain useful for the Samsung source residual profile. The Adapter must not distort iPhone input to reproduce or compensate a known Samsung-model failure.

## 7. Training and validation

For each usable development pair:

1. initialize bounded gain and near-identity matrix by input-side weighted least squares;
2. initialize a monotonic luminance curve;
3. jointly refine those low-dimensional parameters through the frozen Samsung TM using observable output losses;
4. fit a tiny `z -> theta` predictor from interpretable metadata/image statistics.

Development validation is genuine scene-group five-fold cross-validation:

- each fold trains a new predictor;
- validation scene groups do not participate in pair-parameter targets, normalization or predictor fitting for that fold;
- all variants of the same scene stay in one fold.

Acceptance requires:

- at least 4/5 folds with positive median improvement;
- at least 30/40 development pairs improved out-of-fold;
- paired-bootstrap lower bound above zero;
- positive locked-holdout median improvement;
- no Adapter boundary saturation;
- sufficient qualified teacher coverage;
- Samsung source replay non-regression when real source evaluation is supplied.

The final ten pairs may be opened only after Adapter capacity, features and losses are frozen.

## 8. Required executable chain

The first deliverable must implement:

```text
source/calibration manifest loading
→ teacher qualification
→ scene-group cross-validation
→ final Adapter artifact
→ locked-holdout evaluation
→ artifact load
→ real iPhone inference
→ output + provenance manifest
```

The artifact is bound to Samsung model, source manifest, calibration manifest, feature schema and validation report hashes/identities. A failed Phase-1 artifact cannot run inference.

## 9. Phase 2 boundary

Phase 2 is not the current delivery target. It remains disabled by default and is blocked unless Phase 1 passes on real locked data.

The current branch may retain tested Phase-2 interfaces, but it must not claim Phase-2 effectiveness, pixel pseudo-GT validity, or a complete self-evolution loop.

The next allowed Phase-2 experiment is only:

```text
global_underexposure
+ source-grounded direction
+ L1 parameter supervision
+ low-capacity luminance residual
```

L2, L3, VLM arbiters and pixel routing are deferred and cannot block Phase-1 delivery.

## 10. Evidence labels

Synthetic fixtures prove mechanics only. They must set:

```text
synthetic = true
real_data_effectiveness_verified = false
```

Real effectiveness can be claimed only after executing the frozen 50-pair calibration protocol and independent source/target holdouts. Tests or reports must never convert unavailable evidence into PASS.
