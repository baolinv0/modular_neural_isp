# Phase-1 Requirement Audit

Audit target: `feature/cross-camera-domain-adaptation-v2`

Requirement baseline: `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`

## Decision

**Architecture and executable Phase-1 code: ALIGNED**

**Real 50-pair empirical acceptance: NOT EXECUTED IN THIS REPOSITORY SESSION**

The second statement is an evidence boundary, not an implementation substitution. No synthetic result is treated as proof of real cross-device effectiveness.

## Requirement-by-requirement check

| ID | Frozen requirement | Implementation evidence | Status |
|---|---|---|---|
| R1 | Do not supervise an unobservable Samsung-equivalent linear input | `phase1_protocol.py` optimizes pair parameters through `FrozenSamsungTM`; student output is compared with the native Samsung teacher output | PASS |
| R2 | Samsung GT qualifies the teacher instead of replacing it | Samsung source P75/P90 profile and per-pair teacher qualification are computed before pair distillation; rejected teachers produce no pair target | PASS |
| R3 | Canonicalization owns deterministic exposure/AWB/color operations | `canonicalization.py` performs white/exposure normalization, comparable AWB alignment and declared common-space CCM with independent confidence fields | PASS |
| R4 | Limit the Adapter for 50-pair data | `TargetCameraAdapter` is bounded gain + near-identity 3x3 + six-point monotonic luminance curve; no spatial U-Net or arbitrary local color transform is used in Phase 1 | PASS |
| R5 | Enforce exactly 40 development and 10 locked pairs | `load_calibration_manifest` rejects any other count and rejects scene-group overlap between development and locked data | PASS |
| R6 | Use alignment quality to restrict legal losses | `AlignmentEvidence.enabled_losses` maps scene-only, ROI and low-frequency quality to progressively stronger legal losses; required masks are validated | PASS |
| R7 | Perform true scene-group five-fold validation | Each fold computes pair targets, normalization, PCA basis and weighted-ridge coefficients from its training groups only; validation groups are never fitted in that fold | PASS |
| R8 | Use input-side WLS only as initialization, then optimize through frozen TM | `ObservablePairSolver.initialize` fits bounded gain/matrix/curve; `refine` back-propagates observable output loss through the frozen Samsung model | PASS |
| R9 | Prefer a small-sample low-dimensional predictor | Final `z -> theta` mapping uses deterministic fold-local PCA features and teacher-weighted ridge with an unregularized global bias | PASS |
| R10 | Require fold/prevalence/bootstrap/locked acceptance | Phase-1 report requires 4/5 positive folds, 30/40 out-of-fold improvements, positive bootstrap lower bound and positive locked median | PASS |
| R11 | Validate global tone, ROI and highlight separately | Locked evaluation stores separate median improvements for global tone, ROI and highlight; highlight regression and missing ROI coverage block acceptance | PASS |
| R12 | Keep Samsung behavior frozen/non-regressing | `FrozenSamsungTM` disables parameter gradients; a state-dict SHA is checked before and after Phase-1 training. Because the target Adapter is not inserted in the Samsung source branch, unchanged backbone state implies source replay output is unchanged | PASS |
| R13 | Persist a usable and provenance-bound artifact | Artifact contains model, source manifest, calibration manifest, feature schema, support geometry, training config and validation evidence; model and calibration mismatches fail closed | PASS |
| R14 | Provide an executable Phase-1 chain | CLI provides `train-phase1`, `evaluate-phase1` and `real-run`; output tensor and run manifest are persisted | PASS |
| R15 | Reject unsupported target inputs | Inference computes calibration-support distance from the artifact's development support and fails closed above the configured bound | PASS |
| R16 | Keep Phase 2 outside the first delivery | Default and real-example configurations set `phase2.enabled=false`; `real-run` records Phase 2 as blocked | PASS |
| R17 | Keep synthetic evidence honest | Synthetic fixtures and canary set `real_data_effectiveness_verified=false` | PASS |
| R18 | Verify with independent CI | GitHub Actions compiles code, runs all cross-camera tests, validates both configs and exercises strict load/forward/input-gradient behavior with the included Samsung checkpoint | PASS |

## Items intentionally not claimed

The following require the actual captured data and cannot be established by repository-only synthetic tests:

1. the real 40-pair out-of-fold improvement prevalence;
2. the real ten-pair locked global/ROI/highlight improvements;
3. scene coverage sufficiency of the captured 50 pairs;
4. visual Samsung-style fidelity on independent iPhone target holdout images;
5. whether Phase 2 should activate on real target data.

Until those data-backed checks pass, the correct system state is:

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
PHASE2_BLOCKED
```
