# Phase-1 Requirement Audit

Audit target: `feature/cross-camera-domain-adaptation-v2`

Requirement baseline: `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`

External review baselines:

- review of `89e0aa942ec5d41e4a4a950001c5400cb5964c1a`;
- re-review of `133aafca080d91cb669ecc7b9a8285f340b78f04`.

## Decision

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
FULL_PGT_NOT_IMPLEMENTED
PHASE2_BLOCKED
```

All implementation-level findings from both external reviews have been remediated. Real cross-device effectiveness remains unverified because the captured 50-pair set and independent source/target holdouts were not executed in this repository session.

## Requirement-by-requirement check

| ID | Frozen requirement | Implementation evidence | Status |
|---|---|---|---|
| R1 | Do not supervise an unobservable Samsung-equivalent linear input | `phase1_protocol.py` refines input-side parameters through `FrozenSamsungTM`; student output is compared with native Samsung teacher output | PASS |
| R2 | Samsung GT qualifies the teacher instead of replacing it | Source P75/P90 profile and per-pair qualification precede distillation; rejected teachers produce no pair target | PASS |
| R3 | Canonicalization owns deterministic exposure/AWB/color operations | Train, evaluate and real inference construct `DeviceCanonicalizer(config.canonicalization)` | PASS |
| R4 | Bind actual canonicalization to provenance | Artifact schema 2 stores canonicalization payload and SHA; evaluation/inference reject mismatches | PASS |
| R5 | Limit Adapter capacity for 50-pair data | Bounded gain + near-identity 3x3 + six-point monotonic luminance curve | PASS |
| R6 | Enforce exactly 40 development and 10 locked pairs | Strict manifest loading rejects count and scene-group overlap violations | PASS |
| R7 | Protect dataset role and independence | Formal loaders validate device roles, non-empty and bound IDs, non-negative tensors, source uniqueness, pair uniqueness and development/locked content disjointness | PASS |
| R8 | Do not trust alignment labels without numeric evidence | Frozen policy treats declared quality as an upper bound and downgrades unsupported ROI/low-frequency claims | PASS |
| R9 | Perform true scene-group five-fold validation | Every fold computes targets, normalization, PCA and weighted ridge only from training scene groups | PASS |
| R10 | Use WLS only as initialization, then optimize through frozen TM | Pair solver performs bounded initialization and observable-output refinement through frozen Samsung TM | PASS |
| R11 | Prefer a low-dimensional predictor | Deterministic fold-local PCA plus teacher-weighted ridge with unregularized global bias | PASS |
| R12 | Require fold/prevalence/bootstrap/locked acceptance | 4/5 folds, 30/40 prevalence, positive bootstrap lower bound and positive locked median are mandatory | PASS |
| R13 | Keep locked data out of runtime-policy calibration | Support and minimum Adapter-margin thresholds are calibrated from the 40 development pairs only | PASS |
| R14 | Validate global tone, ROI and highlight separately | Locked report stores separate medians and blocks highlight regression or insufficient ROI coverage | PASS |
| R15 | Keep Samsung behavior frozen | Frozen wrapper disables parameter gradients and the backbone state hash is checked before/after training | PASS |
| R16 | Persist a provenance-bound artifact | Schema-2 artifact binds model/source/calibration/config/policy/support/margin/validation identities | PASS |
| R17 | Reject non-finite artifact policy | Artifact and alignment-policy loaders reject NaN/Inf/negative thresholds; runtime support and margin measurements must also be finite | PASS |
| R18 | Reject synthetic artifacts in real execution and evaluation | `real-run` and `evaluate-phase1` require `artifact.data_mode == real`; evaluation success also requires real calibration acceptance | PASS |
| R19 | Freeze calibration support policy | Development-only thresholds are artifact-bound and no CLI widening option exists | PASS |
| R20 | Reject Adapter boundary saturation | Real inference rejects non-finite, non-positive or below-calibrated parameter margin | PASS |
| R21 | Separate calibration acceptance from real effectiveness | Manifest uses three independent evidence fields; unavailable source/target evidence remains false | PASS |
| R22 | Expose one authoritative training entry | `phase1_training.py` contains only shared primitives and artifact operations; `phase1_protocol.train_phase1` is the sole training implementation | PASS |
| R23 | Keep Phase 2 outside the delivery at every entry point | Real config rejects Phase 2/pixel routing and `CrossCameraPipeline.run()` rejects any non-synthetic Phase-2 call before candidate generation | PASS |
| R24 | Provide executable Phase-1 chain | CLI provides train, evaluate and real inference with output/provenance artifacts | PASS |
| R25 | Verify focused and repository-wide behavior | GitHub Actions compiles, runs 66 focused tests and 97 full-repository tests, validates configs and exercises the Samsung checkpoint interface | PASS |

## Second-review finding closure

| Finding | Resolution |
|---|---|
| Programmatic API bypasses real Phase-2 block | `CrossCameraPipeline.run()` raises `PHASE2_NOT_IMPLEMENTED` whenever `phase2_enabled` and `synthetic=False` |
| Locked pairs calibrate deployment margin | `_calibrated_margin_threshold()` selects exactly the 40 development pairs before preparation/evaluation |
| NaN/Inf thresholds bypass fail-closed | Artifact, alignment policy and runtime measurements require `math.isfinite()` |
| `evaluate-phase1` accepts synthetic artifact | Both CLI and evaluation helper require a real artifact; success requires `real_phase1_calibration_accepted` |
| Formal manifest loader is weak | Device roles, metadata/tensor binding, non-negative tensors, unique content and cross-split disjointness are enforced |
| Old leaking training function remains in source | Function removed from source; reload cannot restore it |
| CI covers only cross-camera tests | Workflow now runs focused and full repository test discovery and retains both logs |

## Exact-head verification

```text
HEAD: ba9f9df1457d07ef28c46f09306f53bd24f07fbb
GitHub Actions run: 30011106271
Result: PASS
```

```text
compileall: PASS
focused cross-camera tests: 66 PASS
full repository tests: 97 PASS
configuration validation: PASS
Samsung checkpoint interface canary: PASS
```

## Items intentionally not claimed

Repository-only tests cannot establish real 40-pair out-of-fold improvement, real ten-pair locked improvement, independent source replay, independent iPhone target good-case/failure/OOD behavior, valid real Phase-2 activation, or complete Qwen/InternVL/Ovis/pixel pseudo-supervision behavior.

These remain data-backed or separately scoped acceptance work, not synthetic PASS results.
