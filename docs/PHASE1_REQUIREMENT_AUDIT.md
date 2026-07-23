# Phase-1 Requirement Audit

Audit target: `feature/cross-camera-domain-adaptation-v2`

Requirement baseline: `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`

External review baseline: review of `89e0aa942ec5d41e4a4a950001c5400cb5964c1a`

## Decision

```text
PHASE1_CODE_READY
REAL_PHASE1_ACCEPTANCE_PENDING
FULL_PGT_NOT_IMPLEMENTED
PHASE2_BLOCKED
```

The implementation-level Phase-1 findings from the external review have been remediated. Real cross-device effectiveness is still not established because the captured source/calibration manifests and independent target holdouts were not available in this repository session.

## Requirement-by-requirement check

| ID | Frozen requirement | Implementation evidence | Status |
|---|---|---|---|
| R1 | Do not supervise an unobservable Samsung-equivalent linear input | `phase1_protocol.py` refines pair parameters through `FrozenSamsungTM`; student output is compared with native Samsung teacher output | PASS |
| R2 | Samsung GT qualifies the teacher instead of replacing it | Source P75/P90 profile and per-pair qualification precede distillation; rejected teachers produce no pair target | PASS |
| R3 | Canonicalization owns deterministic exposure/AWB/color operations | The CLI constructs `DeviceCanonicalizer(config.canonicalization)` for train/evaluate/real-run | PASS |
| R4 | Bind the actual canonicalization configuration to provenance | Artifact schema 2 stores canonicalization payload and SHA; evaluation/inference reject mismatches | PASS |
| R5 | Limit the Adapter for 50-pair data | Bounded gain + near-identity 3x3 + six-point monotonic luminance curve; no Phase-1 spatial U-Net or arbitrary local color transform | PASS |
| R6 | Enforce exactly 40 development and 10 locked pairs | Manifest loading rejects count or scene-group overlap violations | PASS |
| R7 | Do not trust alignment labels without numeric evidence | `AlignmentPolicy` treats the declared level as an upper bound and programmatically downgrades unsupported ROI/low-frequency claims | PASS |
| R8 | Perform true scene-group five-fold validation | Every fold computes targets, normalization, PCA and weighted ridge only from training scene groups | PASS |
| R9 | Use input-side WLS only as initialization, then optimize through frozen TM | Pair solver performs bounded initialization and observable-output refinement through the frozen model | PASS |
| R10 | Prefer a small-sample low-dimensional predictor | Deterministic fold-local PCA plus teacher-weighted ridge with an unregularized global bias | PASS |
| R11 | Require fold/prevalence/bootstrap/locked acceptance | 4/5 folds, 30/40 prevalence, positive bootstrap lower bound and positive locked median are mandatory | PASS |
| R12 | Validate global tone, ROI and highlight separately | Locked report stores separate medians and blocks highlight regression or insufficient ROI coverage | PASS |
| R13 | Keep Samsung behavior frozen/non-regressing | Frozen wrapper disables gradients and state hash is checked before/after training | PASS |
| R14 | Persist a provenance-bound artifact | Schema-2 artifact binds model/source/calibration/config/policy/support/margin/validation identities | PASS |
| R15 | Reject synthetic artifacts in real execution | `real-run` requires `artifact.data_mode == real`; CLI no longer exposes a data-mode override | PASS |
| R16 | Freeze calibration support policy | Threshold is calibrated from leave-one-scene-group-out development distances and stored in the artifact; no CLI widening option exists | PASS |
| R17 | Reject Adapter boundary saturation | Real inference rejects non-positive or below-calibrated parameter-bound margin | PASS |
| R18 | Separate calibration acceptance from real effectiveness | Manifest uses `real_phase1_calibration_accepted`, `real_source_replay_verified`, and `real_target_effectiveness_verified`; unavailable evidence remains false | PASS |
| R19 | Expose only one supported training entry | Supported callers use `phase1_protocol.train_phase1`; the obsolete leaky public symbol is removed from the package module namespace | PASS |
| R20 | Keep Phase 2 outside the delivery | Real config rejects `phase2.enabled=true` and real pixel routing; experimental interfaces remain synthetic-only | PASS |
| R21 | Provide executable Phase-1 chain | CLI provides train, evaluate and real inference with output/provenance artifacts | PASS |
| R22 | Verify with independent CI | GitHub Actions compiles, runs 57 cross-camera tests, validates configs and exercises the included Samsung checkpoint interface | PASS |

## External review finding closure

| Finding | Resolution |
|---|---|
| Phase-2 critical evidence hard-coded | Real Phase 2 and pixel routing are now rejected before execution; experimental pipeline is not production-routable |
| `real_data_effectiveness_verified` false positive | Removed from hardened real manifest and replaced by three independent evidence fields |
| YAML canonicalization parsed but unused | One config-derived canonicalizer is used consistently and its hash is artifact-bound |
| Synthetic artifact accepted by real-run | Real inference explicitly rejects synthetic artifacts |
| Runtime support threshold can be widened | CLI override removed; artifact-frozen threshold is authoritative |
| Adapter parameter saturation only logged | Saturation or insufficient margin now blocks output |
| Alignment quality disconnected from evidence | Frozen numeric policy preserves or downgrades the declaration; it never upgrades it |
| Second public leaky training path | Obsolete public symbol removed; `phase1_protocol` is the only supported training entry |
| Real Phase-2/VLM/geometry stack incomplete | Explicitly deferred to `IQA-PGT`; not represented as a delivered feature |

## Items intentionally not claimed

Repository-only tests cannot establish:

1. real 40-pair out-of-fold prevalence;
2. real ten-pair locked global/ROI/highlight improvement;
3. independent Samsung source replay non-regression metrics;
4. independent iPhone target good-case, failure and OOD holdout behavior;
5. whether Phase 2 should activate on real target data;
6. complete Qwen/InternVL/Ovis/pixel pseudo-supervision behavior.

Those remain data-backed or separately scoped acceptance work, not synthetic PASS results.
