# BUILD_REPORT

## Status

`IMPLEMENTED` for the bounded, mechanically verifiable MVP. Real cross-device effectiveness remains unverified because the frozen 50-pair calibration set and real iPhone holdouts are not present in this environment.

## Baseline

- Repository: `baolinv0/modular_neural_isp`
- Development branch: `feature/cross-camera-domain-adaptation-v2`
- Baseline branch: `feature/modular-capture-pipeline`
- Baseline commit: `efbfdfea87385254cb36e23354fa9b7f1ae2e4ce`
- Initial working tree: clean
- Initial test result: 31 tests passed, zero failures/skips
- IQA-PGT reference: `baolinv0/IQA-PGT`, `v1@83b040d5430f4c953e06368e39c823a5eb793848`
- IQA reference: `baolinv0/IQA`, `fix/tmqa-p1-remediation-v0.2.0@1d73bc0fd14bcffb30feff49471e6c9d2f05cb45`
- Local Samsung checkpoint exercised: `photofinishing/models/photofinishing_s24-style-0.pth`, SHA-256 `5137b1a9da936814544a0259add95530e124d954fac8e08ece61333be630c09f`

## CURRENT STATE before modification

The repository contained a runnable modular capture pipeline, adapters for the repository photofinishing model, 31 passing tests, and included Samsung photofinishing checkpoints. It did not contain a cross-camera calibration contract, target-domain adapter, source residual estimator, pseudo-supervision teachers, projection/certification/router, cross-camera CLI, canary, or lineage manifest.

No CUDA device, real iPhone data, 50-pair calibration set, trained target adapter, or local Qwen/InternVL/Ovis checkpoint was available. The included Samsung checkpoint was available and could be loaded on CPU.

## GAPS addressed

- No observable direction anchor: added Samsung-teacher distillation, Samsung-GT teacher qualification, source residual profiles, and source/locked validation gates.
- VLM EV bottleneck: VLM diagnosis is semantic-only and cannot emit numerical correction strength.
- Overlapping teachers: L1 is global; L2 is ROI-conditioned with feathered spatial correction; L3 is a local proposal that must be projected.
- Unsafe or ungrounded pixel targets: raw generated images are ineligible, projection creates a new hash, and projected outputs receive all 13 gates again.
- Hard-coded, unproven gates: profiles can be calibrated from source distributions and bind dataset/profile hashes; synthetic defaults are labeled and cannot authorize real pixel supervision.
- Additive uncertainty: replaced by typed, independent fields and a necessary-condition route table.
- Single perceptual arbiter: added local-only InternVL AB/BA and optional Ovis interfaces; neither can accept a candidate.
- Missing data/lineage controls: added strict metadata/source/calibration/target/config/manifest contracts and canonical tensor hashes.
- Missing executable verification: added deterministic CLI canary, real checkpoint interface canary, and one unified script.

## IMPLEMENTATION SCOPE

### Implemented

1. Strict BLC+WB 16-bit linear input, source, exact 40+10 calibration, target, config, manifest and lineage contracts.
2. Deterministic canonicalization: white level, known comparable applied-AWB gain alignment, common CCM, bounded exposure prior, masks and independent confidences.
3. Identity-initialized low-capacity `TargetCameraAdapter`: gain → near-identity 3x3 matrix → six-point monotonic luminance curve.
4. Per-pair WLS/isotonic initialization, refinement through a frozen Samsung TM, tiny `z -> theta` predictor fitting, teacher P75/P90 qualification, and 40/10 validation criteria.
5. Source residual profile, fixed interpretable `psi`, OOD-aware estimator, dynamic global/face ROI, standardized vector DirectionAlignment, and Phase 2 activation criteria.
6. Distinct L1/L2/L3 orchestration, local-only open-source policy and real adapter interfaces for Qwen3-VL, Qwen-Image-Edit, InternVL and optional Ovis.
7. Pre-projection safety, TM-space projection, 69-scalar luminance-only `TMResidualAdapter`, source identity/non-regression losses.
8. Thirteen non-compensable certification gates, distribution-derived threshold profiles, structured uncertainty, supervision routing, manifests and lineage.
9. CLI, deterministic Synthetic Canary, real-mode fail-closed preflight, real Samsung checkpoint load/forward/gradient canary, unified verification.

### Not implemented

1. Full AWB re-estimation, black-level correction, 3A, full color-style transfer, denoising, deblurring, super-resolution, multi-frame HDR reconstruction, online/RL adaptation, or Samsung backbone retraining.
2. Real 50-pair calibration training/evaluation, because the data is unavailable.
3. Execution of real Qwen3-VL/Qwen-Image-Edit/InternVL/Ovis weights, because local checkpoints and GPU are unavailable.
4. Production real-run artifact loading beyond strict preflight; it deliberately stops instead of substituting an untrained adapter or synthetic profile.

## Modified files

- `cross_camera_tm/`: contracts, canonicalization, adapters, Phase 1 fitting/validation, residual profiles, teachers, local-model policy, projection, residual adapter, certification, routing, lineage, manifest, configuration, pipeline, canary and CLI.
- `tests/test_cross_camera_*.py`: behavioral, failure-path, deterministic and integration tests.
- `configs/cross_camera_tm_v2.yaml`: strict synthetic-canary configuration with pixel routing disabled.
- `main/run_cross_camera_adaptation.py`: CLI entrypoint.
- `scripts/`: unified verification, compatibility wrapper, canary comparator, and real Samsung checkpoint canary.
- `docs/superpowers/`: frozen design and implementation plan.
- `docs/PSEUDO_SUPERVISION_IMPLEMENTATION_REPORT.md`: implementation and evidence report.
- `capture_pipeline/README.md`: bounded integration and commands.
- `.gitignore`: generated Python/test/run artifacts.

## Tests added or updated

- Strict field, hash, split and upstream-state rejection.
- Canonicalization ordering, confidence and masks.
- Adapter identity/capacity/order/monotonicity/confidence; per-pair initialization/refinement; predictor fitting; Phase 1 validation.
- Fixed `psi`, source support/OOD, dynamic ROI, DirectionAlignment and activation.
- L1/L2 distinction, L3 proposal-only policy, projection/color/detail behavior, bounded residual adapter, local/open-source enforcement.
- All 13 gates, unavailable/non-compensation, raw injection, real-profile restriction, structured routes, manifest and lineage.
- Strict config, deterministic end-to-end canary, Phase 2 disabled path, L3 projection/re-certification fallback, CLI files and real-run missing-profile rejection.

## Validation executed

The final fresh commands and exact results are recorded after implementation in `docs/PSEUDO_SUPERVISION_IMPLEMENTATION_REPORT.md`. The unified command is:

```bash
scripts/run_cross_camera_domain_adaptation_verification.sh
```

It runs compile checks, all old and new unittests, strict config validation, two independent Synthetic Canary runs and byte-semantic comparison, manifest/output validation, and strict load/forward/input-gradient checks against the included Samsung checkpoint.

Fresh post-documentation result: exit `0`; compile `PASS`; 77 tests `OK` with zero skip/xfail; deterministic canary `PASS` with route `parameter`; real checkpoint interface canary `PASS`; output validation `PASS`; real-data effectiveness explicitly `UNVERIFIED`.

## Known limitations

- Mechanical tests do not demonstrate Samsung-style improvement on real iPhone images.
- The source direction/profile and gate thresholds must be fitted from independent real Samsung/calibration data before real routing.
- The 50-pair data sufficiency, scene coverage and approximate-registration quality remain empirical questions.
- Incomplete phone HDR metadata limits deterministic exposure canonicalization; low confidence keeps the residual near identity.
- Local VLM/editor/arbiter interfaces are implemented but only explicitly labeled deterministic doubles were mechanically exercised.
- Pixel supervision is disabled in the default MVP. Real pixel routing additionally requires a non-synthetic calibrated profile, projection, full re-certification and every necessary condition.

## Requirement mapping

| Requirement | Code | Evidence |
|---|---|---|
| Strict data contract | `contracts.py`, `config.py`, `manifest.py` | contract/config/manifest tests |
| Open-source-only/local-only | `policy.py` | remote/closed rejection tests |
| Qwen3-VL diagnosis | `Qwen3VLDiagnosisAdapter` | semantic-only/no-strength test |
| L1/L2/L3 orchestration | `teachers.py`, `pipeline.py` | distinction and L3 fallback integration tests |
| Qwen-Image-Edit local adapter | `policy.py` | explicitly labeled local interface-double test |
| Pre-projection safety and TM projection | `projection.py` | malformed proposal and projection tests |
| Projected target full certification | `certification.py`, `pipeline.py` | L3 fallback re-certifies; 13-gate tests |
| InternVL anonymous AB and Ovis inspection | `policy.py`, `pipeline.py` | typed local interfaces; neither authorizes acceptance |
| Consistency/uncertainty/router | `routing.py` | no additive score; route necessary-condition tests |
| CLI/manifest/lineage | `cli.py`, `manifest.py`, `lineage.py` | CLI output and lineage tests |
| Synthetic Canary/unified verification | `canary.py`, `scripts/` | deterministic double-run and real checkpoint interface canary |

## Builder boundary

This report is Builder evidence, not final acceptance. Reviewer and Evaluator decisions are independent artifacts produced by their assigned agents.
