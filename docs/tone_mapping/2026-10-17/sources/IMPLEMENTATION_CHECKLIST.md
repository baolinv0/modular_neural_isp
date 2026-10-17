# Multi-Model Pseudo-GT Single-Scene Probe v1.1 — IMPLEMENTATION CHECKLIST

## A. Scope and Governance

- [ ] Read and freeze `CONTRACT.md`.
- [ ] Confirm this task does not train or fine-tune a TM model.
- [ ] Confirm no result will be described as real GT.
- [ ] Confirm the final decision is limited to `GO`, `PROBE`, or `KILL_OR_PIVOT`.
- [ ] Confirm Boogu visual-reference mode is not assumed.
- [ ] Confirm no manual best-of-N selection is allowed.
- [ ] Confirm Raw failures and rejected outputs remain evidence.

## B. Input Preparation

- [ ] Add `target.png` and `reference.png`.
- [ ] Verify same scene and similar physical illumination.
- [ ] Document the target failure and why the reference is better.
- [ ] Add target/reference subject and background masks.
- [ ] Add optional face/highlight masks where applicable.
- [ ] Compute file hashes.
- [ ] Create `input_manifest.json`.

## C. Environment

- [ ] Create reproducible environment file.
- [ ] Verify GPU memory and disk requirements.
- [ ] Load credentials only through environment variables.
- [ ] Prevent API keys from entering logs.
- [ ] Record package, CUDA, and driver versions.

## D. Model Capability Probe

### Core Models

- [ ] GPT exact model ID recorded; single-image edit checked; ordered two-image support verified or rejected.
- [ ] Gemini exact model ID recorded; single-image edit checked; ordered two-image support verified or rejected.
- [ ] FLUX exact endpoint recorded; `prompt_upsampling=false`; ordered two-image support verified or rejected.
- [ ] Qwen exact model/revision recorded; seed control verified; ordered two-image support verified or rejected.

### Boogu

- [ ] Boogu Edit exact model/revision recorded.
- [ ] 1K single-image edit verified.
- [ ] `steps=50`, `guidance=5.0`, `seed=42`.
- [ ] Reference mode marked unsupported unless explicitly proven.
- [ ] Boogu Turbo exact model/revision recorded.
- [ ] Named Turbo revision verified or marked `UNVERIFIED_REVISION`.
- [ ] `steps=4`, `guidance=1.0`, `seed=42`.
- [ ] Reference mode marked unsupported unless explicitly proven.

### Capability Evidence

- [ ] Write `metrics/capabilities.json`.
- [ ] Unsupported modes return structured status.
- [ ] No collage workaround exists.

## E. Prompt Freeze

- [ ] Save no-reference prompt.
- [ ] Save same-scene reference prompt.
- [ ] Save Boogu Chinese prompt.
- [ ] Record prompt hashes.
- [ ] Prevent per-model prompt optimization.
- [ ] Preserve actual provider-side prompt when observable.

## F. Generation

### Core No-Reference

- [ ] GPT result saved.
- [ ] Gemini result saved.
- [ ] FLUX result saved.
- [ ] Qwen result saved.

### Core Reference

- [ ] GPT result or structured unsupported status saved.
- [ ] Gemini result or structured unsupported status saved.
- [ ] FLUX result or structured unsupported status saved.
- [ ] Qwen result or structured unsupported status saved.

### Boogu

- [ ] Boogu Edit no-reference Raw saved.
- [ ] Boogu Turbo no-reference Raw saved.
- [ ] No Boogu native visual-reference output generated in v1.1.

### Integrity

- [ ] Exactly one first-attempt result per model/mode.
- [ ] Retries use separate attempt IDs.
- [ ] No first attempt is overwritten.
- [ ] No output is manually retouched.
- [ ] No hidden best-of-N selection occurred.
- [ ] Every request has metadata and status.

## G. Baselines

### B1

- [ ] Subject mask used.
- [ ] Maximum +0.5 EV lift.
- [ ] Highlight protection and edge-aware feathering.
- [ ] Color, texture, and geometry preserved.
- [ ] Background-stability unit test passes.

### B2

- [ ] Uses semantic-region statistics.
- [ ] Does not depend on pixel alignment.
- [ ] Transfer strength 0.5.
- [ ] Maximum ±0.75 EV.
- [ ] Highlight protection implemented.
- [ ] Target color and texture preserved.

## H. Tone-Only Projection

- [ ] Conservative global alignment only.
- [ ] No dense warping to hide regeneration.
- [ ] Linear sRGB and luminance equation implemented.
- [ ] Log2 luminance used.
- [ ] Gaussian sigma ratio 0.02.
- [ ] EV delta clamp ±1.0.
- [ ] Alpha 0.5.
- [ ] Target chromaticity preserved.
- [ ] New clipping ratio computed; threshold 0.005.
- [ ] Every successful Raw has a Tone-only output.
- [ ] Parameters are identical across models.
- [ ] Projection is deterministic.

## I. Metrics

### Global

- [ ] p10, p25, p50, p75, p90, p95, p99.
- [ ] shadow occupancy.
- [ ] highlight occupancy.
- [ ] p99–p10 dynamic range.

### Semantic

- [ ] subject mean/median luminance.
- [ ] background mean/median luminance.
- [ ] subject/background ratio.
- [ ] optional face/global ratio.
- [ ] region-level highlight occupancy.
- [ ] reference statistics computed independently.

### Content Diagnostics

- [ ] face identity.
- [ ] pose/silhouette.
- [ ] object addition/removal.
- [ ] text corruption.
- [ ] background geometry.
- [ ] light-source and shadow structure.
- [ ] color/skin-tone drift.
- [ ] high-frequency energy change.
- [ ] sharpening/denoising/texture synthesis warning.

### Reference Metrics

- [ ] No target-reference PSNR/SSIM/LPIPS/L1/L2 used as GT evidence.
- [ ] Reference utilization evaluated semantically and distributionally.

### Outputs

- [ ] `image_metrics.json`.
- [ ] `summary.csv`.

## J. Blind Review

- [ ] Anonymous IDs generated.
- [ ] Model mapping stored separately.
- [ ] Raw no-reference sheet.
- [ ] Tone-only no-reference sheet.
- [ ] Raw reference sheet.
- [ ] Tone-only reference sheet.
- [ ] Baseline comparison sheet.
- [ ] Review CSV generated.
- [ ] At least three reviewers, or confidence marked LOW.
- [ ] Individual reviews preserved.
- [ ] Median aggregation computed.
- [ ] Raw and Tone-only reviewed separately.

## K. Hard Gates

- [ ] Target improvement recorded.
- [ ] Content preservation recorded.
- [ ] Non-target stability recorded.
- [ ] Tone extractability recorded.
- [ ] Reference utilization recorded where applicable.
- [ ] Reference-copy protection recorded where applicable.
- [ ] Universal gate decided.
- [ ] Reference gate decided where applicable.
- [ ] Immediate-rejection reasons checked.
- [ ] No weighted score overrides a failed gate.
- [ ] `gate_results.json` generated.

## L. Required Comparisons

- [ ] Core no-reference comparison.
- [ ] Per-model reference versus no-reference comparison.
- [ ] Raw versus Tone-only comparison.
- [ ] No-reference models versus B1.
- [ ] Reference models versus B2.
- [ ] Boogu Edit versus core no-reference models.
- [ ] Boogu Edit versus B1.
- [ ] Boogu Edit versus Boogu Turbo.
- [ ] Benefit attributed to Tone, color, texture/restoration, or regeneration.

## M. Decision

### Core

- [ ] Select `GO`, `PROBE`, or `KILL_OR_PIVOT`.
- [ ] Cite hard-gate and baseline evidence.
- [ ] State whether reference mode adds value.
- [ ] State whether model directions are consistent.

### Boogu

- [ ] Select `BOOGU_GO`, `BOOGU_PROBE`, or `BOOGU_KILL`.
- [ ] State whether Tone-only retained benefit.
- [ ] State whether Boogu beat B1.
- [ ] State Edit versus Turbo tradeoff.
- [ ] State whether a three-seed follow-up is justified.

## N. Tests

### Unit

- [ ] config validation.
- [ ] adapter capability handling.
- [ ] unsupported reference mode.
- [ ] immutable first attempt.
- [ ] exact projection parameters.
- [ ] clipping computation.
- [ ] feature order.
- [ ] mask validation.
- [ ] baseline behavior.
- [ ] hard-gate logic.

### Integration

- [ ] synthetic target/reference fixture.
- [ ] deterministic fake adapter.
- [ ] full pipeline to report.
- [ ] Boogu single-image-only behavior.
- [ ] missing credential handling.
- [ ] one failed model does not block others.

### Evidence

- [ ] focused tests saved.
- [ ] full regression saved.
- [ ] CLI smoke test saved.
- [ ] final commit recorded.
- [ ] generated artifacts excluded from source commits unless intentional.

## O. Final Deliverables

- [ ] `CONTRACT.md`.
- [ ] `PLAN.md`.
- [ ] `IMPLEMENTATION_CHECKLIST.md`.
- [ ] frozen `experiment.yaml`.
- [ ] input manifest.
- [ ] target/reference images and masks.
- [ ] capability matrix.
- [ ] all Raw outputs or structured failures.
- [ ] all Tone-only outputs.
- [ ] B1 and B2.
- [ ] metadata and logs.
- [ ] metrics and blind review.
- [ ] contact sheets.
- [ ] gate results.
- [ ] `REPORT.md`.
- [ ] branch, commit, and reproducible run instructions.

## P. Final Non-Claims

The report must explicitly state:

- [ ] one scene only;
- [ ] no TM model was trained;
- [ ] pseudo-GT training benefit is not established;
- [ ] cross-scene generalization is not established;
- [ ] production readiness is not established;
- [ ] unsupported modes were not fabricated;
- [ ] Boogu native visual-reference editing was not tested in v1.1.
