# Multi-Model Pseudo-GT Single-Scene Probe v1.1 — IMPLEMENTATION PLAN

## 1. Objective

Implement a reproducible single-scene experiment runner that compares:

- no-reference image editing;
- same-scene, non-pixel-aligned reference editing when natively supported;
- deterministic Tone-only projection;
- simple and reference-statistical Tone baselines;
- quantitative statistics and anonymous human review;
- a dedicated Boogu no-reference extension.

## 2. Required Repository Deliverables

Recommended modules:

```text
src/
├── adapters/
│   ├── base.py
│   ├── gpt_image_2.py
│   ├── gemini_image.py
│   ├── flux_2.py
│   ├── qwen_image_edit.py
│   └── boogu_image_edit.py
├── probe/
│   ├── schema.py
│   ├── capability_probe.py
│   ├── runner.py
│   ├── baselines.py
│   ├── alignment.py
│   ├── tone_projection.py
│   ├── metrics.py
│   ├── contact_sheet.py
│   └── report.py
└── cli/
    └── run_single_scene_probe.py
```

Exact structure may follow the existing repository, but responsibilities must remain separable.

## 3. Phase 0 — Environment and Capability Probe

### 3.1 Dependency Check

Verify:

- Python and GPU environment;
- local model dependencies;
- API credentials are referenced only through environment variables;
- disk space for model weights and outputs;
- required image codecs;
- reproducible environment lock file.

### 3.2 Per-Model Capability Probe

For each adapter:

1. initialize exact model/revision;
2. run one low-cost single-image smoke test;
3. verify output decoding;
4. test ordered two-image input where claimed;
5. record model ID, revision, reference support, seed support, prompt rewriting, output dimensions, latency, and error status.

Do not continue into an unsupported mode.

### 3.3 Capability Output

Create `metrics/capabilities.json` with structured statuses such as:

```json
{
  "boogu_image_edit": {
    "single_image_edit": true,
    "ordered_multi_image_edit": false,
    "seed_control": true,
    "status": "PASS_SINGLE_IMAGE_ONLY"
  }
}
```

## 4. Phase 1 — Input Preparation

### 4.1 Validate Target and Reference

Validation must check:

- files exist and decode;
- target/reference are not identical;
- dimensions and aspect ratios;
- masks match their respective images;
- masks are nonempty;
- target failure description is present in metadata.

### 4.2 Input Metadata

Create `inputs/input_manifest.json` with:

- scene ID;
- target and reference hashes;
- capture source/device when available;
- failure type;
- mask paths;
- reason reference is better;
- known target/reference differences.

## 5. Phase 2 — Frozen Configuration

Create `experiment.yaml` containing the exact model IDs, revisions, prompts, generation rules, projection parameters, and baseline parameters from `CONTRACT.md`.

The runner must save a resolved, frozen copy in the evidence package.

## 6. Phase 3 — Model Adapter Interface

Define one common interface:

```python
class ImageEditAdapter:
    def capabilities(self) -> dict:
        ...

    def edit_no_reference(self, target_path, prompt, output_path, metadata_path):
        ...

    def edit_with_reference(self, target_path, reference_path, prompt, output_path, metadata_path):
        ...
```

Rules:

- unsupported reference mode raises a typed exception;
- exceptions become structured metadata;
- adapters never select among hidden candidates;
- successful outputs are immutable;
- retries create new attempt records.

## 7. Phase 4 — Generation

### 7.1 Core Models

For each core model:

1. run no-reference edit;
2. run reference edit only after capability verification;
3. save Raw output;
4. save request metadata and errors.

### 7.2 Boogu

Run:

- Boogu Edit, no-reference;
- Boogu Edit Turbo, no-reference.

Do not run visual-reference mode in v1.1.

### 7.3 Required Metadata

Per output save:

- experiment ID;
- model ID and exact revision;
- mode;
- seed;
- prompt path and actual prompt;
- prompt rewrite status;
- input hashes;
- request/local run ID;
- generation parameters;
- output dimensions;
- status and error details.

## 8. Phase 5 — Traditional Baselines

### 8.1 `B1_SIMPLE_TONE`

- use target subject mask;
- maximum +0.5 EV lift;
- protect highlights;
- edge-aware feathering;
- preserve chromaticity;
- no sharpening or denoising.

### 8.2 `B2_REFERENCE_STAT_TONE`

1. compute target and reference subject/background luminance statistics;
2. estimate desired semantic-region relation from reference;
3. move target 50% toward that relation;
4. clamp local change to ±0.75 EV;
5. protect highlights;
6. preserve target color and texture;
7. do not use pixel registration.

## 9. Phase 6 — Alignment and Tone-Only Projection

### 9.1 Alignment

Use conservative global alignment only:

- resize;
- optional translation/scale/rotation or homography;
- reject severe mismatch;
- report alignment confidence.

Do not use optical flow to force generated content onto the target.

### 9.2 Projection

Implement the frozen projection contract exactly.

Required outputs:

- `tone_only.png`;
- `delta_low_ev.npy` or equivalent;
- `alignment.json`;
- `projection_metrics.json`.

Required tests:

- exact alpha and clamp consumption;
- chromaticity preservation;
- target-sized output;
- clipping reporting;
- identity projection when Raw equals target;
- deterministic repeatability.

## 10. Phase 7 — Metrics

Compute:

- fixed global tone feature vector;
- target subject/background statistics;
- independent reference-region statistics;
- machine-assisted content-change warnings;
- Raw versus Tone-only differences;
- baseline comparisons.

Create:

- `metrics/image_metrics.json`
- `metrics/summary.csv`

## 11. Phase 8 — Contact Sheets and Blind Review

Generate anonymous IDs and store the mapping separately.

Required contact sheets:

- Raw no-reference;
- Tone-only no-reference;
- Raw reference;
- Tone-only reference;
- baseline comparison.

The review CSV must include candidate ID, mode, all frozen scores, reviewer ID, and comments. Aggregate by median and preserve individual reviews.

## 12. Phase 9 — Hard-Gate Evaluation

Implement the gates in `CONTRACT.md` exactly.

Do not use a weighted total as the decision mechanism.

Create `metrics/gate_results.json` with every gate input, pass/fail result, and rejection reason.

## 13. Phase 10 — Report Generation

Generate `REPORT.md` with:

1. experiment and frozen configuration;
2. input scene description;
3. capability probe;
4. generation successes/failures;
5. Raw comparison;
6. Tone-only comparison;
7. baseline comparison;
8. reference versus no-reference analysis;
9. Boogu Edit versus Turbo;
10. hard-gate results;
11. core decision;
12. Boogu decision;
13. unresolved limitations;
14. explicit non-claims.

The report must attribute apparent benefit separately to Tone, color, texture/restoration, or regeneration.

## 14. CLI

Recommended command:

```bash
python -m cli.run_single_scene_probe \
  --config experiments/single_scene_probe_v1_1/experiment.yaml \
  --output experiments/single_scene_probe_v1_1
```

Optional resumable phases:

```bash
python -m cli.run_single_scene_probe --config ... --phase capabilities
python -m cli.run_single_scene_probe --config ... --phase generate
python -m cli.run_single_scene_probe --config ... --phase project
python -m cli.run_single_scene_probe --config ... --phase metrics
python -m cli.run_single_scene_probe --config ... --phase report
```

Every phase must be resumable without overwriting evidence.

## 15. Test Plan

### Unit Tests

- configuration validation;
- adapter capability handling;
- unsupported reference mode;
- output immutability;
- exact projection parameters;
- clipping computation;
- percentile feature order;
- mask handling;
- baseline behavior;
- gate logic.

### Integration Tests

- synthetic target/reference pair;
- deterministic fake model adapter;
- complete pipeline to report;
- Boogu single-image-only behavior;
- missing API key handling;
- one model failure does not block the others.

### Acceptance Smoke Test

A synthetic fixture must produce:

- successful fake no-reference result;
- successful fake reference result;
- structured unsupported-reference result;
- both baselines;
- Tone-only outputs;
- metrics;
- gate file;
- report.

## 16. Directory Structure

```text
experiments/
└── single_scene_probe_v1_1/
    ├── CONTRACT.md
    ├── PLAN.md
    ├── IMPLEMENTATION_CHECKLIST.md
    ├── experiment.yaml
    ├── inputs/
    ├── prompts/
    ├── outputs/
    │   ├── gpt_image_2/
    │   ├── gemini_image/
    │   ├── flux_2_pro/
    │   ├── qwen_image_edit/
    │   ├── boogu_image_edit/
    │   └── boogu_image_edit_turbo/
    ├── baselines/
    ├── metrics/
    ├── visualizations/
    ├── logs/
    └── REPORT.md
```

## 17. Final Engineering Return

Engineering must return:

- repository branch and commit;
- exact environment instructions;
- model capability matrix;
- artifact paths;
- complete test commands and results;
- unsupported modes;
- final report;
- no claim beyond this single-scene probe.
