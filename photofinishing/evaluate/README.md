# Non-Aligned Photofinishing Evaluation

This tool evaluates multiple Photofinishing experiment chains:

```text
Pretrained reference-camera model
        -> Stage 1 luminance/tone adaptation
        -> Stage 2 chroma adaptation variants
```

against same-scene reference-camera product images that are **not assumed to be pixel aligned** with the target-camera input.

The evaluator is designed to answer four project questions:

1. Does Stage 1 make target-camera brightness and global tone closer to the reference?
2. After Stage 1, is a separate color adaptation stage still necessary?
3. Does Stage 2 add color improvement without damaging Stage-1 luminance?
4. Are current results conclusive, or should evaluation/training data be expanded?

## 1. Inputs

### Experiment JSON

Use `example_experiments.json` as the template. Each group contains one matched chain:

```json
{
  "name": "seed42",
  "pretrained": {"checkpoint": ".../pretrained.pth"},
  "stage1": {"checkpoint": ".../stage1/best.pth"},
  "stage2": [
    {
      "name": "affine_residual",
      "checkpoint": ".../affine/best.pth",
      "run_config": ".../affine/run_config.json"
    },
    {
      "name": "full_lut",
      "checkpoint": ".../full_lut/best.pth",
      "run_config": ".../full_lut/run_config.json"
    }
  ]
}
```

Multiple groups may represent different seeds, data subsets, pretrained models, or repeated experiments. Cross-group results are averaged per scene before the final confidence interval, so repeated seeds are not incorrectly treated as independent scenes.

For Stage-2 checkpoints, the evaluator reuses the existing strict loader:

- affine checkpoints require a valid `run_config.json`;
- the affine wrapper is reconstructed before loading the state dict;
- best/last checkpoint SHA-256 must match the run config;
- incompatible checkpoint/head combinations fail closed.

### Extended manifest

Required columns remain compatible with the training manifest:

```text
sample_id,input_path,reference_path,split
```

Optional columns:

| Column | Purpose |
|---|---|
| `metadata_path` | Required for `--input-mode raw_metadata` |
| `reference_repeat_path` | Second reference-camera capture for the measurable noise floor |
| `input_valid_mask_path` / `reference_valid_mask_path` | Common or trustworthy FoV on each image |
| `input_ignore_mask_path` / `reference_ignore_mask_path` | Dynamic or unreliable regions to exclude |
| `input_skin_mask_path` / `reference_skin_mask_path` | Non-aligned skin-region color comparison |
| `input_sky_mask_path` / `reference_sky_mask_path` | Non-aligned sky color comparison |
| `input_vegetation_mask_path` / `reference_vegetation_mask_path` | Non-aligned vegetation comparison |
| `scene_tags` | Semicolon-separated slices such as `portrait;night;hdr` |

Semantic masks must be supplied on both target-input and reference sides. They are compared as separate color distributions; no pixel correspondence is created.

## 2. Run

From the repository root:

```bash
python -m photofinishing.evaluate \
  --config photofinishing/evaluate/example_experiments.json \
  --manifest photofinishing/evaluate/example_manifest.csv \
  --split test \
  --output-dir runs/non_aligned_evaluation \
  --device cuda \
  --image-size 512 \
  --bootstrap-samples 5000 \
  --min-count 20 \
  --min-win-rate 0.60
```

Useful options:

```text
--panel-limit 50       Save at most 50 panels per experiment group
--panel-limit 0        Save panels for every scene
--no-panels            Disable comparison panels
--save-outputs         Save each model output as PNG
--output-bit-depth 16  Save 16-bit PNG outputs
```

## 3. Stage-1 metrics

Stage 1 is evaluated from the paired increment:

```text
distance(Pretrained, Reference) - distance(Stage1, Reference)
```

Positive values mean improvement.

| Metric | Meaning |
|---|---|
| `signed_ev_error` | Directional brightness bias; positive is brighter, negative is darker. Diagnostic only. |
| `absolute_ev_error` | Absolute median brightness mismatch in EV |
| `log_luma_quantile_mae` | Shadow/midtone/highlight quantile-curve error |
| `tone_shape_mae` | Quantile error after removing median exposure; rejects simple global brightening as a complete solution |
| `log_luma_w1` | Wasserstein distance between log-luminance distributions |
| `shadow_ratio_error` | Difference in deep-shadow proportion |
| `highlight_ratio_error` | Difference in bright-highlight proportion |
| `clipping_ratio_error` | Difference in near-black/near-white channel clipping |

Stage 1 is accepted only when brightness error and at least one additional tone-distribution measure improve without clipping regression.

## 4. Stage-2 metrics

Stage 2 is evaluated only from the additional increment:

```text
distance(Stage1, Reference) - distance(Stage2, Reference)
```

Inherited Stage-1 improvement does not count as Stage-2 evidence.

| Metric | Meaning |
|---|---|
| `luminance_conditioned_cbcr_swd` | Primary metric: CbCr sliced Wasserstein distance separately in shadow, midtone, and highlight bands |
| `cbcr_swd` | Complete-image CbCr distribution distance |
| `chroma_mean_error` | Global blue-yellow/red-green center mismatch |
| `chroma_covariance_error` | Color spread and coupling mismatch |
| `saturation_w1` | CIELAB chroma-magnitude distribution distance |
| `neutral_axis_error` | Tint mismatch in low-chroma regions |
| `semantic_*_lab_swd` | Optional skin/sky/vegetation Lab color-distribution distance |
| `semantic_*_area_gap` | Difference in semantic area ratio; diagnostic for composition mismatch |
| `semantic_composition_max_gap` | Largest available semantic-composition gap |

A Stage-2 variant is accepted only when:

- the primary color metric improves consistently;
- at least two auxiliary color metrics agree;
- Stage-1 EV, tone shape, and clipping do not regress;
- the sample count and bootstrap confidence interval meet the configured evidence gates.

## 5. Reference noise floor

When `reference_repeat_path` is available, the evaluator measures:

```text
distance(ReferenceRepeat, Reference)
```

Stage 2 is considered necessary only when the Stage-1 color residual is consistently above this reference-camera capture/non-alignment variability.

Without repeated references, the tool reports Stage-2 necessity as `undetermined`; it does not claim that a small Stage-2 loss proves a real color residual.

## 6. Statistical outputs

Every transition reports:

- count;
- mean improvement;
- median improvement;
- standard deviation;
- win rate;
- negative-improvement rate;
- p10 worst-tail improvement;
- paired bootstrap 95% confidence interval.

Cross-group/seed evidence is first averaged per scene and only then bootstrapped.

## 7. Generated files

```text
output_dir/
  report.md
  decisions.json
  summary.json
  summary.csv
  per_sample_metrics.csv
  incremental_improvements.csv
  panels/<group>/<sample>.jpg
  outputs/<group>/<model>/<sample>.png   # when --save-outputs
```

### Recommended reading order

1. `report.md`: direct answers for Stage 1, Stage 2, model choice, and data expansion.
2. `decisions.json`: exact machine-readable statuses and reasons.
3. `incremental_improvements.csv`: identify which scenes improve or regress.
4. `panels/`: verify that metric gains correspond to real brightness/color improvement.
5. `summary.json`: full group, cross-group, slice, tail, and confidence-interval evidence.

## 8. Data expansion interpretation

The tool separates two decisions:

### Expand evaluation data

Recommended when:

- unique evaluation scenes are below `--min-count`;
- a Stage-2 confidence interval remains inconclusive.

This should normally be addressed before expanding the training set.

### Expand training data

Recommended only when Stage 2 is shown to be necessary and either:

- the Stage-2 gain changes sign across groups/seeds;
- failures concentrate in specific scene tags such as night, portrait, or HDR.

If the Stage-1 residual is already close to the repeated-reference noise floor, adding more Stage-2 training data is not justified by the current evidence.

## 9. Limitations

- Distribution agreement does not prove local pixel correctness.
- Large viewpoint or semantic-composition differences can still bias global metrics; use valid/ignore and semantic masks.
- Local face, sky, and highlight tone quality requires visual review and may require the later pixel-aligned pseudo-GT stage.
- The evaluator determines whether the current non-aligned supervision is effective; it does not replace blind human preference testing.
