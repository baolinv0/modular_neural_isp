# adjustTM Visual Evidence Gallery

The visual gallery converts the cached benchmark images and scene-level records into a portable HTML evidence package. It is designed to answer both questions:

1. Which method wins quantitatively?
2. What does that difference look like on real images?

## Command

```bash
python -m adjustTM.benchmark.build_case_gallery \
  --manifest benchmark_runs/run_001/manifest.json \
  --output-root benchmark_runs/run_001/outputs \
  --reference-records benchmark_runs/run_001/metrics/reference_records.jsonl \
  --dense-records benchmark_runs/run_001/outputs/dense_control_records.jsonl \
  --control-records benchmark_runs/run_001/metrics/control_records.jsonl \
  --vlm-records benchmark_runs/run_001/metrics/vlm_aggregate.jsonl \
  --methods frozen_baseline exposure_global gamma_global \
            param_residual parallel_adapter film dual_lora \
  --focus-method film \
  --comparison-baseline gamma_global \
  --representative-count 6 \
  --best-count 6 \
  --failure-count 6 \
  --disagreement-count 6 \
  --asset-mode copy \
  --crop-fraction 0.25 \
  --output-dir benchmark_runs/run_001/qualitative
```

`--vlm-records` is optional. When it is absent, the metric–perception disagreement page is generated but contains no selected cases.

## Case selection

The gallery is deterministic for fixed records and method names.

- **Representative cases**: scenes whose nonzero-level log-luminance error is near the focus method's median and whose improvement over the selected baseline is not extreme.
- **Where learning helps**: scenes with the largest positive reduction in log-luminance error relative to the selected baseline.
- **Failure cases**: scenes ranked by a composite of GT-fidelity risk, dense-control violations, dead zones, jumps and optional VLM naturalness. The score uses within-dataset ranks so metrics with different units cannot dominate solely by scale.
- **Metric–perception disagreement**: scenes whose GT-fidelity rank and VLM naturalness rank disagree most strongly.

The categories avoid duplicate scenes when enough test scenes are available. Small synthetic or debug datasets may reuse a scene because there are not enough distinct candidates.

## Pages

```text
qualitative/
├── index.html
├── representative_cases.html
├── best_improvements.html
├── failure_cases.html
├── metric_vlm_disagreement.html
├── scene_browser.html
├── case_index.json
└── assets/
```

Each selected case contains:

- target plus all methods at the same brightness level;
- target and per-method trajectories over every requested level;
- log-luminance, SSIM, LPIPS, clipping and shadow badges when available;
- automatically selected highlight, shadow and texture crops;
- dense mean-log-luminance, clipping, deep-shadow and chroma-drift curves;
- the nine-level target mean-log-luminance curve overlaid on model curves.

`scene_browser.html` exposes every materialized scene, brightness level and method through local selectors.

## Crop alignment

GT files can retain their original dimensions while cached model outputs are resized by the benchmark. Crop boxes are detected once on the real camera `a_000` image, represented relative to its width and height, and remapped independently to every target and model output. This prevents a visually convincing but spatially incorrect crop comparison.

Automatic crops are:

- **highlight**: window with the highest average linearized display luminance proxy;
- **shadow**: window with the lowest average luminance;
- **texture**: window with the highest Sobel gradient energy.

## Asset modes

- `copy`: portable package; safest default and used by the example run plan.
- `hardlink`: avoids duplicate storage when source and destination share a filesystem; falls back to copy.
- `symlink`: smallest package but less portable; falls back to copy when unsupported.

The source benchmark outputs remain lossless 16-bit sRGB PNGs. Gallery crops preserve the source PNG bit depth.

## Link from the quantitative report

```bash
python -m adjustTM.benchmark.report \
  --summary benchmark_runs/run_001/statistics/summary.json \
  --visual-gallery benchmark_runs/run_001/qualitative/index.html \
  --output-dir benchmark_runs/run_001/report
```

The generated `report.html` then includes a `Visual evidence` entry linking the quantitative tables to the qualitative case package.
