# adjustTM Benchmark

This package implements the fixed evaluation protocol for controllable brightness Tone Mapping.

## Evidence boundaries

The nine targets do not have the same provenance:

- `a_000`: real camera RGB paired with the WB+CCM linear input. This is the only level used for claims about real camera reconstruction quality.
- `a_m100`, `a_p100`: fixed, pre-generated Flux endpoints. Metrics measure Flux endpoint fidelity.
- Six nonzero intermediate levels: Retinex interpolation. Metrics measure teacher-trajectory fidelity.

The benchmark never labels the mean score over all nine targets as absolute image quality.

## Compared methods

Main ranking:

- `frozen_baseline`
- `exposure_global`
- `gamma_global`
- `param_residual`
- `parallel_adapter`
- `film`
- `dual_lora`

Diagnostic upper bounds, excluded from the main ranking:

- `exposure_oracle`
- `gamma_oracle`

Global exposure/gamma parameters are calibrated only on the training/calibration scenes. Oracle parameters are optimized separately for each test scene and level using test GT, so they are diagnostic only.

## Output layout

```text
benchmark_runs/<run_id>/
├── protocol.json
├── manifest.json
├── methods.json
├── calibration/simple_baselines.json
├── outputs/
│   ├── cache_identity.json
│   ├── inference_records.jsonl
│   ├── dense_control_records.jsonl
│   └── <method>/<level>/<scene>.png
├── metrics/
│   ├── reference_records.jsonl
│   ├── trajectory_records.jsonl
│   ├── control_records.jsonl
│   └── vlm_*.jsonl
├── human_study/
├── statistics/summary.json
└── report/
```

All cached method outputs are lossless 16-bit sRGB PNGs. Cache reuse is rejected when the manifest, protocol, checkpoint hash, method configuration, level set, dense-alpha set, or image geometry changes.

## 1. Build the independent test manifest

```bash
python -m adjustTM.benchmark.build_manifest \
  --input-dir /data/test/input_linear \
  --gt-root /data/test/gt_levels \
  --scene-list /data/splits/test_scenes.json \
  --scene-tags /data/test/scene_tags.csv \
  --output benchmark_runs/run_001/manifest.json
```

The command fails on missing levels, invalid image types, dimension mismatches, duplicate scene names, and unreadable images.

## 2. Generate frozen outputs for calibration scenes

Create a calibration manifest from training scenes, then declare only `frozen_baseline` in a temporary methods file and run:

```bash
python -m adjustTM.benchmark.generate_outputs \
  --manifest benchmark_runs/calibration/manifest.json \
  --protocol adjustTM/benchmark/configs/protocol.example.json \
  --methods benchmark_runs/calibration/frozen_method.json \
  --output-root benchmark_runs/calibration/outputs \
  --dense-steps 0 \
  --device cuda
```

The frozen outputs used by calibration are under:

```text
benchmark_runs/calibration/outputs/frozen_baseline/a_000/
```

## 3. Calibrate deployable simple baselines

Use the calibration/train manifest, never the test manifest:

```bash
python -m adjustTM.benchmark.calibrate_baselines \
  --manifest benchmark_runs/calibration/manifest.json \
  --baseline-output-root benchmark_runs/calibration/outputs/frozen_baseline \
  --exposure-min -4 \
  --exposure-max 4 \
  --exposure-steps 161 \
  --gamma-min 0.25 \
  --gamma-max 4.0 \
  --gamma-steps 161 \
  --device cuda \
  --output benchmark_runs/run_001/calibration/simple_baselines.json
```

The implementation streams one scene at a time. It evaluates the complete parameter grid at full floating-point precision without loading the complete dataset into memory. Parameter sequences are projected to satisfy:

```text
exposure: nondecreasing, alpha=0 -> 0 EV
gamma:    nonincreasing, alpha=0 -> gamma 1
```

## 4. Configure methods

Copy `configs/methods.example.json`, replace all placeholder paths, and place `frozen_baseline` before global exposure/gamma because those runners reuse it.

## 5. Generate fixed method outputs and dense control records

```bash
python -m adjustTM.benchmark.generate_outputs \
  --manifest benchmark_runs/run_001/manifest.json \
  --protocol benchmark_runs/run_001/protocol.json \
  --methods benchmark_runs/run_001/methods.json \
  --output-root benchmark_runs/run_001/outputs \
  --dense-steps 41 \
  --max-side 512 \
  --multiple 16 \
  --device cuda
```

The default geometry preserves aspect ratio, resizes the longest side to at most 512, pads to a multiple of 16, then removes padding before saving. GT is resized to the exact cached output size during evaluation.

For every method and scene, the generator records:

- nine fixed-level outputs;
- 41-point dense mean log luminance;
- clipping and deep-shadow ratios;
- chromaticity drift from alpha zero;
- Gain/GTM parameters when available;
- runtime;
- tensor-level alpha-zero drift.

A method exposing a baseline reference fails immediately when alpha-zero drift exceeds `1e-7`.

## 6. Generate diagnostic oracles

```bash
python -m adjustTM.benchmark.generate_oracles \
  --manifest benchmark_runs/run_001/manifest.json \
  --baseline-output-root benchmark_runs/run_001/outputs/frozen_baseline \
  --methods exposure gamma \
  --steps 161 \
  --output-root benchmark_runs/run_001/outputs \
  --parameter-output benchmark_runs/run_001/calibration/oracle_parameters.jsonl \
  --device cuda
```

Oracle parameters minimize pixelwise log-luminance MAE. They must not be placed in the main-method list.

## 7. Reference and trajectory metrics

```bash
python -m adjustTM.benchmark.evaluate_reference \
  --manifest benchmark_runs/run_001/manifest.json \
  --output-root benchmark_runs/run_001/outputs \
  --methods frozen_baseline exposure_global gamma_global \
            param_residual parallel_adapter film dual_lora \
            exposure_oracle gamma_oracle \
  --output benchmark_runs/run_001/metrics/reference_records.jsonl \
  --trajectory-output benchmark_runs/run_001/metrics/trajectory_records.jsonl \
  --device cuda \
  --lpips
```

Reference records contain RGB/luminance PSNR and SSIM, optional LPIPS, log-luminance MAE, gradient MAE, chromaticity error, clipping, deep-shadow ratio, and semantic GT group.

Trajectory records are generated only when all nine levels are present. They contain curve MAE, adjacent-step MAE, endpoint-range error, Spearman correlation, and nine-level monotonic violations.

## 8. Dense, GT-independent control metrics

```bash
python -m adjustTM.benchmark.evaluate_control \
  --dense-records benchmark_runs/run_001/outputs/dense_control_records.jsonl \
  --dead-zone-threshold 0.002 \
  --jump-ratio-threshold 3.0 \
  --output benchmark_runs/run_001/metrics/control_records.jsonl
```

This stage reports scene-level violation rate and magnitude, strict monotonic pass, dead zones, jumps, normalized second-difference smoothness, positive/negative control ranges, total range, and range balance.

## 9. VLM evaluation

Export auditable VLM tasks without binding the benchmark to one model vendor:

```bash
python -m adjustTM.benchmark.evaluate_vlm \
  --manifest benchmark_runs/run_001/manifest.json \
  --output-root benchmark_runs/run_001/outputs \
  --methods frozen_baseline exposure_global gamma_global \
            param_residual parallel_adapter film dual_lora \
  --levels a_m100 a_m050 a_p050 a_p100 \
  --kinds intent naturalness \
  --tasks-output benchmark_runs/run_001/metrics/vlm_tasks.jsonl
```

Each task includes image paths, the complete Chinese prompt, required score keys, and score ranges. `intent` receives center, target, and candidate images. `naturalness` receives only center and candidate images.

To execute a local Qwen-IQA/VLM wrapper, provide a command that reads one task JSON from stdin and emits one response JSON to stdout:

```bash
python -m adjustTM.benchmark.evaluate_vlm ... \
  --backend-command "python /path/to/qwen_iqa_backend.py" \
  --repeats 3 \
  --max-retries 2 \
  --responses-output benchmark_runs/run_001/metrics/vlm_responses.jsonl \
  --aggregate-output benchmark_runs/run_001/metrics/vlm_aggregate.jsonl
```

Responses are schema-validated. Repeats are aggregated by median and preserve standard deviation, range, confidence, raw output, retries, and unstable-judgment flags. Intent and naturalness are never merged into one score.

## 10. Blinded human study

```bash
python -m adjustTM.benchmark.build_human_study \
  --manifest benchmark_runs/run_001/manifest.json \
  --output-root benchmark_runs/run_001/outputs \
  --methods frozen_baseline exposure_global gamma_global \
            param_residual parallel_adapter film dual_lora \
  --levels a_m100 a_m050 a_p050 a_p100 \
  --scene-count 32 \
  --candidates-per-trial 4 \
  --blocks-per-scene-level 2 \
  --asset-mode copy \
  --seed 42 \
  --output-dir benchmark_runs/run_001/human_study
```

Scenes are selected round-robin across available tags. Candidate exposure and pair co-occurrence are balanced. All candidate images are copied/hard-linked/symlinked to randomized asset names, so method names do not appear in the HTML or image URLs. The unblinding map is stored separately.

Open these local files:

```text
human_study/intent_match.html
human_study/naturalness.html
```

The page records best/worst choices and duration and downloads a response CSV.

Analyze responses:

```bash
python -m adjustTM.benchmark.analyze_human_study \
  --responses /data/human_responses.csv \
  --method-map benchmark_runs/run_001/human_study/blinded_method_map.json \
  --output-dir benchmark_runs/run_001/human_study/results
```

Raters failing attention, repeat-consistency, or response-time gates are excluded before Bradley-Terry ranking.

## 11. Scene-level statistics

```bash
python -m adjustTM.benchmark.compare_methods \
  --reference-records benchmark_runs/run_001/metrics/reference_records.jsonl \
  --control-records benchmark_runs/run_001/metrics/control_records.jsonl \
  --main-methods frozen_baseline exposure_global gamma_global \
                 param_residual parallel_adapter film dual_lora \
  --diagnostic-methods exposure_oracle gamma_oracle \
  --comparison-baselines frozen_baseline exposure_global gamma_global \
  --protocol benchmark_runs/run_001/protocol.json \
  --output benchmark_runs/run_001/statistics/summary.json
```

Level values are first averaged within a scene. The program then reports scene count, mean, median, scene-bootstrap 95% CI, worst-5% CVaR, paired delta, win/tie/loss, paired permutation p-value, and Holm-adjusted p-value.

## 12. Report

```bash
python -m adjustTM.benchmark.report \
  --summary benchmark_runs/run_001/statistics/summary.json \
  --output-dir benchmark_runs/run_001/report
```

The report preserves separate main-method and oracle sections. No opaque weighted overall score is produced.

## 13. Resumable orchestration

`run.py` executes exact commands from a JSON plan and records completed stages. A changed config hash invalidates the state file.

```bash
python -m adjustTM.benchmark.run \
  --config adjustTM/benchmark/configs/run.example.json \
  --stages manifest calibrate generate oracle reference control compare report \
  --state benchmark_runs/run_001/benchmark_state.json
```

## Tests

```bash
PYTHONPATH=. python -m compileall -q adjustTM/benchmark
PYTHONPATH=. python -m pytest adjustTM/tests/test_benchmark.py -q
PYTHONPATH=. python -m pytest adjustTM/tests -q
```
