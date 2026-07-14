# adjustTM: brightness-only controllable tone mapping

`adjustTM` compares four ways of adding continuous brightness control to one shared, pretrained luminance-only Modular Neural ISP baseline:

- `param_residual`
- `parallel_adapter`
- `film`
- `dual_lora`

Only GainNet and GlobalToneMappingNet receive trainable control parameters. Original Gain/GTM weights and the entire LTM remain frozen. The LTM is still active and can respond to changed upstream images, but it receives no trainable control parameters. Color LUTs and learned gamma are not used. Output encoding uses the fixed standard sRGB OETF.

## Training handoff

For the complete operator-facing training procedure, fairness constraints, four-GPU commands, resume workflow, evaluation protocol and delivery checklist, read:

- [TRAINING_HANDOFF_CN.md](TRAINING_HANDOFF_CN.md)

## Precision benchmark

For the fixed test manifest, deployable Exposure/Gamma baselines, diagnostic oracles, four-model comparison, semantic GT grouping, dense control analysis, VLM tasks, blinded human study, scene-level statistics and HTML/CSV/XLSX reports, read:

- [benchmark/README.md](benchmark/README.md)
- [benchmark/VISUAL_GALLERY.md](benchmark/VISUAL_GALLERY.md) for representative, improvement, failure, local-crop, control-curve and full-scene visual evidence.

## Data layout

```text
input_linear/
  scene_0001.png       # uint16 linear RGB; WB + CCM already applied

gt_levels/
  a_m100/scene_0001.png
  a_m075/scene_0001.png
  a_m050/scene_0001.png
  a_m025/scene_0001.png
  a_000/scene_0001.png
  a_p025/scene_0001.png
  a_p050/scene_0001.png
  a_p075/scene_0001.png
  a_p100/scene_0001.png
```

All ten files for a scene must have identical filenames and dimensions. Input images must be 16-bit PNG. GT images may be 8-bit or 16-bit sRGB PNG. Missing files are fatal.

## Fair sampling

Each scene is expanded into all `C(9,2)=36` low/high level pairs. The 36 pairs are partitioned into two 18-pair balanced blocks. Every 18-sample block contains every level exactly four times. Scene assignment rotates deterministically, so every scene-pair combination appears exactly once per epoch.

The batch size must be a positive multiple of 18. Four experiments using the same scene split, seed, batch size and epoch number receive the same sample order.

## Baseline checkpoint smoke gate

Run this before full training. It loads the real baseline checkpoint into every method and verifies:

- checkpoint key and tensor-shape compatibility;
- exact `alpha=0` equivalence;
- no baseline gradients or parameter updates;
- a successful control-parameter optimizer step;
- control checkpoint save/load round trip.

```bash
python -m adjustTM.smoke \
  --baseline-checkpoint /models/luminance_only_baseline.pth \
  --input /data/input_linear/scene_0001.png \
  --device cuda \
  --output adjustTM/smoke_results.json
```

## Train

```bash
python -m adjustTM.train \
  --input-dir /data/input_linear \
  --gt-root /data/gt_levels \
  --baseline-checkpoint /models/luminance_only_baseline.pth \
  --control-method param_residual \
  --output-dir adjustTM/checkpoints \
  --batch-size 18 \
  --val-fraction 0.1 \
  --epochs 30 \
  --seed 42 \
  --amp
```

Run the same command with `parallel_adapter`, `film`, and `dual_lora`. Reuse the same `--output-dir` or explicitly pass the same `--split-manifest` and `--manifest-dir`. The code stores:

- deterministic scene split JSON;
- train/validation sample-index JSON with SHA-256;
- method configuration and baseline checkpoint SHA-256;
- `control_last.pth`;
- `control_best.pth` selected by validation log-luminance MAE;
- optional periodic epoch checkpoints;
- JSONL training and validation logs.

Resume an interrupted run with:

```bash
python -m adjustTM.train ... \
  --resume adjustTM/checkpoints/film/control_last.pth
```

The resume checkpoint restores control parameters, optimizer, scheduler, AMP scaler and random-number-generator states.

### Default parameter budget

| Method | Trainable control parameters |
|---|---:|
| param_residual | 1064 |
| parallel_adapter | 1096 |
| film | 976 |
| dual_lora | 1024 |

Training checks each method against a target of 1040 parameters with ±10% tolerance. `--allow-parameter-mismatch` is reserved for intentional capacity ablations.

## Objective

```text
L = L_logY + lambda_grad * L_gradient
             + lambda_mono * L_monotonic
             + lambda_zero * L_alpha0
```

The objective does not use VGG, DeltaE, CbCr, LUT, saturation or learned-gamma losses. Color is not structurally guaranteed to remain unchanged because nonlinear RGB tone curves can alter chromaticity; evaluation therefore reports chroma drift even though it is not optimized.

## Evaluate

```bash
python -m adjustTM.evaluate \
  --input-dir /data/test/input_linear \
  --gt-root /data/test/gt_levels \
  --baseline-checkpoint /models/luminance_only_baseline.pth \
  --control-checkpoint adjustTM/checkpoints/film/control_best.pth \
  --control-method film \
  --output adjustTM/results/film.json
```

Reported metrics include:

- per-level log-luminance MAE;
- luminance PSNR and SSIM;
- chromaticity `rg` error to GT and drift from the alpha-zero output;
- clipping and deep-shadow ratios;
- nine-level control-curve MAE;
- adjacent-level step error;
- endpoint range error;
- nine-level Spearman correlation;
- dense alpha-sweep violation rate and scene pass rate;
- alpha-zero maximum baseline drift;
- Gain and three GTM parameter ranges;
- measured inference latency.

## Checkpoint compatibility

The baseline loader accepts:

- a plain baseline `state_dict`;
- a dictionary under `state_dict`, `model_state_dict`, or `model`;
- common `module.`, `model.`, `baseline.` and photofinishing wrapper prefixes;
- full photofinishing checkpoints containing removed `_gamma_net`, `_lut_net`, or `_3d_lut` keys.

Gain, GTM and LTM keys and shapes must match exactly. Control checkpoints must contain the complete expected key set for the selected method; partial or cross-method checkpoints fail fast.

## Tests

```bash
python -m compileall -q adjustTM
pytest adjustTM/tests -q
```

The focused GitHub Actions workflow runs both commands on every `adjustTM` pull request.
