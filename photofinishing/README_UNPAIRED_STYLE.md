# Same-Scene Non-Pixel-Aligned Style Adaptation

This experiment adapts an existing Photofinishing checkpoint using same-scene A-camera references that are not pixel aligned with the B-camera input/output.

## Scope

- **Stage 1 (`luminance`)** trains only `GainNet` and `GlobalToneMappingNet`.
- **Stage 2 (`chroma`)** compares two capacity-controlled color adaptation modes.
- `LocalToneMappingNet`, `GammaNet`, and the optional 3D LUT stay frozen.
- No pixelwise loss is calculated between output and reference.

## Manifest

```csv
sample_id,input_path,reference_path,metadata_path,split
scene_0001,data/b/scene_0001.png,data/a/scene_0001.png,,train
scene_0002,data/b/scene_0002.png,data/a/scene_0002.png,,val
scene_0003,data/b/scene_0003.png,data/a/scene_0003.png,,test
```

`input_path` and `reference_path` may have different resolution, crop, or viewpoint. Paths are resolved relative to the manifest. `metadata_path` is required only for `--input-mode raw_metadata` and must contain `cam_illum` and `ccm`.

## Stage 1: Gain + Global Tone Mapping

```bash
cd photofinishing
python train_unpaired_style.py \
  --stage luminance \
  --manifest /path/to/manifest.csv \
  --load /path/to/source_photofinishing.pth \
  --output-dir runs/luminance \
  --train-split train \
  --validation-split val
```

The loss uses per-sample log exposure, luminance percentiles, and a quantile approximation of one-dimensional Wasserstein distance. A small parameter anchor prevents unrestricted drift from the source checkpoint.

## Stage 2: Chroma Mapping Capacity Comparison

Both Stage-2 runs must use the same Stage-1 checkpoint, manifest, splits, seed, image size, optimizer settings, epoch count, and loss weights. The only experimental variable is `--chroma-head`.

### Control: full adaptive LuTNet

```bash
python train_unpaired_style.py \
  --stage chroma \
  --chroma-head full_lut \
  --manifest /path/to/manifest.csv \
  --load runs/luminance/best.pth \
  --output-dir runs/chroma_full_lut \
  --train-split train \
  --validation-split val
```

This mode fine-tunes the complete original image-adaptive LuTNet, including its base LUT, CbCr-histogram encoder, luminance guidance, and nonlinear 24x24 LUT predictor.

### Low capacity: frozen LuTNet plus affine residual

```bash
python train_unpaired_style.py \
  --stage chroma \
  --chroma-head affine_residual \
  --manifest /path/to/manifest.csv \
  --load runs/luminance/best.pth \
  --output-dir runs/chroma_affine \
  --train-split train \
  --validation-split val
```

This mode freezes the complete Stage-1 LuTNet and preserves its scene-adaptive nonlinear color mapping. It trains only:

- a bounded 2x2 residual matrix on CbCr;
- a bounded two-dimensional CbCr bias;
- six scalar parameters in total.

The affine head is initialized as an exact identity residual, so iteration-0 output and LUT are identical to the Stage-1 checkpoint. The base LuTNet output is not additionally clamped. Matrix residual entries are bounded to +/-0.15 and bias entries to +/-0.05.

Both modes use the same differentiable CbCr histogram, chroma moments, saturation distribution, final-luminance preservation, and LUT-delta anchor/smoothness losses relative to a frozen copy of the Stage-1 checkpoint.

Each Stage-2 `run_config.json` records `chroma_head`, trainable parameter names, trainable parameter count, checkpoint hashes, manifest hash, seed, and loss weights. Affine runs also record the matrix and bias limits.

## Evaluation

The evaluator reads the adapted `run_config.json` beside `best.pth` and reconstructs the correct chroma head before strict checkpoint loading.

```bash
python eval_unpaired_style.py \
  --manifest /path/to/manifest.csv \
  --split test \
  --baseline-load /path/to/source_photofinishing.pth \
  --adapted-load runs/chroma_full_lut/best.pth \
  --output runs/chroma_full_lut/test_metrics.json

python eval_unpaired_style.py \
  --manifest /path/to/manifest.csv \
  --split test \
  --baseline-load /path/to/source_photofinishing.pth \
  --adapted-load runs/chroma_affine/best.pth \
  --output runs/chroma_affine/test_metrics.json
```

Use `--adapted-run-config` when the run config is not beside the adapted checkpoint. Legacy full-LUT checkpoints without a run config remain loadable as `full_lut`; affine checkpoints require their run config.

The report contains per-scene and mean baseline/adapted distances plus the reconstructed `adapted_chroma_head`. Positive `mean_improvement` means the adapted model is closer to the same-scene reference for that non-pixel metric.

## Comparison Criteria

Do not select a head from mean Chroma loss alone. Compare:

- mean and worst-decile Chroma improvement;
- fraction of test scenes with negative improvement;
- luminance drift from the shared Stage-1 checkpoint;
- clipping and visible green/magenta failure cases;
- train/validation gap and cross-seed variance.

The affine branch is preferred when it reaches similar mean improvement with fewer regressions or lower variance. The full LUT is justified only when its additional nonlinear capacity produces stable holdout gains rather than fitting noisy non-aligned statistics.

## Claim boundary

A lower non-pixel style distance does not prove pixel-level correctness or final image preference. The experiment must also inspect clipping, color collapse, scene-to-scene variance, and source/good-case replay before integration into the cross-camera training pipeline.
