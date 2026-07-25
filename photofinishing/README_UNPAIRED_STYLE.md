# Same-Scene Non-Pixel-Aligned Style Adaptation

This experiment adapts an existing Photofinishing checkpoint using same-scene A-camera references that are not pixel aligned with the B-camera input/output.

## Scope

- **Stage 1 (`luminance`)** trains only `GainNet` and `GlobalToneMappingNet`.
- **Stage 2 (`chroma`)** loads the Stage-1 checkpoint and trains only `LuTNet`.
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

## Stage 2: Chroma Mapping

```bash
python train_unpaired_style.py \
  --stage chroma \
  --manifest /path/to/manifest.csv \
  --load runs/luminance/best.pth \
  --output-dir runs/chroma \
  --train-split train \
  --validation-split val
```

The chroma stage requires the Stage-1 `run_config.json`, normally found beside `best.pth`. It uses a differentiable two-dimensional CbCr histogram, chroma moments, saturation distribution, final-luminance preservation, and LUT-delta anchor/smoothness relative to a frozen copy of the Stage-1 checkpoint.

## Evaluation

```bash
python eval_unpaired_style.py \
  --manifest /path/to/manifest.csv \
  --split test \
  --baseline-load /path/to/source_photofinishing.pth \
  --adapted-load runs/chroma/best.pth \
  --output runs/chroma/test_metrics.json
```

The report contains per-scene and mean baseline/adapted distances. Positive `mean_improvement` means the adapted model is closer to the same-scene reference for that non-pixel metric.

## Claim boundary

A lower non-pixel style distance does not prove pixel-level correctness or final image preference. The experiment must also inspect clipping, color collapse, scene-to-scene variance, and source/good-case replay before integration into the cross-camera training pipeline.
