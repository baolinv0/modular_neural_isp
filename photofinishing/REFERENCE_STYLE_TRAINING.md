# Same-Scene Unaligned Reference-Style Fine-Tuning

This experiment uses `Bin` as the existing photofinishing model input and a same-scene but non-pixel-aligned `AGT` image as a style reference. It does **not** use pixel reconstruction against `AGT`.

## Two-stage protocol

### Stage 1: brightness and global tone

Trainable:

- digital Gain (`_gain_net`);
- Global Tone Mapping (`_gtm_net`).

Frozen:

- Local Tone Mapping (`_ltm_net`);
- CbCr LUT (`_lut_net`);
- Gamma (`_gamma_net`);
- optional 3D LUT.

Reference loss:

- sorted log-luminance distribution;
- luminance mean and standard deviation;
- soft shadow occupancy;
- soft highlight occupancy.

### Stage 2: chroma

Start from the best Stage-1 checkpoint.

Trainable:

- CbCr LUT network (`_lut_net`) only.

Frozen:

- Gain;
- GTM;
- LTM;
- Gamma;
- optional 3D LUT.

Reference loss:

- non-spatial Cb/Cr marginal distributions;
- chroma moments and covariance;
- chroma-magnitude distribution.

A frozen copy of the best Stage-1 model supplies a same-input luminance anchor. This anchor is pixel aligned with the current output because both are rendered from the same `Bin`; it is not supervision from `AGT`. It prevents color fine-tuning from changing the Stage-1 tone result.

The reference image can have a different viewpoint and resolution. Both images are independently resized. No aligned crop, optical-flow warp, SSIM, PSNR, or pixel L1 loss is applied between the output and `AGT`.

## Manifest

Use one JSONL manifest with scene-disjoint `train`, `val`, and `test` rows:

```json
{"sample_id":"scene_001","input_path":"data/Bin/001.png","reference_path":"data/AGT/001.png","split":"train","weight":1.0}
{"sample_id":"scene_002","input_path":"data/Bin/002.png","reference_path":"data/AGT/002.png","split":"val","weight":1.0}
{"sample_id":"scene_003","input_path":"data/Bin/003.png","reference_path":"data/AGT/003.png","split":"test","weight":1.0}
```

Paths may be absolute or relative to the manifest.

Supported inputs:

- uint16 linear-RGB PNG;
- uint8 RGB image;
- normalized float32 `.npy` image.

All loaded arrays must be finite and in `[0,1]` after integer normalization.

## Train

```bash
python photofinishing/train_reference_style.py \
  --manifest /path/to/reference_pairs.jsonl \
  --base-checkpoint photofinishing/models/default.pth \
  --output-dir runs/reference_style_v1 \
  --tone-epochs 10 \
  --chroma-epochs 10 \
  --tone-lr 1e-5 \
  --chroma-lr 1e-5
```

Outputs:

```text
runs/reference_style_v1/
├── tone_best.pth
├── tone_last.pth
├── chroma_best.pth
├── chroma_last.pth
└── training_summary.json
```

## Evaluate

```bash
python photofinishing/evaluate_reference_style.py \
  --manifest /path/to/reference_pairs.jsonl \
  --base-checkpoint photofinishing/models/default.pth \
  --tone-checkpoint runs/reference_style_v1/tone_best.pth \
  --chroma-checkpoint runs/reference_style_v1/chroma_best.pth \
  --split test \
  --output-json runs/reference_style_v1/test_metrics.json
```

The report compares:

- base versus tone-stage luminance-style loss;
- base versus final chroma-style loss;
- final-output luminance change relative to the frozen tone-stage result.

## First experiment acceptance

The experiment is promising only if a scene-disjoint test set shows:

1. tone-stage luminance-style loss decreases relative to the base model;
2. final chroma-style loss decreases relative to the base and tone checkpoints;
3. final luminance remains close to the tone-stage anchor;
4. blind human comparison confirms that reduced statistical distance corresponds to the intended A-camera rendering style;
5. normal B-domain good cases and source-domain replay do not show material regression.

## Evidence boundary

This code verifies staged parameter freezing and enables a first non-pixel reference-loss experiment. It does not establish that the same-scene references are compositionally similar enough, that the learned style is perceptually correct, or that the resulting model can safely generate final pixel pseudo-GT. Those questions require a real scene-disjoint experiment and independent blind evaluation.