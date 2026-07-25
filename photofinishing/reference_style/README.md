# Non-pixel-aligned reference-style training

This experiment uses same-scene but unaligned `B_out / A_GT` pairs. It does **not** create a pixel pseudo-GT and never applies a pixelwise loss between the two images.

## Stage 1: luminance style

Train only `_gain_net` and `_gtm_net`. Freeze LTM, chroma LUT, optional 3D LUT, and gamma. The loss matches luminance distributions using a soft histogram, moments, shadow/midtone/highlight occupancy, and global contrast. A normalized-gradient anchor to the pretrained output limits structural changes.

## Stage 2: chroma style

Start from the Stage-1 checkpoint. Freeze Gain, GTM, LTM, and Gamma. Train `_lut_net` and optionally `_3d_lut`. The loss matches Cb/Cr distributions and moments while preserving the Stage-1 luminance and neutral regions.

## Manifest

```json
{"sample_id":"scene_001","source":"/data/B_out/001.png","reference":"/data/A_GT/001.png"}
```

The two images may have different spatial resolutions. Batch size is fixed to one in v1.

## Run

```bash
python photofinishing/train_unaligned_reference.py \
  --manifest /data/train.jsonl \
  --model-path photofinishing/models/default/model.pth \
  --output-dir runs/reference_style_v1 \
  --luma-epochs 5 \
  --chroma-epochs 5
```

Outputs:

- `reference_style_luma.pth`
- `reference_style_chroma.pth`
- `reference_style_training_report.json`

## Evidence boundary

This is an experiment for whether unaligned distribution supervision can move photofinishing toward same-scene reference style. It does not prove pixel-level pseudo-GT validity or cross-camera holdout improvement.
