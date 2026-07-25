# Same-scene unpaired Photofinishing adaptation

This experiment uses a camera-B model input and a same-scene, non-pixel-aligned
camera-A product image. It never applies a pixel-wise loss between the two images.

## Protocol

1. **Stage 1:** train only `_gain_net` and `_gtm_net` with global and semantic-ROI
   luminance distribution losses. `_ltm_net`, `_lut_net`, `_gamma_net`, and the
   optional 3D LUT remain frozen.
2. **Stage 2:** freeze the complete Stage-1 luminance path and train only
   `_lut_net`. Chroma distribution losses are combined with an explicit luminance
   preservation loss against a frozen Stage-1 model.

## Manifest

Each JSONL row is strict and paths are resolved relative to the manifest:

```json
{
  "sample_id": "scene001_b",
  "scene_group": "scene001",
  "input_path": "inputs/scene001.npy",
  "reference_path": "references/scene001.png",
  "split": "train",
  "confidence": 0.9,
  "regions": {
    "face": {
      "input_mask": "masks_b/scene001_face.png",
      "reference_mask": "masks_a/scene001_face.png",
      "weight": 2.0
    }
  }
}
```

The input and reference masks are independent. No geometric correspondence is
assumed inside a semantic region.

## Run

```bash
python photofinishing/train_unpaired_style.py \
  --train-manifest /data/train.jsonl \
  --validation-manifest /data/validation.jsonl \
  --checkpoint photofinishing/models/default.pth \
  --config photofinishing/unpaired_style_default.json \
  --output-dir runs/unpaired_style_v1 \
  --device cuda
```

Use `--dry-run` to validate manifests, checkpoint compatibility, model forward,
and region contracts without optimization.

## Losses

Stage 1 uses log-exposure, luminance CDF/Wasserstein, percentiles, soft
shadow/midtone/highlight statistics, optional semantic-region CDF losses, and
content anchors to the frozen source output.

Stage 2 uses CbCr histogram and moment matching, saturation CDF, optional
semantic-region chroma losses, luminance and edge preservation against the frozen
Stage-1 model, and identity/TV/bounded-displacement regularization on the CbCr LUT.

## Evidence boundary

This implementation proves the training mechanics and module-freezing protocol.
It does not prove that global/ROI distribution matching improves real target-domain
images. Real validation must use scene-disjoint data and report failure/good-case
holdouts, clipping, shadow, hue, saturation, scene diversity, and human preference.
