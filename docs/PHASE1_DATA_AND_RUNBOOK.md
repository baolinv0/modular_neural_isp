# Phase-1 Data and Execution Runbook

This runbook implements the frozen requirements in `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`.

## 1. Tensor files

All image artifacts are local `.pt` files containing either:

```python
torch.Tensor
```

or:

```python
{"tensor": torch.Tensor}
```

Image shape must be `[1, 3, H, W]`. ROI/alignment masks use `[1, 1, H, W]`. Tensors must be finite. Images are non-negative BLC+WB linear RGB; metadata declares whether white-level normalization is still required.

Manifest tensor hashes use `cross_camera_tm.contracts.canonical_tensor_sha256`, not the raw `.pt` file hash.

## 2. Linear metadata JSON

Each image has one strict metadata object:

```json
{
  "sample_id": "iphone-0001",
  "device": "iPhone",
  "white_level": 65535.0,
  "is_normalized": false,
  "black_level_corrected": true,
  "white_balanced": true,
  "awb_gains_applied": [2.1, 1.0, 1.7],
  "reference_awb_gains": null,
  "awb_gains_comparable": false,
  "ccm_to_common": null,
  "exposure_time_s": 0.01,
  "iso": 100.0,
  "aperture": 1.8,
  "reference_exposure_product": 0.3086419753,
  "hdr_confidence": 0.9,
  "metadata_complete": true
}
```

Boolean fields must be JSON booleans, not strings.

Set `awb_gains_comparable=true` only when applied/reference gains share a valid coordinate definition. If sensor-RGB coordinates differ and no reliable CCM is available, use the metadata gains only to describe/undo the known device operation; do not insert a cross-device gain ratio.

## 3. Samsung source manifest

```json
{
  "schema_version": 1,
  "samples": [
    {
      "sample_id": "source-0001",
      "scene_group": "source-scene-0001",
      "samsung_tensor": "tensors/source-0001.pt",
      "samsung_gt_tensor": "tensors/source-0001-gt.pt",
      "metadata": "metadata/source-0001.json",
      "samsung_sha256": "<canonical tensor sha256>",
      "gt_sha256": "<canonical tensor sha256>"
    }
  ]
}
```

At least ten independent Samsung source samples are required for teacher P75/P90 qualification. A larger held-out source set is preferred.

## 4. Cross-device calibration manifest

Exactly 50 pairs are required: 40 `development`, 10 `locked`. Locked scene groups must not occur in development.

```json
{
  "schema_version": 1,
  "pairs": [
    {
      "pair_id": "pair-0001",
      "scene_group": "scene-0001",
      "split": "development",
      "iphone_tensor": "tensors/iphone-0001.pt",
      "samsung_tensor": "tensors/samsung-0001.pt",
      "samsung_gt_tensor": "tensors/samsung-0001-gt.pt",
      "iphone_metadata": "metadata/iphone-0001.json",
      "samsung_metadata": "metadata/samsung-0001.json",
      "alignment": {
        "quality": "scene_only",
        "overlap": 0.82,
        "forward_backward_consistency": 0.88,
        "valid_roi_fraction": 0.74,
        "residual_displacement_px": 4.3
      },
      "roi_mask": null,
      "alignment_mask": null,
      "iphone_sha256": "<canonical tensor sha256>",
      "samsung_sha256": "<canonical tensor sha256>",
      "gt_sha256": "<canonical tensor sha256>"
    }
  ]
}
```

Legal alignment levels:

```text
scene_only
roi
low_frequency
```

- `roi` requires `roi_mask`.
- `low_frequency` requires both `roi_mask` and `alignment_mask`.
- All images must already be resampled/warped to the same tensor canvas. The loss level, not a claimed pixel threshold, determines how that canvas may be used.

## 5. Train Phase 1

```bash
PYTHONPATH=. python main/run_cross_camera_adaptation.py train-phase1 \
  --config configs/cross_camera_tm_v2.real.example.yaml \
  --source-manifest /data/cross_camera/source_manifest.json \
  --calibration-manifest /data/cross_camera/calibration_manifest.json \
  --output-dir outputs/cross_camera_phase1
```

Outputs:

```text
phase1_adapter.pt
phase1_training_report.json
```

Exit status `0` means the mechanical and acceptance conditions encoded by the frozen protocol passed. It does not replace human inspection of real image quality.

## 6. Re-evaluate locked calibration data

```bash
PYTHONPATH=. python main/run_cross_camera_adaptation.py evaluate-phase1 \
  --config configs/cross_camera_tm_v2.real.example.yaml \
  --calibration-manifest /data/cross_camera/calibration_manifest.json \
  --adapter-checkpoint outputs/cross_camera_phase1/phase1_adapter.pt \
  --output-dir outputs/cross_camera_phase1_eval
```

This command reloads the versioned artifact and re-runs the locked-pair observable-output evaluation.

## 7. Run a real iPhone input

```bash
PYTHONPATH=. python main/run_cross_camera_adaptation.py real-run \
  --config configs/cross_camera_tm_v2.real.example.yaml \
  --adapter-checkpoint outputs/cross_camera_phase1/phase1_adapter.pt \
  --input /data/iphone/sample.pt \
  --metadata /data/iphone/sample.json \
  --output-dir outputs/cross_camera_inference/sample
```

Outputs:

```text
phase1_output.pt
run_manifest.json
```

Inference fails closed when:

- the artifact did not pass locked Phase-1 acceptance;
- the Samsung checkpoint hash differs;
- metadata/tensor contracts are invalid;
- the input lies beyond the configured calibration-support distance.

`real-run` executes Phase 1 only. Phase 2 remains blocked and is recorded as such in the run manifest.
