# Phase-1 Data and Execution Runbook

This runbook implements the frozen requirements in `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`.

## 1. Tensor files

All image artifacts are local `.pt` files containing either a `torch.Tensor` or `{"tensor": torch.Tensor}`.

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

Set `awb_gains_comparable=true` only when applied/reference gains share a valid coordinate definition. If sensor-RGB coordinates differ and no reliable CCM is available, metadata gains may describe or undo a known device operation, but must not create an unsupported cross-device gain ratio.

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

The declared quality is only an upper bound. The supported CLI derives an effective quality from the numeric evidence and may downgrade it.

Frozen default evidence policy:

```text
ROI supervision:
  overlap >= 0.50
  valid_roi_fraction >= 0.30

Low-frequency supervision:
  overlap >= 0.70
  valid_roi_fraction >= 0.50
  forward_backward_consistency >= 0.80
  residual_displacement_px <= 2.0
```

If low-frequency evidence fails but ROI evidence passes, the pair is downgraded to ROI. If ROI evidence also fails, it is downgraded to scene-only tone/statistics supervision. Numeric evidence never upgrades a weaker declared label.

- `roi` requires `roi_mask`.
- `low_frequency` requires both `roi_mask` and `alignment_mask`.
- Approximate pairs never enable full-resolution pixel loss.

## 5. Train Phase 1

```bash
PYTHONPATH=. python main/run_cross_camera_adaptation.py train-phase1 \
  --config configs/cross_camera_tm_v2.real.example.yaml \
  --source-manifest /data/cross_camera/source_manifest.json \
  --calibration-manifest /data/cross_camera/calibration_manifest.json \
  --output-dir outputs/cross_camera_phase1
```

`train-phase1` in real mode always produces a real-data artifact. There is no CLI switch that can relabel it as synthetic.

Outputs:

```text
phase1_adapter.pt
phase1_training_report.json
```

The saved schema-2 artifact binds:

- Samsung checkpoint SHA;
- source and calibration manifest SHA;
- feature schema and normalization;
- canonicalization configuration and SHA;
- alignment policy and SHA;
- development calibration support geometry;
- frozen maximum support distance;
- frozen minimum Adapter parameter-bound margin;
- validation report and evidence labels.

The support threshold is calibrated from leave-one-scene-group-out development distances. The Adapter margin threshold is calibrated from qualified calibration samples. Neither threshold can be widened at inference time.

## 6. Re-evaluate locked calibration data

```bash
PYTHONPATH=. python main/run_cross_camera_adaptation.py evaluate-phase1 \
  --config configs/cross_camera_tm_v2.real.example.yaml \
  --calibration-manifest /data/cross_camera/calibration_manifest.json \
  --adapter-checkpoint outputs/cross_camera_phase1/phase1_adapter.pt \
  --output-dir outputs/cross_camera_phase1_eval
```

Evaluation rejects:

- a different Samsung checkpoint;
- a different calibration manifest;
- a different canonicalization configuration;
- a different alignment policy;
- an invalid or unsealed artifact.

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
- the artifact is synthetic;
- the Samsung checkpoint hash differs;
- canonicalization or alignment-policy identity differs;
- metadata/tensor contracts are invalid;
- calibration-support distance exceeds the artifact-frozen threshold;
- predicted Adapter parameters approach or cross the calibrated boundary threshold.

The run manifest separates evidence levels:

```text
real_phase1_calibration_accepted
real_source_replay_verified
real_target_effectiveness_verified
```

Passing the 40+10 calibration protocol can set only the first field. The latter two remain false until independent source replay and target holdout evaluations are actually executed.

## 8. Phase-2 boundary

Real configuration rejects either of the following:

```text
phase2.enabled=true        -> PHASE2_NOT_IMPLEMENTED
pixel_route_enabled=true   -> PIXEL_ROUTING_NOT_IMPLEMENTED
```

Synthetic canaries may exercise experimental Phase-2 interfaces, but they cannot authorize real routing. `real-run` executes Phase 1 only and records `phase2_status=PHASE2_NOT_IMPLEMENTED`.
