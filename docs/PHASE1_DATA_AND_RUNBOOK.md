# Phase-1 Data and Execution Runbook

This runbook implements the frozen requirements in `docs/CROSS_CAMERA_REQUIREMENT_BASELINE.md`.

## 1. Tensor contract

Image artifacts are local `.pt` files containing either a `torch.Tensor` or `{"tensor": torch.Tensor}`.

Required image contract:

```text
shape: [1, 3, H, W]
dtype after load: float32
values: finite and non-negative
encoding: BLC + WB linear RGB
```

ROI/alignment masks use `[1, 1, H, W]`. Manifest hashes use `cross_camera_tm.contracts.canonical_tensor_sha256`, not the raw `.pt` file hash.

## 2. Metadata contract

Each image has a strict `LinearMetadata` JSON object. Boolean fields must be JSON booleans rather than strings.

Formal binding rules:

- source metadata device must contain `Samsung`;
- calibration iPhone metadata device must contain `iPhone`;
- calibration Samsung metadata device must contain `Samsung`;
- source metadata `sample_id` must equal the manifest source `sample_id`;
- every metadata `sample_id` must equal the corresponding tensor filename stem;
- sample IDs and scene groups must be non-empty;
- metadata sample IDs must be unique within their role.

Set `awb_gains_comparable=true` only when applied/reference gains share a valid coordinate definition.

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

Requirements:

- at least ten source samples;
- at least five independent source scene groups;
- unique source sample IDs;
- unique Samsung source image hashes;
- finite non-negative Samsung input and GT;
- Samsung input and GT shapes must match.

Duplicating one source image under a different ID is rejected because it would distort the teacher qualification distribution.

## 4. Cross-device calibration manifest

Exactly 50 pairs are required:

```text
40 development
10 locked
```

Locked scene groups must not occur in development.

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

Dataset-level independence rules:

- pair IDs are unique;
- complete pair content signatures are unique;
- development and locked iPhone hashes are disjoint;
- development and locked Samsung hashes are disjoint;
- development and locked GT hashes are disjoint;
- exact content duplication under renamed IDs is rejected;
- all tensors are finite, non-negative and share the declared canvas.

## 5. Alignment evidence

Declared levels:

```text
scene_only
roi
low_frequency
```

The label is only an upper bound. A frozen `AlignmentPolicy` checks overlap, valid ROI fraction, forward-backward consistency and residual displacement, and may downgrade:

```text
low_frequency → roi → scene_only
```

It never upgrades a weaker declaration. `roi` requires `roi_mask`; `low_frequency` requires both `roi_mask` and `alignment_mask`. All alignment thresholds must be finite.

## 6. Train Phase 1

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

The authoritative training implementation is `cross_camera_tm.phase1_protocol.train_phase1`. `phase1_training.py` contains shared primitives only.

Runtime support and minimum Adapter-margin thresholds are calibrated from the 40 development pairs only. The locked ten pairs do not set deployment thresholds.

## 7. Re-evaluate the locked calibration set

```bash
PYTHONPATH=. python main/run_cross_camera_adaptation.py evaluate-phase1 \
  --config configs/cross_camera_tm_v2.real.example.yaml \
  --calibration-manifest /data/cross_camera/calibration_manifest.json \
  --adapter-checkpoint outputs/cross_camera_phase1/phase1_adapter.pt \
  --output-dir outputs/cross_camera_phase1_eval
```

The command rejects synthetic artifacts, model/config/policy mismatch, a different calibration manifest and artifacts without real Phase-1 calibration acceptance.

## 8. Run a real iPhone input

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

Inference fails closed when the artifact is synthetic or failed, provenance differs, input contracts are invalid, support distance is non-finite/outside support, or Adapter margin is non-finite/non-positive/below the frozen threshold.

## 9. Phase-2 boundary

Real Phase 2 is prohibited at both configuration and library API boundaries.

```text
real config + phase2.enabled=true → PHASE2_NOT_IMPLEMENTED
real config + pixel routing       → PIXEL_ROUTING_NOT_IMPLEMENTED
CrossCameraPipeline.run(synthetic=False, phase2_enabled=True)
                                  → PHASE2_NOT_IMPLEMENTED
```

Synthetic canaries remain mechanical tests only and cannot establish real cross-device effectiveness.

## 10. Verification

```bash
bash scripts/run_cross_camera_domain_adaptation_verification.sh
```

CI additionally runs focused cross-camera tests, full repository unittest discovery, config validation and the Samsung checkpoint interface canary.
