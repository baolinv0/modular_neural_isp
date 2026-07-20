# Cross-Camera Samsung-Style Domain Adaptation Implementation Report

## Outcome

The repository now contains a bounded two-phase path that maps already black-level-corrected and device-white-balanced iPhone linear RGB toward the input behavior expected by the frozen Samsung-trained photofinishing model. The Samsung model remains the style carrier. Phase 1 learns device-input compensation from the 50-pair contract; Phase 2 is conditional and can learn only a supported luminance residual.

This is a mechanically verified implementation, not evidence of real iPhone improvement. The real calibration dataset and target holdouts were unavailable.

## First-principles correction to the original open-loop proposal

The original pseudo-GT scheme could establish safety but not the direction of Samsung-to-iPhone domain compensation. This implementation adds three independent constraints without leaving the Modular Neural ISP line:

1. `F_s(C_s(X_s))` is the style teacher for same-scene pairs, while Samsung GT qualifies that teacher.
2. Source-domain residual distributions anchor the legal magnitude, interval, support and dynamic ROI used in Phase 2.
3. Both adapters have intentionally small, physically interpretable hypothesis spaces; the Samsung backbone is frozen.

Source consistency can falsify an invalid direction, but cannot prove target correctness. Therefore real completion requires development/locked calibration results, target holdout improvement and source non-regression.

## Implemented data flow

```text
iPhone BLC+WB linear RGB
  -> strict metadata/hash contract
  -> metadata-derived canonicalization
  -> TargetCameraAdapter (gain -> 3x3 residual -> monotonic curve)
  -> frozen Samsung TM
  -> Phase 2 activation/support check
  -> lowest sufficient L1 / L2 / optional local L3 proposal
  -> L3 pre-projection safety + TM-space projection
  -> full 13-gate certification
  -> structured uncertainty + supervision router
  -> hash-bound manifest and lineage
```

Color-domain input mismatch is handled only by the bounded Phase 1 gain and near-identity 3x3 matrix. Phase 2 intentionally remains luminance-only. Consequently L3 color, geometry, texture and invented detail are discarded rather than becoming a pixel target.

## Safety and direction rules

- Qwen3-VL may classify a supported failure and supply ROI/highlight-risk hints; it has no EV or correction-strength field.
- Numerical correction comes from a source-calibrated residual estimator. OOD queries return `UNAVAILABLE`.
- L1 is global; L2 applies a distinct feathered face/background field; L3 is an appearance proposal only.
- InternVL AB/BA and optional Ovis results are separate evidence. Neither overall IQA nor a VLM preference can accept a target.
- Uncertainty fields are not added into a scalar. Routing uses necessary conditions.
- Any critical `FAIL` or `UNAVAILABLE` prevents acceptance. Gate results cannot compensate each other.
- Raw generated images are rejected for pixel supervision. L3 output must receive a new projected hash and a fresh full certification.
- A synthetic profile can certify a synthetic canary only. It cannot authorize real pixel routing.

## Verification environment

- Baseline: `efbfdfea87385254cb36e23354fa9b7f1ae2e4ce`
- Working branch: `feature/cross-camera-domain-adaptation-v2`
- PyTorch: `2.5.1+cpu` from `/tmp/cross-camera-torch`
- CUDA/GPU: unavailable
- Real Samsung checkpoint: available and exercised on CPU
- Real 50-pair calibration set: unavailable
- Real iPhone target/holdout set: unavailable
- Real Qwen3-VL, Qwen-Image-Edit, InternVL and Ovis checkpoints: unavailable

## Verification commands and actual results

Final evidence is produced by:

```bash
scripts/run_cross_camera_domain_adaptation_verification.sh
```

The script performs:

- `python -m compileall -q cross_camera_tm main tests scripts`;
- `python -m unittest discover -s tests -v` with zero skip/xfail policy;
- strict YAML validation;
- two deterministic Synthetic Canary executions and semantic JSON comparison;
- manifest/report existence checks;
- strict load, CPU forward, frozen-weight and non-zero input-gradient checks for `photofinishing_s24-style-0.pth`.

Fresh post-documentation run on 2026-07-21:

- exit code: `0`;
- compileall: `PASS`;
- unittest: `77 tests`, `OK`, zero failures/errors/skips/xfails;
- strict config: `valid`, config SHA-256 `5da90a3b3b3809083f2b6bc52f7e5aac844f05c5189470ab8e7383ca0009707b`;
- Synthetic Canary determinism: `PASS`, route `parameter`, certification `true`, candidate SHA-256 `12e92bf88de874d147a3a4609be4ebf5ee0ef02fb818bc3995a272b721ec2e21`;
- real Samsung checkpoint interface canary: `PASS`, strict load, finite output/gradient, non-zero input gradient, zero trainable model parameters;
- output validation: `PASS`;
- real-data effectiveness: `UNVERIFIED`.

## Synthetic Canary

Expected and mechanically observed invariants:

- `synthetic: true`;
- `real_model: false`;
- `real_data_effectiveness_verified: false`;
- deterministic candidate SHA across two runs;
- Phase 2 activation criteria execute on 60 deterministic synthetic records;
- all 13 gates pass for the analytic L1 fixture;
- route is `parameter`, not pixel;
- the manifest records synthetic/model/profile/config/input flags and hashes.

The L3 integration test separately forces L1 failure, then observes raw proposal -> safety -> TM projection -> second full certification -> preference route. It verifies `raw_generated=false` and `projected=true` in the final manifest.

## Real checkpoint canary

The included Samsung checkpoint is a real repository artifact, not a mock. The canary verifies strict state-dict loading, a finite `[1,3,64,64]` output, zero trainable model parameters, and a finite non-zero input gradient. This proves interface and differentiability compatibility only. It does not qualify the model for cross-camera use or validate any real data result.

## Unverified items

1. Whether 50 balanced pairs cover the actual iPhone/Samsung device residual.
2. Group-aware five-fold and locked-ten improvement on the real calibration set.
3. Teacher qualification and source residual/gate distributions on independent Samsung paired data.
4. Target holdout improvement and source-domain non-regression.
5. Real local VLM/editor/arbiter throughput and perceptual calibration.
6. Whether parameter/range supervision matches or exceeds pixel supervision; production pixel routing remains disabled until this ablation exists.

## Builder verdict

`MECHANICALLY_READY_WITH_REAL_DATA_BLOCKERS` — the bounded interfaces, failure semantics, tests, CLI and verification path are implemented. The system must not be described as having learned or improved real iPhone-to-Samsung behavior until the frozen real-data protocol passes.
