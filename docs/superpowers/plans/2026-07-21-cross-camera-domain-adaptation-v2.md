# Cross-Camera Samsung-Style Domain Adaptation v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal, complete and mechanically verifiable two-phase iPhone-to-Samsung TM domain-adaptation path to the existing Modular Neural ISP repository.

**Architecture:** Canonicalize metadata-derivable device differences, train a bounded low-capacity input adapter by same-scene Samsung-teacher distillation, then optionally train a narrower luminance-only residual adapter when a source-grounded failure remains. Keep the Samsung photofinishing model frozen as the style carrier and fail closed whenever source support or calibrated evidence is unavailable.

**Tech Stack:** Python 3.12, PyTorch 2.5, standard-library dataclasses/JSON/argparse, PyYAML, unittest, Bash.

## Global Constraints

- Baseline is `baolinv0/modular_neural_isp@efbfdfea87385254cb36e23354fa9b7f1ae2e4ce`.
- Inputs are already black-level-corrected and white-balanced 16-bit linear RGB; do not re-run black-level correction or full AWB estimation.
- Use the 50 balanced same-scene approximate pairs only through alignment-qualified statistical, ROI and low-frequency losses; never assume pixel alignment.
- Freeze the Samsung TM backbone. Phase 2 also freezes canonicalization and TargetCameraAdapter.
- No full color-style transfer, 3A, multi-frame HDR, denoising, deblurring, super-resolution, online update, RL or end-to-end backbone training.
- No unavailable critical gate auto-pass, no overall-IQA acceptance, no VLM approval, no raw generated pixel target and no real pixel route without a real calibration profile.
- Synthetic/deterministic validation must be labeled and must not be presented as real iPhone improvement.

---

### Task 1: Strict contracts and deterministic canonicalization

**Files:**
- Create: `cross_camera_tm/contracts.py`
- Create: `cross_camera_tm/canonicalization.py`
- Create: `cross_camera_tm/__init__.py`
- Create: `tests/test_cross_camera_contracts_canonicalization.py`
- Create: `.gitignore`

**Interfaces:**
- Produces `LinearMetadata`, `AlignmentQuality`, `FailureType`, `GateStatus`, `CanonicalizationResult`, `DeviceCanonicalizer.canonicalize(image, metadata)` and canonical tensor hashing.
- Consumed by all later Phase 1/2 modules.

- [ ] Write tests for unknown-field rejection, non-finite/range rejection, 16-bit normalization, known AWB gain alignment, CCM/exposure ordering, confidence degradation and reliable masks.
- [ ] Run `PYTHONPATH=/tmp/cross-camera-torch:. python -m unittest tests.test_cross_camera_contracts_canonicalization -v` and confirm imports/behaviors fail because the package is absent.
- [ ] Implement immutable strict dataclasses and canonicalization in the frozen order.
- [ ] Re-run the focused command and confirm all tests pass.

### Task 2: Low-capacity TargetCameraAdapter and Phase 1 loss

**Files:**
- Create: `cross_camera_tm/adapters.py`
- Create: `cross_camera_tm/losses.py`
- Create: `cross_camera_tm/phase1.py`
- Create: `tests/test_cross_camera_phase1.py`

**Interfaces:**
- Produces `TargetCameraAdapter`, `FrozenSamsungTM`, `TeacherQualifier`, `PairParameterSolver`, `DistillationLoss`, `gradient_canary` and Phase 1 result records.
- Consumes canonical tensors/metadata from Task 1 and any existing `torch.nn.Module` Samsung TM.

- [ ] Write tests for identity initialization, gain->matrix->curve order, monotonic curve, confidence gating, alignment-level loss enablement, P75/P90 teacher qualification, WLS/isotonic initialization and gradients through a frozen tiny TM.
- [ ] Run the focused tests and confirm RED for missing interfaces.
- [ ] Implement the minimal modules, retaining gradients through the frozen TM while freezing its parameters.
- [ ] Re-run focused tests to GREEN and then run the real repository Samsung checkpoint canary.

### Task 3: Source residual profile, fixed psi and Phase 2 activation

**Files:**
- Create: `cross_camera_tm/residuals.py`
- Create: `tests/test_cross_camera_residuals.py`

**Interfaces:**
- Produces `PsiFeatureExtractor`, `SourceResidualProfile`, `SourceResidualEstimator`, `ResidualEstimate`, `DynamicROIBuilder`, `DirectionAlignmentGate` and `assess_phase2_activation`.
- Consumed by teachers, certification and pipeline.

- [ ] Write tests for exact feature schema/order, calibrated nearest-neighbor support, OOD unavailable behavior, face/global dynamic ROI, vector interval/sign/distance/ratio conditions and stable-activation criteria.
- [ ] Run focused tests and confirm RED.
- [ ] Implement the low-capacity calibrated estimator and deterministic gates without high-dimensional embeddings.
- [ ] Re-run focused tests to GREEN.

### Task 4: Distinct teachers, projection and bounded TMResidualAdapter

**Files:**
- Create: `cross_camera_tm/teachers.py`
- Create: `cross_camera_tm/projection.py`
- Create: `cross_camera_tm/tm_residual.py`
- Create: `cross_camera_tm/policy.py`
- Create: `tests/test_cross_camera_teachers_projection.py`

**Interfaces:**
- Produces L1 global candidates, L2 ROI-conditioned candidates, local-only L3 proposal contracts, `TMSpaceProjector`, `TMResidualAdapter` and `OpenSourceLocalPolicy`.
- Consumed by certification/pipeline.

- [ ] Write tests proving L1 is global, L2 modifies face and background differently with smooth boundaries, raw L3 remains proposal-only, projection creates a new tensor hash, curve is monotonic, 8x8 gain is smooth and the residual adapter returns a full image with identity initialization.
- [ ] Run focused tests and confirm RED.
- [ ] Implement the teachers, local-only policy, mandatory L3 projection and 6-point/8x8 residual adapter.
- [ ] Re-run focused tests to GREEN.

### Task 5: Full certification, structured routing, manifests and lineage

**Files:**
- Create: `cross_camera_tm/certification.py`
- Create: `cross_camera_tm/routing.py`
- Create: `cross_camera_tm/lineage.py`
- Create: `cross_camera_tm/manifest.py`
- Create: `tests/test_cross_camera_certification_routing.py`

**Interfaces:**
- Produces `Certifier.certify`, `StructuredUncertainty`, `SupervisionRouter.route`, `LineageStore` and `ManifestWriter`.
- Consumes residual estimates, candidates, projections and calibration profiles.

- [ ] Write tests for all critical gate names, non-compensation, unavailable fail-closed, direction failure, boundary failure, projected full re-certification, raw-generated injection rejection, uncalibrated real pixel rejection and complete hashed lineage.
- [ ] Run focused tests and confirm RED.
- [ ] Implement the smallest deterministic certification/routing/provenance layer satisfying those failures.
- [ ] Re-run focused tests to GREEN.

### Task 6: End-to-end pipeline, strict config, CLI and deterministic canary

**Files:**
- Create: `cross_camera_tm/config.py`
- Create: `cross_camera_tm/pipeline.py`
- Create: `cross_camera_tm/canary.py`
- Create: `cross_camera_tm/cli.py`
- Create: `configs/cross_camera_tm_v2.yaml`
- Create: `main/run_cross_camera_adaptation.py`
- Create: `tests/test_cross_camera_pipeline_cli.py`
- Create: `scripts/compare_cross_camera_canary.py`
- Create: `scripts/run_cross_camera_domain_adaptation_verification.sh`

**Interfaces:**
- Produces strict config validation, Phase 1/2 orchestration, synthetic canary JSON, CLI exit codes and a unified verification entrypoint.

- [ ] Write tests for unknown config rejection, Phase 1->adapted baseline->Phase 2 order, fail-closed disabled Phase 2, deterministic rerun equality, CLI artifacts and explicit synthetic/real-model flags.
- [ ] Run focused tests and confirm RED.
- [ ] Implement orchestration and scripts using deterministic tensor fixtures; do not simulate real-data claims.
- [ ] Re-run focused tests and the exact CLI twice to GREEN.

### Task 7: Documentation, requirement mapping and fresh verification

**Files:**
- Create: `BUILD_REPORT.md`
- Modify: `capture_pipeline/README.md`

**Interfaces:**
- Records immutable baseline/upstream SHAs, files, commands, actual results, limitations and requirement-to-code/test mapping.

- [ ] Run `python -m compileall -q cross_camera_tm main tests scripts` with the temporary torch path active.
- [ ] Run all existing and new unittests with verbose output and confirm zero failures/skips/xfails.
- [ ] Run strict config validation, deterministic canary twice, exact comparison, output validation and real Samsung checkpoint CPU gradient canary.
- [ ] Run the unified verification script from a clean output directory and record its exact output.
- [ ] Inspect `git diff --check`, `git status`, changed files and frozen scope boundaries.
- [ ] Complete `BUILD_REPORT.md` with evidence and explicit unverified real-data items.
- [ ] Re-run the complete verification after the final report edit before creating the Builder commit.
