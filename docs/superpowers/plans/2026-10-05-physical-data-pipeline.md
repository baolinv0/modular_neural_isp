# Physical data pipeline implementation plan

> **For agentic workers:** Use superpowers:executing-plans task by task; parallel workers have disjoint files. The user's implementation request supplies execution authorization.

**Goal:** Export Bayer RAW capture datasets for Apple single frame and Samsung HDR and run existing factorial AE/TM studies from them.
**Architecture:** A shared acquisition model, explicit source importer, relocatable archive/CLI, and existing training adapter.
**Tech Stack:** Python, PyTorch, NumPy, optional imageio/OpenEXR, pytest.
**Spec:** docs/superpowers/specs/2026-10-05-physical-data-pipeline.md

## Global Constraints
No fabricated target-phone calibration or training gains. Keep legacy RGB compatibility. Explicit radiometry, common fixed reference, source-group splits, no future policy inputs, capture digital gain 1, virtual gain only inside TM.

## Review Focus
- Non-RGGB CFA phase and green interpolation.
- Scene support insufficient for rolling shutter/readout.
- Same original source via different files/splits.
- Relocated archives and partial generation failures.
- Per-candidate target/radiance normalization or virtual gain applied twice.

### Task 1: Physical acquisition
Files: acquisition.py, tests/test_capture_tm_acquisition.py.
- [x] Write and run failing Bayer/photons/integration/reference/readout tests.
- [x] Implement AcquisitionProfile, RawCaptureResult, prepare_scene, capture_raw.
- [x] Run focused tests and report limits.

### Task 2: Source import
Files: pipeline_sources.py, tests/test_capture_tm_pipeline_sources.py.
- [x] Write and run failing domain/layout/time/source grouping/relative-scale tests.
- [x] Implement recipe import into existing Scene manifest; support float HDR arrays and sequences.
- [x] Run focused tests with real encoded fixtures.

### Task 3: Archive and CLI
Files: pipeline.py, pipeline_cli.py, configs/capture_tm/*.json, tests/test_capture_tm_pipeline.py.
- [x] Write and run failing export/causality/reproducibility/relocation/no-overwrite tests.
- [x] Generate shared previews, fixed references and candidate banks with native RAW and metadata; validate schemas.
- [x] Run demo generation and validator.

### Task 4: Training and deployment bridge
Files: joint_experiment.py, joint_algorithm.py, tests/test_capture_tm_pipeline_training.py.
- [x] Write and run failing archived-input all-group training/checkpoint tests.
- [x] Prepare frozen ISP caches directly from archive; bind acquisition to checkpoints.
- [x] Run all eight groups and checkpoint select/finish inference.

### Task 5: Review, docs and delivery
- [x] Run entire suite and small end-to-end acquisition/training example.
- [x] Independent reviewer checks physics, grouping and integration; fix concrete issues.
- [x] Save usage and validation evidence in repository.
- [ ] Commit and update target branch without force; read remote commit back.
