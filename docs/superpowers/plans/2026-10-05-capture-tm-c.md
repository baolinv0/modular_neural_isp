# Capture-TM C Implementation Plan

> **For agentic workers:** Use parallel implementation on the disjoint ownership below, then independent AE/TM/ISP review. New numerical behavior and reported regressions use failing/passing tests; CLI workflows also receive actual end-to-end runs.

**Goal:** Physical exposure candidates, two patent-inspired TM paths, an AdaptiveAE-inspired observation ranker and a runnable dataset/oracle/train/evaluate/infer workflow.

**Architecture:** Preserve upstream PhotofinishingModule and checkpoints; add `capture_tm`. Freeze the renderer while producing candidate labels and training the causal policy. All comparisons share intent, source split, observations and legal actions.

**Tech Stack:** PyTorch 2.5+, numpy, Pillow, YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-capture-tm-c-design.md`.

## Global constraints

- Capture time is seconds, gain is multiplicative, EV is log2 scale; render gain is separate.
- Scene tensors are sensor-linear relative radiance; encoded XYZ/sRGB requires explicit conversion.
- No clean radiance or future candidate images in policy inference.
- No random renderer fallback, no hardware or real-quality claims from synthetic tests.

## Review focus

- Gain/ISO changes must not create photons or blur.
- Temporal support, source split leakage and frozen target appearance.
- Clipping provenance before TM and duplicated gain.
- Checkpoint load, frozen renderer and masked invalid candidates.
- Missing calibration, OETF-encoded RawGen XYZ and relative proxy limitations.

## Tasks and ownership

- [x] Data/simulator: `types.py`, `simulation.py`, `data.py`, `sources.py`; physical/noise/motion/split/import tests.
- [x] TM: `tone.py`, `modular.py`; compensation/intent/HDR/monotonicity/frozen-checkpoint tests.
- [x] Policy/control: `policy.py`, `control.py`; causal observations/gradient/masking/delay tests.
- [x] Root integration: `objective.py`, `experiment.py`, `cli.py`, configuration and integration tests; build labels, train, checkpoint reload, evaluate, infer, direct render and comparison report. No optimizer resume claim.
- [x] Run complete new package tests and upstream collection; record dependency or pre-existing test limitations.
- [x] Independent AE/TM/ISP review, reproduce reported defects, fix, rerun affected tests and full suite.
- [ ] Commit and publish an isolated GitHub branch and draft PR with source mapping, runnable commands and verification limits.
