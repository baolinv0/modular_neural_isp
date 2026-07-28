# Non-Aligned Photofinishing Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and execute each task with RED/GREEN verification.

**Goal:** Evaluate multiple pretrained, Stage-1, and Stage-2 Photofinishing checkpoints against same-scene non-pixel-aligned references, then answer whether Stage 1 improves luminance, whether Stage 2 is necessary/effective, which Stage-2 capacity is preferable, and whether more evaluation or training data is required.

**Architecture:** A JSON experiment config defines multiple checkpoint groups. An extended CSV manifest provides target-camera inputs, reference-camera images, optional repeated references, valid/ignore masks, semantic masks, and scene tags. The evaluator runs every checkpoint on the same inputs, computes position-independent luminance and chroma metrics, summarizes paired incremental improvements with bootstrap confidence intervals, writes machine-readable reports, and renders comparison panels.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, NumPy 1.26.4, OpenCV 4.11, pytest. No SciPy, scikit-image, or network dependency.

## Global Constraints

- Preserve `photofinishing/eval_unpaired_style.py` and existing training behavior.
- Support multiple experiment groups and multiple Stage-2 variants per group.
- Stage-1 improvement is always `distance(pretrained, reference) - distance(stage1, reference)`.
- Stage-2 improvement is always `distance(stage1, reference) - distance(stage2, reference)`.
- Lower metric values are better; positive incremental improvement means improvement.
- Metrics must not assume pixel correspondence.
- Semantic metrics are computed only when separate input/reference semantic masks are supplied.
- Affine checkpoints must use the existing fail-closed run-config reconstruction and SHA validation.
- Reports must expose unavailable evidence as `unavailable` or `undetermined`, never silently fabricate it.
- The first implementation must run with the repository's existing CI dependencies.

---

## Task 1: Configuration and Extended Manifest

**Files:**
- Create: `photofinishing/evaluate/config.py`
- Create: `photofinishing/evaluate/data.py`
- Test: `photofinishing/evaluate/tests/test_config_data.py`

**Produces:**
- `load_experiment_config(path) -> EvaluationConfig`
- `load_evaluation_manifest(path, split) -> list[EvaluationRecord]`
- Multiple checkpoint groups, multiple Stage-2 variants, repeated references, valid/ignore masks, semantic masks, and scene tags.

## Task 2: Non-Aligned Luminance and Chroma Metrics

**Files:**
- Create: `photofinishing/evaluate/metrics.py`
- Test: `photofinishing/evaluate/tests/test_metrics.py`

**Produces:**
- Stage-1 metrics: signed/absolute EV error, log-luma quantile MAE, exposure-normalized tone-shape MAE, log-luma Wasserstein, shadow/highlight ratio error, clipping-ratio error.
- Stage-2 metrics: CbCr sliced Wasserstein, luminance-conditioned CbCr SWD, chroma mean/covariance errors, saturation Wasserstein, neutral-axis error, optional skin/sky/vegetation Lab SWD.
- Reference-repeat noise-floor metrics when provided.

## Task 3: Paired Statistics and Bootstrap

**Files:**
- Create: `photofinishing/evaluate/statistics.py`
- Test: `photofinishing/evaluate/tests/test_statistics.py`

**Produces:**
- Mean, median, win rate, negative-improvement rate, p10, standard deviation, and paired bootstrap 95% confidence interval.
- Deterministic behavior from an explicit seed.

## Task 4: Stage Decisions and Data Expansion Recommendation

**Files:**
- Create: `photofinishing/evaluate/decision.py`
- Test: `photofinishing/evaluate/tests/test_decision.py`

**Produces:**
- Stage-1 effectiveness decision.
- Stage-2 necessity decision against reference-repeat noise floor.
- Stage-2 effectiveness decision with luminance-regression gates.
- Stage-2 variant recommendation.
- Separate recommendations for expanding evaluation data and training data, including reasons and failure slices.

## Task 5: Multi-Checkpoint Evaluation and Reporting

**Files:**
- Create: `photofinishing/evaluate/model_loader.py`
- Create: `photofinishing/evaluate/reporting.py`
- Create: `photofinishing/evaluate/run_evaluation.py`
- Create: `photofinishing/evaluate/__init__.py`
- Test: `photofinishing/evaluate/tests/test_reporting_pipeline.py`

**Produces:**
- Inference for pretrained, Stage-1, and all Stage-2 variants.
- Per-sample metric CSV, incremental-improvement CSV, summary CSV/JSON, decision JSON, and comparison panels.
- Fail-closed errors for invalid paths, duplicate labels, incompatible checkpoints, empty splits, and unavailable semantic evidence.

## Task 6: Documentation, Examples, and CI

**Files:**
- Create: `photofinishing/evaluate/README.md`
- Create: `photofinishing/evaluate/example_experiments.json`
- Create: `photofinishing/evaluate/example_manifest.csv`
- Modify: `.github/workflows/unpaired-photofinishing.yml`

**Verification:**
- `python -m compileall -q photofinishing/evaluate`
- `pytest -q photofinishing/evaluate/tests`
- Existing focused unpaired tests.
- Existing real Photofinishing interface canary.

## Final Requirement Audit

Before completion, verify explicitly that the implementation covers:

1. Multiple pretrained/Stage-1/Stage-2 groups.
2. Same-scene non-pixel-aligned references.
3. Stage-1 luminance and tone consistency.
4. Stage-2 color consistency independent of luminance.
5. Stage-1 and Stage-2 incremental improvements.
6. Reference noise floor.
7. Optional common-FoV, ignore, and semantic masks.
8. Mean/median/win-rate/negative-rate/p10/bootstrap CI.
9. Stage-1 effectiveness, Stage-2 necessity/effectiveness, and model-capacity decisions.
10. Evaluation-data and training-data expansion recommendations.
11. Per-scene results, slice results, machine-readable summaries, and visual comparison panels.
12. Backward compatibility with the existing Photofinishing code and affine checkpoint provenance rules.
