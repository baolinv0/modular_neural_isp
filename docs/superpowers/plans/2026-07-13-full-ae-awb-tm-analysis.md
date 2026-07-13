# Full AE–AWB–TM Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stage-observable command-line pipeline that reuses the repository's existing RAW decoding, AWB/CCM estimation, post-capture AE, and photofinishing modules and saves structured intermediate analysis.

**Architecture:** Add a model-independent `FullPipelineAnalyzer` that executes the existing `PipeLine` twice: a capture/color/exposure pass without photofinishing, then a tone-rendering pass using the post-AE linear image through the `lsrgb` bypass. Add a CLI adapted from `main/demo.py` for input decoding, model construction, image export, and JSON/log output.

**Tech Stack:** Python 3.9, PyTorch 2.5.1, NumPy 1.26.4, repository image utilities, standard-library `unittest`.

## Global Constraints

- Preserve the repository's processing order: AWB/CCM before post-capture AE, then Gain → GTM → LTM → chroma → gamma.
- Do not modify trained model definitions or checkpoint keys.
- Do not duplicate AWB, AE, or TM algorithms.
- Keep `main/demo.py` behavior unchanged.
- Use test-first implementation for production Python modules.
- Fail explicitly on missing mandatory stage outputs or non-finite images.

---

### Task 1: Specify and test the analyzer contract

**Files:**
- Create: `tests/test_full_pipeline_analysis.py`

**Interfaces:**
- Consumes: future `FullPipelineAnalyzer`, `compute_luminance_stats`, and `to_jsonable` from `main/full_pipeline_analysis.py`.
- Produces: executable behavioral contract for the implementation.

- [ ] **Step 1: Write a fake pipeline and failing two-pass test**

Create a fake callable that records keyword arguments. On the first call it returns RAW, denoised RAW, pre-AE linear sRGB, post-AE linear sRGB, EV, illuminant, CCM, CCT, and tint. On the second call it asserts that `lsrgb` is exactly the first call's post-AE output and returns Gain/GTM/LTM/chroma/gamma intermediates and parameters.

- [ ] **Step 2: Add failing numerical helper tests**

Test a constant gray tensor so luminance mean, percentiles, clipping ratios, and dynamic-range values are known. Test recursive JSON conversion of tensors, NumPy arrays, NumPy scalars, lists, and dictionaries.

- [ ] **Step 3: Add failing validation test**

Return a first-pass dictionary without `ev` and assert that the analyzer raises `KeyError` naming the missing key.

- [ ] **Step 4: Run RED verification**

Run:

```bash
python -m unittest tests.test_full_pipeline_analysis -v
```

Expected: import failure because `main/full_pipeline_analysis.py` does not exist.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_full_pipeline_analysis.py
git commit -m "test: specify full pipeline analyzer behavior"
```

---

### Task 2: Implement the reusable analyzer

**Files:**
- Create: `main/full_pipeline_analysis.py`
- Test: `tests/test_full_pipeline_analysis.py`

**Interfaces:**
- Produces:
  - `compute_luminance_stats(image: Tensor | ndarray) -> dict[str, float]`
  - `to_jsonable(value: Any) -> Any`
  - `FullPipelineAnalyzer.run(raw, metadata, illum=None, ccm=None, **kwargs) -> dict`

- [ ] **Step 1: Implement image normalization helpers**

Accept BCHW, CHW, or HWC RGB data, detach tensors, convert to float64 NumPy, reject unsupported layouts and non-finite values, and compute Rec.709 luminance.

- [ ] **Step 2: Implement luminance statistics**

Return `mean`, `std`, `p01`, `p50`, `p99`, `low_clip_ratio`, `high_clip_ratio`, and `robust_dynamic_range_stops` using epsilon-safe logarithms.

- [ ] **Step 3: Implement recursive JSON conversion**

Convert tensors and arrays to Python scalars/lists, preserve dictionaries and sequences, and reject unsupported opaque objects with a clear `TypeError`.

- [ ] **Step 4: Implement pass-1 execution**

Call the injected pipeline with `photofinishing=False`, requested `auto_exposure`, `enhancement_strength=0.0`, `sharpening_amount=0.0`, `log_messages=True`, and `report_time=True`. Validate required keys and retain pre-AE and post-AE linear images.

- [ ] **Step 5: Implement pass-2 execution**

Call the same pipeline with pass-1 `denoised_raw`, post-AE `lsrgb`, pass-1 illuminant/CCM, `auto_exposure=False`, `photofinishing=True`, and `return_intermediate=True`. Validate required keys.

- [ ] **Step 6: Assemble report and stage dictionary**

Return ordered stage images, luminance statistics, estimated color/exposure values, photofinishing parameters, wall-clock timings, and missing optional stages.

- [ ] **Step 7: Run GREEN verification**

Run:

```bash
python -m unittest tests.test_full_pipeline_analysis -v
```

Expected: all analyzer tests pass.

- [ ] **Step 8: Commit implementation**

```bash
git add main/full_pipeline_analysis.py tests/test_full_pipeline_analysis.py
git commit -m "feat: add stage-wise full pipeline analyzer"
```

---

### Task 3: Add the full-pipeline analysis CLI

**Files:**
- Create: `main/analyze_full_pipeline.py`
- Modify: `main/README.md`

**Interfaces:**
- Consumes: existing `PipeLine`, `utils.img_utils`, `utils.file_utils`, and `FullPipelineAnalyzer`.
- Produces: command-line analysis entry point and documented output layout.

- [ ] **Step 1: Add CLI arguments**

Support the same essential input/model/device flags as `demo.py`, plus `--re-compute-awb`, `--pref-awb`, `--disable-auto-exposure`, `--post-process-ltm`, `--solver-iterations`, `--no-downscale-ps`, `--enhancement-model-path`, and optional editing controls.

- [ ] **Step 2: Implement input decoding**

Reuse the same DNG, PNG-16 + JSON, and JPEG/PNG linearization rules as `demo.py`. Convert the normalized RAW image to a BCHW float32 tensor on the selected device.

- [ ] **Step 3: Implement AWB selection**

When `--re-compute-awb` is absent, pass metadata illuminant and CCM. When present, pass `None` for both and load the repository AWB models from constants.

- [ ] **Step 4: Run analyzer and save stages**

Create `<output-dir>/<basename>-analysis`, save available linear intermediates as PNG-16, final output as JPEG, `analysis.json`, and the original pipeline log as `pipeline.log`.

- [ ] **Step 5: Update usage documentation**

Add one reproducible command and explain the two-pass analysis semantics and stage filenames in `main/README.md`.

- [ ] **Step 6: Run syntax verification**

Run:

```bash
python -m py_compile main/full_pipeline_analysis.py main/analyze_full_pipeline.py tests/test_full_pipeline_analysis.py
```

Expected: exit code 0.

- [ ] **Step 7: Commit CLI and docs**

```bash
git add main/analyze_full_pipeline.py main/README.md
git commit -m "feat: add AE AWB TM analysis CLI"
```

---

### Task 4: Verify requirements and create pull request

**Files:**
- Review all files changed on `feature/full-ae-awb-tm-analysis`.

**Interfaces:**
- Produces: verified branch and reviewable pull request.

- [ ] **Step 1: Run the complete model-free test suite**

```bash
python -m unittest tests.test_full_pipeline_analysis -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run syntax compilation**

```bash
python -m py_compile main/full_pipeline_analysis.py main/analyze_full_pipeline.py tests/test_full_pipeline_analysis.py
```

Expected: exit code 0.

- [ ] **Step 3: Inspect branch diff**

Confirm that no trained architecture, checkpoint loader, or existing demo behavior changed.

- [ ] **Step 4: Create a draft pull request**

Use title `Add stage-wise AE/AWB/TM analysis pipeline`. Include architecture, outputs, verification evidence, and the limitation that full model inference requires local model weights and dataset files.
