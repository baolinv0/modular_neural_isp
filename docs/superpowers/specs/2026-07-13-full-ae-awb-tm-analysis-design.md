# Full AE–AWB–TM Analysis Pipeline Design

## Goal

Add a reproducible, stage-observable inference path that follows the repository's real processing order:

```text
RAW decode/normalize
→ raw denoising
→ AWB illuminant estimation or metadata illuminant
→ CCM estimation or metadata CCM
→ RAW-to-linear-sRGB
→ post-capture AE estimation and EV application
→ Gain
→ Global Tone Mapping
→ Local Tone Mapping
→ chroma mapping
→ gamma/output encoding
→ optional detail enhancement and sharpening
```

The implementation must reuse the existing `PipeLine` and trained models. It must not duplicate or replace the repository's AWB, AE, or tone-mapping algorithms.

## Existing Code Findings

- `main/demo.py` performs input decoding, model construction, optional AWB re-estimation, optional automatic exposure, and final rendering through one `PipeLine.forward` call.
- `PipeLine._color_correction` estimates the illuminant when `illum=None`, computes or selects the CCM, and converts denoised RAW to linear sRGB.
- `PipeLine._auto_exposure` runs after color correction. It searches 49 EV candidates in linear sRGB and selects the candidate whose luminance histogram best matches a target histogram.
- `PhotofinishingModule` applies digital Gain, GTM, LTM, chroma mapping, and gamma in that order.
- `PipeLine.forward(return_intermediate=True)` already exposes the photofinishing images and predicted parameters, but it does not expose the post-AE/pre-TM linear image as a named intermediate.

## Chosen Architecture

Use a two-pass analyzer wrapper around the existing pipeline.

### Pass 1: capture/color/exposure

Call `PipeLine.forward` with:

- `photofinishing=False`
- `auto_exposure=True` unless explicitly disabled
- `enhancement_strength=0.0`
- `sharpening_amount=0.0`

This returns:

- input RAW
- denoised RAW
- pre-AE linear sRGB in `output['lsrgb']`
- estimated/given illuminant and CCM
- estimated CCT/tint
- estimated EV
- post-AE linear sRGB in `output['srgb']`

Because photofinishing and enhancement are disabled, `output['srgb']` is the post-AE linear image despite the legacy key name.

### Pass 2: tone rendering

Feed the post-AE linear image back to `PipeLine.forward` through its `lsrgb` bypass with:

- `auto_exposure=False`
- `photofinishing=True`
- `return_intermediate=True`

This prevents AWB and AE from being recomputed and returns:

- Gain/GTM/LTM/chroma/gamma intermediates
- Gain, GTM, LTM, LUT, and gamma parameters
- final output

The second pass receives the first pass's illuminant, CCM, metadata, and denoised RAW so existing assertions and metadata-dependent behavior remain valid.

## New Files

### `main/full_pipeline_analysis.py`

Reusable analysis layer containing:

- `compute_luminance_stats(image)`
- `to_jsonable(value)`
- `FullPipelineAnalyzer`
- stage collection and report generation
- explicit validation of expected pipeline outputs

The module depends only on NumPy and PyTorch. It accepts any pipeline-like object with the existing `PipeLine.forward` contract, which allows model-free unit testing.

### `main/analyze_full_pipeline.py`

Console entry point adapted from `main/demo.py` that:

- reads DNG, PNG-16 + JSON metadata, or JPEG/PNG inputs supported by the repository
- constructs `PipeLine`
- selects metadata AWB or neural AWB re-estimation
- runs `FullPipelineAnalyzer`
- saves every available stage image
- saves `analysis.json` and `pipeline.log`

### `tests/test_full_pipeline_analysis.py`

Model-free tests using a fake pipeline to verify:

- exact two-pass call order
- AE is enabled only in pass 1
- photofinishing is enabled only in pass 2
- post-AE output is passed unchanged into TM
- expected stages are collected
- tensor/array report values are JSON serializable
- luminance statistics are numerically correct
- missing mandatory outputs fail fast

## Output Layout

For input `scene.dng`:

```text
<output-dir>/scene-analysis/
  00-raw.png
  01-denoised-raw.png
  02-linear-awb.png
  03-linear-ae.png
  04-gain.png
  05-gtm.png
  06-ltm.png
  07-chroma.png
  08-gamma.png
  09-final.jpg
  analysis.json
  pipeline.log
```

Unavailable optional stages are omitted and listed in `analysis.json`.

## Analysis Report

`analysis.json` contains:

- input path and configuration
- processing order
- luminance statistics for each stage: mean, standard deviation, p01, p50, p99, low/high clipping ratios, and robust dynamic range in stops
- estimated EV
- illuminant, CCM, CCT, and tint
- Gain/GTM/LTM/LUT/gamma parameters
- total wall time for the capture/color/exposure pass and the tone-rendering pass
- missing optional stage names

Detailed module timing printed by the original pipeline is retained in `pipeline.log` by running with `log_messages=True` and `report_time=True`.

## Error Handling

Fail with explicit errors for:

- unsupported input extension
- PNG-16 without matching metadata JSON
- missing RAW or metadata after JPEG extraction/linearization
- absent mandatory pass-1 keys: `lsrgb`, `srgb`, `ev`, `illum`, `ccm`
- absent mandatory pass-2 keys: `srgb`, `gain_param`, `gtm_param`, `ltm_param`
- non-finite image values
- malformed image layout

## Compatibility

- No existing model weights or state-dict keys change.
- `main/demo.py` behavior remains unchanged.
- The analyzer works with the current full photofinishing model and provides a clean integration point for the later luminance-only controllable-TM variants.
- The repository's CC BY-NC license remains applicable.

## Non-goals

- Retraining AE or AWB.
- Changing the AE target histogram or candidate search.
- Changing AWB model selection.
- Reordering AE ahead of AWB; the implementation preserves the existing repository order, AWB/CCM first and post-capture AE second.
- Implementing the four brightness-control training variants in this change.
