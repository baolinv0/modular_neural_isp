# Full AE–AWB–TM Analysis Pipeline

`analyze_full_pipeline.py` runs the repository's existing imaging modules while exposing the intermediate state of the complete chain:

```text
RAW
  -> raw denoising
  -> AWB illuminant estimation or metadata AWB
  -> CCM estimation or metadata CCM
  -> linear sRGB
  -> post-capture auto exposure
  -> digital gain
  -> global tone mapping
  -> local tone mapping
  -> chroma mapping
  -> gamma
  -> optional enhancement and sharpening
  -> final sRGB
```

The analysis wrapper does not reimplement AE, AWB, CCM, or tone mapping. It calls the existing `PipeLine` twice so that the post-AE/pre-TM linear image becomes an explicit boundary:

1. Capture/color/exposure pass with photofinishing disabled.
2. Tone-rendering pass using the first pass's post-AE linear image through the `lsrgb` bypass.

This avoids applying AWB, CCM, AE, or RAW denoising twice.

## Example

Run from the repository root:

```bash
python main/analyze_full_pipeline.py \
  --input-file /data/scene.dng \
  --denoising-model-path denoising/models/generic_base.pth \
  --photofinishing-model-path photofinishing/models/photofinishing_s24-style-0.pth \
  --re-compute-awb \
  --pref-awb \
  --post-process-ltm \
  --output-dir results
```

To use the illuminant and CCM stored in metadata, omit `--re-compute-awb` and `--pref-awb`.

For a normalized 16-bit PNG input, provide a matching JSON file using one of these layouts:

```text
images/scene.png
images/scene.json
```

```text
images/scene.png
images/data/scene.json
```

```text
images/scene.png
data/scene.json
```

An explicit metadata path can also be passed with `--metadata-json`.

## Output

The command creates `<output-dir>/<input-stem>-analysis/`:

```text
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

The PNG intermediates are saved as 16-bit images. `analysis.json` contains:

- selected configuration and device;
- estimated EV;
- illuminant, CCM, CCT, and tint;
- Gain/GTM/Gamma values;
- compact shape/range/statistical summaries for high-dimensional LTM and chroma-LUT parameters;
- per-stage luminance mean, standard deviation, percentiles, clipping ratios, and robust dynamic range;
- first-pass, second-pass, and total wall-clock time;
- the output file manifest.

`pipeline.log` contains the original module-level messages emitted by `PipeLine`.

## Important flags

- `--re-compute-awb`: ignore metadata illuminant and run the configured AWB model.
- `--pref-awb`: apply the preference mapping after AWB estimation; requires `--re-compute-awb`.
- `--use-cc-awb`: force cross-camera AWB; requires `--re-compute-awb`.
- `--disable-auto-exposure`: preserve the AWB/CCM-corrected linear exposure.
- `--ev-value`: apply a manual pre-AE EV offset.
- `--no-downscale-ps`: run photofinishing without the default quarter-resolution parameter-estimation path.
- `--post-process-ltm`: enable LTM coefficient refinement.
- `--solver-iterations`: set the LTM bilateral-solver iterations.

## Verification

Model-independent tests cover:

- two-pass ordering and exact post-AE handoff;
- no duplicate luma/chroma denoising;
- required-stage validation;
- luminance statistics and non-finite rejection;
- compact parameter summaries;
- metadata AWB/CCM selection;
- metadata JSON discovery;
- stage export, JSON report, and pipeline log creation;
- repository-relative built-in model path resolution.

A real end-to-end run additionally requires the checkpoint files referenced by the command and an input image with valid metadata.