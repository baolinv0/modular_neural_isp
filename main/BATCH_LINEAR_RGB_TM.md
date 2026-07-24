# Batch Tone Mapping for 16-bit Linear RGB

`batch_linear_rgb_tm.py` processes a directory of images that have already completed:

```text
Demosaic / reconstruction
→ White balance (AWB)
→ Camera color correction (CCM)
→ Linear sRGB
```

The script starts at the repository's `PhotofinishingModule` and therefore skips RAW denoising, AWB estimation, CCM computation, and RAW-to-linear-sRGB conversion.

## Input contract

Each input must be:

- PNG, TIFF, or TIF;
- unsigned 16-bit (`uint16`);
- exactly three channels;
- RGB values stored linearly in the range `[0, 65535]`;
- already transformed into linear sRGB by AWB and CCM.

The loader uses OpenCV internally but converts its BGR read order back to RGB before inference. Non-`uint16` and non-three-channel files are reported as failures rather than silently converted.

## Processing path

```text
uint16 linear sRGB
→ normalize to float32 [0, 1]
→ PhotofinishingModule
   → digital gain
   → global tone mapping
   → local tone mapping
   → chroma mapping
   → gamma
→ save TM result
→ NAFNet detail enhancement (default / required)
→ save enhanced result
```

By default, photofinishing runs at one-quarter resolution and uses the same Bilateral Guided Upsampling implementation as the main pipeline. Use `--no-downsampling` to run the photofinishing module at full input resolution.

## Basic command

```bash
python main/batch_linear_rgb_tm.py \
    --input-dir /path/to/linear_rgb16 \
    --output-dir /path/to/results \
    --photofinishing-model-path photofinishing/models/model.pth \
    --enhancement-model-path enhancement/models/model.pth
```

The enhancement model is required because every input is processed through both TM and detail enhancement.

The model JSON config is discovered automatically using the repository conventions:

```text
model-directory/model.json
model-parent/config/model.json
model-parent/configs/model.json
```

Explicit configs can be supplied with `--photofinishing-config-path` and `--enhancement-config-path`.

## Recursive scan and LTM refinement

```bash
python main/batch_linear_rgb_tm.py \
    --input-dir /path/to/linear_rgb16 \
    --output-dir /path/to/results \
    --photofinishing-model-path photofinishing/models/model.pth \
    --enhancement-model-path enhancement/models/model.pth \
    --recursive \
    --post-process-ltm \
    --solver-iterations 50
```

## Output naming

`--output-format` defaults to `both`. For an input such as:

```text
scene_01.png
```

the script saves all four outputs by default:

```text
scene_01-tm.png
scene_01-tm.jpg
scene_01-tm-enhanced.png
scene_01-tm-enhanced.jpg
```

The variants mean:

- `-tm`: photofinishing/TM output before detail enhancement;
- `-tm-enhanced`: TM output after the NAFNet detail-enhancement model.

The output format can still be restricted explicitly:

```bash
--output-format png16
--output-format jpeg
--output-format both
```

When `--skip-existing` is enabled, a sample is skipped only if every output required by the selected format already exists. With the default `both` mode, all four files must exist.

The output directory also contains `batch_results.csv` with:

- input path;
- generated output paths;
- status: `ok`, `failed`, or `skipped`;
- elapsed time;
- exception message for failed samples.

The script returns a non-zero process exit code when at least one image fails.

## Run unit tests

```bash
python -m unittest tests/test_batch_linear_rgb.py
```
