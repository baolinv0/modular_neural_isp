# Multi-Model Pseudo-GT Single-Scene Probe v1.1 — CONTRACT

## 1. Status

- **Purpose**: single-scene feasibility probe
- **Scope**: image generation/editing, deterministic tone projection, baseline comparison, metrics, blind review
- **Out of scope**: TM model training, real-data qualification, production readiness, cross-scene generalization
- **Final decision vocabulary**: `GO`, `PROBE`, `KILL_OR_PIVOT`
- **No self-upgrade rule**: passing this probe does not imply pseudo-GT training effectiveness

## 2. Primary Question

For one image with a clear Tone Mapping failure, can heterogeneous image-editing models produce a useful low-frequency tonal target that:

1. improves the specified Tone Mapping failure;
2. preserves the target image content;
3. does not depend primarily on color, texture synthesis, sharpening, denoising, relighting, or image regeneration;
4. remains useful after deterministic Tone-only projection back to the target image;
5. outperforms simple parametric tone adjustment;
6. benefits from a same-scene, non-pixel-aligned reference image when native multi-image editing is supported?

## 3. Frozen Input Contract

### 3.1 Target Image `X`

The target image must have one dominant failure:

> Highlights are already safe, while the main subject or person is too dark in the midtones and has insufficient tonal separation from the background.

Required properties:

- subject, face, and background structure are visible;
- no severe motion blur;
- no severe highlight clipping;
- no dominant AWB failure;
- no geometric corruption;
- improvement does not require super-resolution or major restoration;
- expected correction is primarily within Tone Mapping capability.

### 3.2 Same-Scene Reference Image `R`

The reference image must:

- depict the same scene and principal content;
- be semantically corresponding but not necessarily pixel aligned;
- allow small viewpoint, crop, pose, motion, or occlusion differences;
- have substantially similar physical illumination;
- be visibly better specifically on the target Tone Mapping failure.

The reference image is a **distributional and semantic reference**, not a pixel-level GT.

### 3.3 Required Masks

Minimum required:

- `target_subject_mask.png`
- `target_background_mask.png`
- `reference_subject_mask.png`
- `reference_background_mask.png`

Optional:

- face masks;
- highlight masks;
- sky/window masks.

Manual coarse masks are acceptable for this probe.

## 4. Frozen Model Set

### 4.1 Core Models

| ID | Model identifier | Required modes |
|---|---|---|
| `gpt_image_2` | `gpt-image-2-2026-04-21` | no-reference; native reference if supported by verified API |
| `gemini_image` | `gemini-3.1-flash-image` | no-reference; native reference if supported by verified API |
| `flux_2_pro` | `flux-2-pro` | no-reference; native reference if supported by verified API |
| `qwen_image_edit` | `Qwen/Qwen-Image-Edit-2511` | no-reference; reference mode only after capability verification |

### 4.2 Boogu Extension

| ID | Model identifier | Required modes |
|---|---|---|
| `boogu_image_edit` | `Boogu/Boogu-Image-0.1-Edit` | no-reference only |
| `boogu_image_edit_turbo` | `Boogu/Boogu-Image-0.1-Edit-Turbo` | no-reference only |

Boogu must not be treated as supporting native `X + R` editing unless the exact checked-out revision and inference interface demonstrably accept two ordered images.

### 4.3 Capability Probe Rule

Before generation, every adapter must record:

- exact model/revision;
- whether single-image edit works;
- whether ordered multi-image input works;
- whether aspect ratio is preserved;
- whether seed is controllable;
- whether prompt rewriting or prompt upsampling occurs;
- request ID or local run ID.

Unsupported modes must return a structured status, not a fabricated result:

```text
REFERENCE_MODE_UNSUPPORTED
```

Do not emulate multi-image support by concatenating images into a collage.

## 5. Frozen Experiment Matrix

### 5.1 Core Models

For every core model:

- `M0_NO_REFERENCE`: `X + frozen no-reference prompt`
- `M1_SAME_SCENE_REFERENCE`: `X + R + frozen reference prompt`, only if native support is verified

One generation per supported mode. No best-of-N selection.

### 5.2 Boogu

- `boogu_image_edit`: one no-reference generation
- `boogu_image_edit_turbo`: one no-reference generation

No native visual-reference experiment for Boogu in v1.1.

### 5.3 Deterministic Outputs

For every successful Raw output:

- save Raw image;
- compute one deterministic Tone-only projection;
- do not manually retouch;
- do not tune projection per model.

### 5.4 Traditional Baselines

Required:

- `B1_SIMPLE_TONE`: subject-aware midtone lift without a reference image
- `B2_REFERENCE_STAT_TONE`: semantic/statistical transfer from `R`, without pixel-level matching

## 6. Frozen Prompt Contract

### 6.1 No-Reference Prompt

```text
Edit Image 1 only as a tone-mapping correction.

The main subject is too dark in the midtones while the highlights are already safe.

Improve the subject midtones and the tonal separation between the subject and background. Preserve the original scene content, identity, geometry, composition, color appearance, texture, highlight detail, black point, and light-source structure.

Do not add or remove objects. Do not beautify, sharpen, denoise, relight, recolor, restyle, reconstruct, or regenerate the scene.

The output must look like the same camera image processed with a better tone-mapping curve, not a newly generated photograph.
```

### 6.2 Same-Scene Reference Prompt

```text
Image 1 is the target image to edit.

Image 2 is a non-pixel-aligned reference photograph of the same scene. Use Image 2 only as a reference for the desired tonal distribution, including subject brightness, subject-to-background luminance relationship, shadow visibility, midtone separation, and highlight roll-off.

Do not copy the geometry, viewpoint, pose, texture, object boundaries, local details, identity, or composition from Image 2.

Apply the inferred tone-mapping characteristics to Image 1 while preserving all content, identity, geometry, composition, color appearance, texture, and light-source structure of Image 1.

Do not add or remove objects. Do not beautify, sharpen, denoise, reconstruct, relight, recolor, restyle, or regenerate the scene.

The output must remain Image 1, processed with a tone distribution guided by Image 2.
```

### 6.3 Boogu Prompt

```text
对输入照片仅执行 Tone Mapping 修正。

当前主要人物或主体的中间调偏暗，但高光已经安全。适度提高主体的中间调可见度，改善主体与背景之间的亮度层次，同时保护高光细节、黑位和原有场景对比度。

必须完整保持原图中的人物身份、五官、姿态、物体、几何结构、构图、颜色、肤色、纹理、光源位置和阴影结构。

不得增加或删除物体，不得重新布光，不得改变背景，不得美颜、锐化、去噪、超分、重建纹理、改变景深、风格化或重新生成图像。

结果必须看起来是同一张相机照片经过更合理的 Tone Mapping 处理，而不是一张重新生成的照片。
```

Do not mix prompt-language ablation into this probe.

## 7. Generation Rules

- one generation per model/mode;
- no manual candidate selection;
- no hidden retry that overwrites a failed output;
- every failure and rejection must be logged;
- preserve aspect ratio where the API permits;
- use approximately 1K output for fair initial comparison;
- never place credentials in logs;
- save the exact prompt actually sent;
- save provider-side prompt rewrite metadata if observable.

Recommended deterministic settings:

```yaml
qwen_image_edit:
  seed: 42

flux_2_pro:
  prompt_upsampling: false

boogu_image_edit:
  resolution: 1K
  num_inference_steps: 50
  text_guidance_scale: 5.0
  seed: 42

boogu_image_edit_turbo:
  resolution: 1K
  num_inference_steps: 4
  text_guidance_scale: 1.0
  seed: 42
```

The Boogu Turbo revision must be explicitly recorded. If the named revision cannot be verified, mark it `UNVERIFIED_REVISION` and do not silently substitute another checkpoint.

## 8. Tone-Only Projection Contract

For target `X` and Raw output `G`:

1. decode to sRGB and convert to linear RGB;
2. resize `G` to the target canvas;
3. allow only conservative global alignment;
4. do not use aggressive dense warping to hide regeneration;
5. compute luminance: `Y = 0.2126R + 0.7152G + 0.0722B`;
6. operate in log2 luminance;
7. apply the same Gaussian low-pass operator to `X` and `G`;
8. use:

```yaml
gaussian_sigma_ratio: 0.02
alpha: 0.5
max_delta_ev: 1.0
max_new_clipping_ratio: 0.005
```

9. compute `Delta_low = LP(log2(Y_G)) - LP(log2(Y_X))`;
10. clamp `Delta_low` to `[-1, +1] EV`;
11. construct `Y_tone = Y_X * 2^(0.5 * Delta_low)`;
12. preserve target chromaticity;
13. save clipping and alignment diagnostics.

Projection parameters are global and immutable for this experiment.

## 9. Metric Contract

### 9.1 Global Tone Features

Required:

- p10, p25, p50, p75, p90, p95, p99;
- shadow occupancy;
- highlight occupancy;
- p99–p10 dynamic range.

### 9.2 Semantic Statistics

Required:

- subject mean/median luminance;
- background mean/median luminance;
- subject/background luminance ratio;
- optional face/global ratio;
- subject and background highlight occupancy.

### 9.3 Content and Non-Target Changes

Must inspect:

- face identity;
- pose and silhouette;
- object additions/removals;
- text corruption;
- background geometry;
- light-source placement;
- shadow structure;
- depth-of-field changes;
- color/skin-tone drift;
- sharpening, denoising, texture synthesis, or hallucination.

### 9.4 Reference-Mode Metrics

Do not use `X` versus `R` pixel-level PSNR, SSIM, LPIPS, L1, or L2 as GT metrics.

Reference utilization must be measured through semantic-region and distributional features.

## 10. Human Review Contract

Review must be anonymous.

Scores are 0–2:

- target failure improvement;
- content preservation;
- non-target stability;
- Tone extractability;
- reference utilization, for reference mode;
- reference-copy protection, for reference mode.

Recommended: at least three independent reviewers.

If fewer than three:

```text
HUMAN_REVIEW_CONFIDENCE = LOW
```

## 11. Hard Gates

### 11.1 Universal Gate

All must pass:

```text
target_failure_improvement >= 1
content_preservation == 2
non_target_stability >= 1
tone_extractability >= 1
```

### 11.2 Reference Gate

Additionally:

```text
reference_utilization >= 1
reference_copy_protection == 2
```

### 11.3 Immediate Rejection

Reject on any of:

- identity change;
- object addition/removal;
- major pose or geometry change;
- copied reference-only content;
- changed light-source structure;
- full-scene regeneration;
- target benefit disappears after Tone-only projection;
- improvement primarily comes from sharpening, denoising, texture synthesis, or beautification.

No weighted total may override a failed hard gate.

## 12. Decision Contract

### GO

All required:

1. at least two heterogeneous model families pass all applicable gates;
2. at least two native reference-mode results are better than their no-reference counterparts;
3. reference results do not copy reference content;
4. Tone-only projection preserves most of the target improvement;
5. at least one model beats the matching traditional baseline;
6. model directions are broadly consistent on subject lift, background protection, and highlight safety.

### PROBE

Any of:

- only one model passes;
- Raw is useful but Tone-only loses substantial benefit;
- reference mode is promising but unstable;
- model beats `B1` but not `B2`;
- a single stochastic sample may explain the result.

Allowed follow-up only:

```text
best model + one representative failed model
× supported modes
× 3 seeds
```

### KILL_OR_PIVOT

Any of:

- all models regenerate or corrupt content;
- all benefit disappears after Tone-only projection;
- traditional baselines match or beat all models;
- reference mode mainly copies reference content;
- model directions are mutually contradictory.

If Raw improves but Tone-only fails, pivot to an explicitly separate `Tone + Color` or `Tone + Color + Texture` hypothesis.

## 13. Boogu-Specific Decision

### BOOGU_GO

Required:

- Boogu Edit passes every universal gate;
- Tone-only retains the main improvement;
- it clearly beats `B1_SIMPLE_TONE`;
- it is not worse than at least one core model in no-reference mode.

### BOOGU_PROBE

Examples:

- Raw looks better but Tone-only weakens;
- Edit and Turbo diverge materially;
- content preservation is slightly unstable;
- result is close to `B1`.

Allowed next step:

```text
Boogu Edit + Boogu Edit Turbo
× same target image
× 3 seeds
```

### BOOGU_KILL

Any of:

- identity or geometry changes;
- clear regeneration;
- Tone-only benefit disappears;
- no advantage over `B1`;
- apparent improvement is mainly texture or restoration.

## 14. Completion Boundary

The probe is complete only when:

- all supported generations or structured failures are present;
- every Raw output has a deterministic Tone-only result;
- both baselines are generated;
- metrics and blind review are complete;
- all hard gates are explicitly decided;
- core and Boogu decisions are recorded;
- the report does not claim TM training effectiveness or generalization.
