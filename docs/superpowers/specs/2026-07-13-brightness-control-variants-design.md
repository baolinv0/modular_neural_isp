# Brightness-Control Variants Design

## Goal

Extend the luminance-only photofinishing baseline with four interchangeable brightness-control mechanisms while preserving the original baseline at alpha = 0 and restricting control to GainNet and GlobalToneMappingNet.

## Baseline Assumptions

- Input images are 16-bit PNG linear RGB files.
- White balance and CCM have already been applied.
- Input loading converts uint16 RGB directly to float32 in [0, 1] by dividing by 65535.
- The baseline checkpoint is already trained for a luminance-only pipeline.
- The baseline pipeline keeps GainNet, GlobalToneMappingNet, and LocalToneMappingNet.
- LocalToneMappingNet is loaded from the baseline checkpoint and remains fully frozen.
- Learned color modules are absent from the experiment path.
- Learned GammaNet is absent from the experiment path.
- Output encoding uses a fixed standard sRGB OETF:

  - 12.92 * x for x <= 0.0031308
  - 1.055 * x^(1/2.4) - 0.055 otherwise

## Data Layout

```text
input_linear/
  scene_0001.png
  scene_0002.png
  ...

gt_levels/
  a_m100/
  a_m075/
  a_m050/
  a_m025/
  a_000/
  a_p025/
  a_p050/
  a_p075/
  a_p100/
```

Files are matched strictly by filename. Missing files are fatal errors.

Level mapping:

| Folder | Alpha |
|---|---:|
| a_m100 | -1.00 |
| a_m075 | -0.75 |
| a_m050 | -0.50 |
| a_m025 | -0.25 |
| a_000 | 0.00 |
| a_p025 | 0.25 |
| a_p050 | 0.50 |
| a_p075 | 0.75 |
| a_p100 | 1.00 |

## Sampling Strategy

Training uses same-scene ordered level pairs.

For nine levels there are 36 unordered pairs. Each epoch expands all scene-pair combinations, then shuffles scene and pair order with a deterministic seed. Across the 36 pairs, each level appears exactly eight times, so the marginal level probability is identical for all nine levels.

Each sample returns:

```python
{
    "in_image": Tensor[3, H, W],
    "gt_low": Tensor[3, H, W],
    "gt_high": Tensor[3, H, W],
    "alpha_low": float,
    "alpha_high": float,
    "scene_name": str,
}
```

The invariant is alpha_low < alpha_high.

## Frozen Baseline Policy

All original baseline parameters are frozen after loading the shared checkpoint.

Trainable parameters are limited to the selected control mechanism inside GainNet and GlobalToneMappingNet.

At alpha = 0, each control mechanism must reduce exactly to the baseline path. This is enforced structurally, not only by loss.

LocalToneMappingNet remains active in the forward path but fully frozen.

## Control Methods

The implementation exposes one CLI switch:

```text
--control-method {param_residual,parallel_adapter,film,dual_lora}
```

All four methods share the same baseline checkpoint, dataset index, pair order, random seed, optimizer schedule, losses, evaluation code, and parameter-budget target.

### 1. param_residual

- Reuse frozen GainNet and GTM feature extractors.
- Add separate positive and negative residual heads for the predicted gain logit and GTM pre-Softplus parameters.
- Apply alpha_pos = max(alpha, 0) and alpha_neg = max(-alpha, 0).
- Final parameter logits equal baseline logits plus scaled positive or negative residuals.
- Initialize final residual projections to zero.

### 2. parallel_adapter

- Add compact bottleneck adapters in parallel to the pooled GainNet and GTM features.
- Adapter output is added to the baseline feature before the frozen prediction head.
- Use separate positive and negative adapters.
- Scale adapter outputs by alpha_pos or alpha_neg.
- Initialize adapter output projections to zero.

### 3. film

- Generate feature-wise scale and bias from alpha_pos and alpha_neg for GainNet and GTM pooled features.
- Apply feature = feature * (1 + scale) + bias.
- Ensure scale and bias are exactly zero at alpha = 0.
- Initialize FiLM output projections to zero.

### 4. dual_lora

- Keep the original GainNet and GTM layers frozen.
- Add separate positive and negative low-rank updates to selected GainNet and GTM linear prediction layers.
- Weight updates are scaled by alpha_pos or alpha_neg.
- LoRA B matrices are zero-initialized.

## Parameter-Budget Matching

The four methods must have approximately matched trainable parameter counts, with a target difference within plus or minus 10 percent.

A utility reports:

- total parameters,
- frozen parameters,
- trainable control parameters,
- per-module trainable parameters.

If a requested configuration exceeds the tolerance, training exits with a clear error unless explicitly overridden.

## Forward Interface

The controlled luminance-only model exposes:

```python
def forward(
    self,
    x: torch.Tensor,
    alpha: torch.Tensor,
    training_mode: bool = False,
) -> Dict[str, torch.Tensor]:
    ...
```

Alpha shape is either [B] or [B, 1] and is normalized internally to [B, 1].

Returned values include at least:

```python
{
    "output": encoded_srgb,
    "linear_output": linear_rgb,
    "gain_factor": gain_factor,
    "gtm_params": gtm_params,
    "ltm_params": ltm_params,
}
```

## Brightness-Only Loss

The unified training objective is:

```text
L = L_logY + lambda_grad * L_grad + lambda_mono * L_mono + lambda_zero * L_zero
```

### Log-luminance reconstruction

Compute luminance from predicted and target sRGB images after converting them to linear RGB with the inverse standard sRGB transfer function. Use Charbonnier or L1 loss in log-luminance space.

### Gradient loss

Compute first-order horizontal and vertical gradients on linear luminance and compare prediction with the corresponding target.

### Monotonic loss

For each same-scene pair alpha_low < alpha_high:

```text
mean_log_luminance(pred_high) >= mean_log_luminance(pred_low) + margin
```

Use a hinge penalty. The margin scales with alpha_high - alpha_low and is configurable.

### Zero-anchor loss

For alpha = 0, compare the controlled model output against the frozen baseline output. The expected error should be at numerical-noise level because every method is structurally zero at alpha = 0.

No VGG, DeltaE, CbCr, LUT, saturation, or learned-gamma losses are used.

## Training Protocol

- Load one shared luminance-only baseline checkpoint.
- Freeze all baseline parameters.
- Enable only one control method per run.
- Train two outputs per sample: low level and high level.
- Use identical seeds and pair ordering across methods.
- Use identical optimizer, scheduler, batch size, image size, and number of optimizer steps.
- Save method-specific checkpoints and logs.

## Evaluation

Report per method:

- log-luminance MAE per level,
- PSNR and SSIM on luminance,
- dense alpha-sweep monotonic violation rate,
- alpha = 0 baseline drift,
- endpoint clipping ratio,
- endpoint deep-shadow ratio,
- gain-factor range,
- GTM-parameter range,
- trainable parameter count,
- inference latency.

Validation and test splits are scene-disjoint. All nine levels from a scene stay in the same split.

## Error Handling

Training must fail fast for:

- missing input or GT files,
- filename mismatches,
- duplicate filenames,
- non-16-bit input PNGs,
- inconsistent spatial dimensions,
- alpha outside [-1, 1],
- checkpoint key mismatches,
- trainable baseline parameters after freezing,
- parameter-budget mismatch beyond tolerance.

## CLI

A new training entry point will support at least:

```bash
python photofinishing/train_brightness_control.py \
  --input-dir /path/to/input_linear \
  --gt-root /path/to/gt_levels \
  --baseline-checkpoint /path/to/luminance_baseline.pth \
  --control-method param_residual \
  --batch-size 18 \
  --seed 42
```

Equivalent commands select `parallel_adapter`, `film`, or `dual_lora`.

## Tests

Tests cover:

- direct uint16 PNG loading and normalization,
- filename-based nine-level matching,
- equal marginal level counts over all pairs,
- same-scene ordered pair construction,
- alpha = 0 exact baseline equivalence for all four methods,
- only Gain/GTM control parameters are trainable,
- LTM remains frozen and active,
- fixed sRGB OETF numerical correctness,
- parameter-budget reporting and tolerance enforcement,
- monotonic-loss direction and margin behavior,
- checkpoint load failure on incompatible keys,
- one training-step smoke test for every control method.

## Non-goals

- No LTM control adapter in this phase.
- No color control.
- No learned gamma.
- No full-network fine-tuning.
- No continuous online Retinex target generation.
- No dynamic convolution or hypernetwork variants.
