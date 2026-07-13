# adjustTM: Brightness-only controllable tone mapping

`adjustTM` compares four ways of adding continuous brightness control to one shared, pretrained luminance-only Modular Neural ISP baseline:

- `param_residual`
- `parallel_adapter`
- `film`
- `dual_lora`

Only GainNet and GlobalToneMappingNet receive trainable control parameters. The original Gain/GTM weights and the entire LTM remain frozen. Color LUTs and learned gamma are not used. The output is encoded with the fixed standard sRGB OETF.

## Data layout

```text
input_linear/
  scene_0001.png       # 16-bit linear RGB, WB + CCM already applied

gt_levels/
  a_m100/scene_0001.png
  a_m075/scene_0001.png
  a_m050/scene_0001.png
  a_m025/scene_0001.png
  a_000/scene_0001.png
  a_p025/scene_0001.png
  a_p050/scene_0001.png
  a_p075/scene_0001.png
  a_p100/scene_0001.png
```

All ten files for a scene must have the same filename and dimensions. Missing or extra GT files are fatal.

## Fair sampling

Each scene is expanded into all `C(9,2)=36` ordered level pairs. Every level appears in exactly eight pairs per scene, so all nine levels have identical marginal frequency in every epoch.

## Train

```bash
python -m adjustTM.train \
  --input-dir /data/input_linear \
  --gt-root /data/gt_levels \
  --baseline-checkpoint /models/luminance_only_baseline.pth \
  --control-method param_residual \
  --output-dir adjustTM/checkpoints \
  --batch-size 18 \
  --epochs 30 \
  --seed 42
```

Run the same command with `parallel_adapter`, `film`, and `dual_lora`. Default control dimensions produce approximately matched trainable parameter counts:

| Method | Default trainable control parameters |
|---|---:|
| param_residual | 1064 |
| parallel_adapter | 1096 |
| film | 976 |
| dual_lora | 1024 |

The training command checks the count against the default target of 1040 with ±10% tolerance.

## Evaluate

```bash
python -m adjustTM.evaluate \
  --input-dir /data/test/input_linear \
  --gt-root /data/test/gt_levels \
  --baseline-checkpoint /models/luminance_only_baseline.pth \
  --control-checkpoint adjustTM/checkpoints/film/control_epoch_030.pth \
  --control-method film \
  --output adjustTM/results/film.json
```

Reported metrics include per-level log-luminance MAE and luminance PSNR, dense-sweep monotonic violations, endpoint clipping/deep-shadow ratios, alpha-zero baseline drift, parameter count, and evaluation time.

## Checkpoint compatibility

The baseline loader accepts:

- a plain baseline `state_dict`;
- a dictionary under `state_dict`, `model_state_dict`, or `model`;
- optional `module.` or `baseline.` prefixes;
- full photofinishing checkpoints containing removed `_gamma_net`, `_lut_net`, or `_3d_lut` keys.

Gain, GTM, and LTM keys must match exactly; incompatible checkpoints fail fast.

## Tests

```bash
pytest adjustTM/tests -q
```
