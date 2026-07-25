# Unaligned Reference-Style Fine-Tuning Plan

## Goal

Validate whether a same-scene but non-pixel-aligned A-camera reference can supervise the existing photofinishing module through distributional losses before generating pixel pseudo-GT.

## Data

Each JSONL record contains:

- `input_path`: B-domain input to the existing photofinishing model (`Bin`);
- `reference_path`: same-scene A-camera product output (`AGT`), not pixel aligned;
- `sample_id`, `split`, and optional confidence `weight`.

The reference is used only through global distribution statistics. No pixel reconstruction, optical-flow warp, or aligned crop loss is used against `AGT`.

## Stage 1: Brightness and global tone

Trainable:

- `_gain_net`;
- `_gtm_net`.

Frozen:

- `_ltm_net`;
- `_lut_net`;
- `_gamma_net`;
- optional `_3d_lut`.

Loss:

- sorted log-luminance distribution;
- luminance mean and standard deviation;
- soft shadow occupancy;
- soft highlight occupancy.

## Stage 2: Chroma

Start from the best Stage-1 checkpoint.

Trainable:

- `_lut_net` only.

Frozen:

- Gain;
- GTM;
- LTM;
- Gamma;
- optional 3D LUT.

Loss:

- Cb/Cr marginal distributions;
- chroma moments and covariance;
- chroma-magnitude distribution;
- same-input luminance anchor to the frozen Stage-1 model.

## Verification

- manifest and image-format validation;
- non-pixel loss invariance to spatial permutation;
- stage-specific `requires_grad` checks;
- gradient leakage checks;
- base/tone/final evaluation report on a scene-disjoint split.

## Evidence boundary

This experiment can establish whether non-aligned statistical supervision moves the model toward the A-camera style. It does not yet establish pixel pseudo-GT validity, user-preference improvement, or safe integration into the cross-camera adaptation loop.