# Preview clipping risk diagnostic, 2026-10-09

Frozen before training. Baseline commit: ae7c762bf18461fb1fb673b81b0ab0bc33c3e78e.
This task is authorized by the user's explicit failure-analysis/retest request;
it does not extend the expired AutoDL budget or open the real S24 formal test.

Hypothesis: an appearance-only teacher and bounded .02 reliability penalty can
prefer destructive captures. A causal preview risk constraint may prevent this,
but does not necessarily improve final quality or cure constant-action policies.

Use the existing Bayer generator: 40 scenes, 32x32, data seed2026, noise0/1,
24 train and8 validation scenes. Evaluate validation only. These analytic scene
families and validation data are development diagnostics, not a fresh holdout.
The generated8 test scenes are excluded from learning, audit and model selection.
All original plans, target, noise, sensor, fusion, WB/CCM, masks, TM capacity,
learning rates, per-module update counts and original J remain unchanged.

Four variants:

1. Original eight groups, warmup10/main10, seeds0/1/2.
2. Guarded eight groups, identical settings; tolerance .01, fixed ADC threshold .98.
3. Original learned groups evaluated with guard only at inference (no updates).
4. Guarded eight groups without frozen-TM warmup; same main10 budget.
   This explicitly has fewer AE updates; it isolates warmup, not equal total compute.

Guard: predict linear sample values from the latest effective sensor preview and
each candidate's effective EV. Risk is the fraction of RGB channel/pixels predicted
to clip in ALL frames. Permit risk <= rule-plan risk+.01. The rule is always feasible.
This is a deployable risk proxy, not a guarantee; censored preview, motion, noise,
full-well response, Bayer interpolation and scene change can invalidate prediction.
The whole bank remains present. Rule AE has the same admissible reference and TM-only
outputs must agree exactly between original and guarded variants.

Separate outcomes:
- Information: native RAW all-frame saturation fraction (simulator label, diagnostic
  only), observed missing fraction, full radiance MSE, fixed-mask bright radiance MSE.
- Policy: per-seed hard action histogram, modal share, categorical action entropy,
  softmax entropy, retained candidates, scene oracle actions and policy regret.
  More actions alone is not success; compare oracle need and best constant policy.
- Image: full display MSE/PSNR, subject luminance, gradient MAE, fixed-target bright
  region MSE and bright gradient error. J is secondary and unchanged.
  Fixed masks come from target, never from selected measurement. Empty regions
  recorded explicitly; avoid treating empty-region zeros as quality evidence.

Acceptance: guard excludes physically destructive high-exposure example, preserves
the rule and HDR short-frame coverage; disabled path exactly preserves scores;
training and reloaded deployment select identical actions from the same previews;
all existing tests remain passing. Scientific support requires improved measured
information AND display quality relative to same-run TM-only, across seeds/scenes,
with paired scene intervals. Validation CIs are descriptive, not confirmatory.

After this diagnostic, freeze any next protocol before a new independent test.
Challenge with information-first sequential AE+TM, a sufficiently trained stronger
TM-only baseline and matched HDR bank before claiming an independent joint benefit.
