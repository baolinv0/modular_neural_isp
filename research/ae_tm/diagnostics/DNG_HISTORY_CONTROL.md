# DNG history-information classical control

Status: **FROZEN 2026-10-09 before outcomes**. New independent local CPU
diagnostic. Do not rewrite DNG_PROTOCOL.json/DNG_EXPERIMENT.md or their outcomes,
the original CaptureTM budget/actions/GT/splits, or S24/formal test. No external
review API is part of this run.

## Inputs and frozen physical experiment

Read the existing private green cache and dng_audit.json for the SAME 68 bounded
processed DNG sources, eight 64x64 green patches each. Keep the audit's source
IDs, camera/pair/group IDs and train/development/diagnostic assignments, source
order, target T(x)=sqrt(x/(x+.5)), masks, noise, and actions unchanged. These
already noisy/possibly clipped sources are surrogate radiance, not clean RAW,
HDR, expert GT, calibrated cameras, or real repeated captures.

For each seed/source use the exact old SeedSequence([seed,pair_id,cam_id]).spawn(2),
with cam_id 0 for iphone and 1 for s25. Child 0 captures exactly three past
previews at exposure .25; child 1 captures each future action in the old order
[.125,.25,.5,1,2,4,8], using the same (repeat,patch,y,x) call. Sensor is
Poisson(1000*e*x)+Normal(0,4^2), max-zero, 1000-electron clipping, 12-bit rint.
Do not draw extra noise or combine future repeats. Every future repeat's
prediction uses H plus only that repeat's one future frame.

## Four renderer families

1. H_blind_inverse: original future-only exposure inversion, clip[0,1], tone.
   Its window candidate is 1.
2. H_blind_Wiener: original future-only wiener_target, windows 1/3/5/7.
3. H_aware_classical_fusion: the three H and one future pixel, using the
   following explicit approximate inverse-variance WLS. Its window is 1.
4. H_aware_fusion_Wiener: the same fused normalized estimate followed by a
   classical per-repeat/per-patch 2D Wiener, windows 1/3/5/7.

xhat is clip(mean(H_code/4095/.25),0,1), computed solely from H. For every
measurement with known e, normalized measurement variance is
(1000*e*xhat+4^2)/(1000^2*e^2)+1/(12*4095^2*e^2). Drop a measurement's weight
iff its observed ADC code is 4095; keep zero codes. Sum weighted normalized
H and the single future observation, divide by total weights, clip[0,1].
Nominal fusion variance is 1/sum(weights). This uses independence/unclipped
variance with noisy H-dependent weights: it is NOT a strict Poisson posterior.
Clipping, zero-code censoring, top-code selection, estimated weights and final
clipping create approximation/bias; variance is not calibrated after censoring.

All-weight-zero pixels explicitly fall back to estimate 1 and variance 1, the
bounded squared-range conservative numerical placeholder, with count recorded.
Fusion Wiener preserves those fallback pixels at 1. Normal .25 previews of
bounded sources should make this case absent; record it, do not hide it.

Fusion Wiener computes 2D local mean/variance on the clipped fused normalized
estimate with reflect boundaries and filter size (1,1,w,w). Its noise map is the local mean nominal fused variance in the SAME
2D window using uniform_filter size (1,1,w,w), with gain
clip(1-noise/max(local_variance,1e-20),0,1). It never filters across repeats or
patches. Clip resulting estimate[0,1], then tone. No neural network/risk model,
training image prior, Bayesian nodes, or new target is introduced.

## Development selection and strict backend cross replay

Each camera/backend independently freezes its window per exposure (trivial
window=1 for inverse/fusion), dev fixed exposure, and the original four
two-action threshold rules: mean brightness, q99 brightness, temporal noise,
local gradient. Threshold candidates are original train-preview quantiles .1
through .9; group-balanced development cost selects thresholds/actions/windows.
Feature computation and rule algorithm are reused exactly. All backends see
the same H/future realizations. Diagnostic costs may be precomputed as in the
old runner, but selector fitting accesses only train/development rows.

Replay every frozen selector against all four frozen renderers: 4x4 per camera.
A renderer retains its own dev-selected windows. Each cell compares against the
SOURCE selector's own dev-selected fixed e, never a reselected cell-specific e.
Report per-action group-balanced risks and paired rule-minus-fixed AND direct
rule-minus-mean-brightness cluster differences/95% percentile CI in every cell.
Do not infer necessity from separate significance versus fixed action.

Sources average within original background group, then groups get equal weight.
Bootstrap resamples the three diagnostic group means, not sources/crops/noise.
Intervals are descriptive, conditional on frozen selection, from one session.
Both cameras/related locations are not additional independent scenes. New
explicit reproducible CI SeedSequence child streams may differ from old CI.

## Output, validation, and completion

Output directory refuses overwrite. JSON and diagnostic-source CSV contain
only identities, protocol/cache hashes, per-action risks, frozen actions/windows,
statistics, configuration, run commit, software and timings. Do not save H,
future/real pixel cubes, original images or contact sheets. This runner has no
Bayesian integration-node comparison; its approximation scope is explicit.

Validate single-future-repeat isolation, weights against independent explicit
WLS, top-code dropping/zero-code retention/finite fallback, spatial patch/repeat
isolation, original H features/source RNG/future-only inverse and Wiener risks,
frozen cross-cell fixed-action semantics and legal inputs/outputs. Run necessary
tests and only a labelled single-seed small-repeat CLI smoke (all 68 cache sources
allowed). Commit only the three new files. Root performs code release and any
separate registered three-seed run; no positive outcome is a completion target.
