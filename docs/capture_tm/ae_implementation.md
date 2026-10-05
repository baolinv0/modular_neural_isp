# Temporal AE for single-frame and HDR capture

`capture_tm.learned_policy.TemporalExposurePolicy` scores a bank of legal
physical capture candidates using only already observed previews. It uses the
same interface for one-frame Apple-inspired exposure control and three-frame
Samsung-inspired HDR plan selection. It is a new lightweight implementation
inspired by AdaptiveAE's observation design, not its official network or RL
training reproduction.

## Inputs and physical conventions

- `previews`: `[B,T,3,H,W]`, sensor-normalized RGB in `[0,1]`, oldest to latest.
- `capture_state`: `[B,T,3]`, settings actually used for each preview.
- `candidate_features`: `[B,K,N,3]`, where `N=1` or `N=3`.
- `feasible_mask`: `[B,K]`, boolean, at least one legal candidate per row.

Both state and candidates use the existing `action_features` convention:
`[log2(t/reference_t), log2(analog_gain), log2(digital_gain)]`.
Equal total EV does not erase the distinction between shutter and gain. This
allows learned preference for different noise/blur tradeoffs. For HDR, candidate
frame order is preserved; the middle frame is the reference-time slot. A
single-frame candidate occupies that slot with absent outer slots.

## Network

The default width-16 network has 56,177 trainable parameters. A shared two-layer
CNN processes every preview, retaining a 2-by-2 spatial summary. A second shared
CNN processes each adjacent pair's raw signed differences and exposure-normalized
signed differences. Normalization uses observed effective exposure and `log1p`
compression; it does not reconstruct clipped radiance. Mean and maximum absolute
differences provide explicit motion statistics. Soft RGB histograms, effective
capture state, and presence flags complete the temporal observation vector.

Short histories are left-padded as absent, with no synthesized future frame.
The ordered candidate plan has a separate MLP. A shared scorer receives context,
plan encoding, and their elementwise interaction. Candidates are scored
independently; permuting the candidate bank permutes scores without changing
preferences. Reordering frames *within* one HDR plan changes its encoding.

Inference uses `scores.argmax(-1)` and captures exactly one selected candidate.
Illegal candidates have score negative infinity. The policy does not consume
clean radiance, candidate captures, ground truth, future frames, or oracle costs.

## Training

`warmstart_loss(scores, costs, feasible_mask)` applies cross entropy to the
best legal candidate, with equal target mass for exact ties. Costs are detached
labels during this stage. Unlike a broad soft utility target, a small but
resolved cost gap still teaches a definite best action. Repeated-noise average
costs should be used to avoid fitting a single noise realization.

`expected_quality_loss(scores, costs, feasible_mask, temperature=1.)` computes
the mean of `sum_k softmax(scores / temperature)[k] * costs[k]`. Every cost must
come from its own independently captured and rendered candidate. The cost tensor
stays attached to its renderer computation, providing gradients to both AE
probabilities and trainable TM parameters. This exact discrete expectation does
not require differentiating the sensor's Poisson sampling or quantization.
It does not blend candidate RAWs or outputs into an unrealizable capture.

NaN/inf in illegal cost or score entries is masked before objective arithmetic.
Nonfinite legal entries and all-illegal rows fail explicitly.

## Focused validation

`tests/test_capture_tm_learned_policy.py` covers candidate permutation and
legality, HDR plan order, motion histories with identical per-frame histograms
and latest frame, effective-state influence, exact expected cost and masked
gradients, real AE/TM parameter gradients, weak-gap warm starts, exact ties, and
fitting a toy dataset with scene-dependent optimal actions. The toy fit is an
optimization check, not image-quality evidence.
