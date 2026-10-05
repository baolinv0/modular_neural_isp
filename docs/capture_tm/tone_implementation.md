# Differentiable conditional tone mapping

`capture_tm.learned_tone.ConditionalToneMapper` implements both schemes with the
shipped S24 style-0 checkpoint, full original predictor resolutions, and 2,663
trainable adapter parameters. All checkpoint parameters are frozen. This is
parameter-efficient adaptation of the original learned gain/GTM/LTM, **not**
full-backbone fine-tuning or an exact implementation of the patent claims.

## Domains and unified gain

For Apple, the renderer first converts capture RGB to reference exposure units:
`R = X / 2**capture_ev`. For Samsung, fusion has already done this conversion and
`R = H`; changing `capture_ev` has no effect. Negative or nonfinite input is an
error. `render_ev` describes independent output appearance and is restricted to
[-16,16] EV for numerical control, as is physical `capture_ev`.

The pretrained GainNet sees an exposure-normalized, smoothly compressed view.
Its prediction, explicit rendering intent, and the learned residual form one
coordinated gain:

`G = G_stock(R) * 2**render_ev * 2**delta_ev`.

This gain is applied once before compression. The full original renderer is not
called afterward, so there is no hidden second GainNet application. Gain and
capture compensation are not used as replacements for physical AE: clipped
sensor information cannot be recovered by this renderer.

## HDR compression and original tone operators

A ratio-preserving shoulder is linear below knee `k` and uses
`f(y) = 1 - (1-k)**2 / (y + 1 - 2*k)` above it, where `y = max(R,G,B)`.
It is continuous with continuous first derivative, remains monotone, and tends
to one without hard-clipping finite highlights before GTM. Adapters adjust the
knee around 0.7 and multiply the checkpoint's three positive GTM coefficients.
The original `apply_tm` equation is retained.

The original LTM predictor supplies five spatial coefficient maps. Global and
spatial adapters multiply these maps, with reliability gating local changes.
The original local `x*g` followed by a hard input clamp is replaced by
`x*g / (1 + x*(g-1))`, a bounded monotone gain on [0,1]. Local and global outputs
retain the original sigmoid blend. The pretrained chroma LUT and gamma are
predicted from the canonical reference view and applied to the adapted output;
their weights and predicted style stay independent of trainable tone adapters.
This is a deliberate appearance-control design, so output is not bit-identical
to the stock pipeline even when all adapters are zero.

## Samsung virtual-gain curve construction

Source and virtual-target PDFs are computed with soft histogram assignments on
the same display-intensity axis. The target samples are `clip(R * 2**render_ev)`;
the source samples are `clip(R)`. Both complete PDFs enter the conditional neural
head, along with physical/render metadata, radiance statistics and confidence.

A fixed small difference-of-PDF-moments term also adjusts the shoulder knee.
Therefore virtual gain changes curve construction even before adapter training,
including in the frozen-TM group. This is a learned histogram-conditioned
adaptation of the Samsung virtual-gain idea, not reproduction of its exact
inverse-CDF construction. The direct virtual gain is applied only once to the
radiance; PDF calculation does not multiply the image a second time.

## Trainability, caching and fair experiments

The conditional MLP and spatial adapter have zero-initialized final layers.
Frozen and trainable variants therefore produce exactly the same initial output
for the same stock checkpoint. Set `trainable=False` to freeze their parameters;
there is no `torch.no_grad` in the renderer, so gradients to inputs remain valid.
The module's `.train()` always leaves stock coefficient predictors in eval mode.

`prepare(x, capture_ev, reliability)` computes only frozen predictions and returns
a flat dictionary of batch-leading tensors: `radiance`, `capture_ev`,
`reliability`, `gain`, `gtm`, `ltm`, `chroma_lut`, `gamma`, and `base_view`.
`render_prepared(prepared, render_ev)` executes all trainable operations.
Precomputation is exact when images/capture metadata, stock weights and sensor
noise realizations are fixed. Recompute it after changing any of these inputs.
Only use detached caches when gradients to the input capture itself are not
needed, as in discrete action expectation training.

Caching avoids recomputing the frozen 384-pixel LTM predictor for every training
epoch and candidate. It does not approximate the predictor or change its input
resolution. A prepared batch may be concatenated or sliced by leading dimension.
Cache identity must include checkpoint, scene/action/noise and preprocessing.

## Verification and limits

The dedicated tests cover single compensation, absence of duplicate Samsung
normalization, useful HDR information above one, rendering-intent response,
finite input gradients with frozen parameters, loss gradients that update
trainable TM, cached/direct equality, batch EV and invalid inputs.

No real-camera aesthetic-quality claim follows from these mechanism tests.
Adapters can alter appearance, but they cannot reconstruct saturation, replace
multi-frame alignment, or certify patent scope. The patent-inspired design
sources are [Apple US9432647B2](https://patents.google.com/patent/US9432647B2) and
[Samsung US12243201B2](https://patents.google.com/patent/US12243201B2/en).

The Samsung mechanism test also isolates PDF-conditioned curve construction from
ordinary image gain: it replaces only the virtual-target PDF with the source PDF
while keeping the same HDR input, coefficients, zero adapters and direct virtual
gain. At render EV zero both paths are identical; at render EV -1 and +1 they
differ. The complete path stays finite and mostly pixelwise monotone across
[-1,0,+1] EV on the HDR probe. This verifies an active mechanism, not an aesthetic
quality improvement. A factorial study using only `render_ev=0` does not test the
benefit of nonidentity virtual-gain PDF conditioning; that requires an additional
held-out rendering-intent sweep or ablation with fixed capture plans.
