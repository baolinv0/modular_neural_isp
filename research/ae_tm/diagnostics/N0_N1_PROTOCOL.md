# N0/N1 continuous reference diagnostic protocol

Status: **FROZEN, 2026-10-09**. This independent simulated two-pixel diagnostic
does not modify/evaluate original capture_tm actions, training, split, budget,
expert GT, or S24 formal test. All evaluation populations are diagnostic.

## Physical model and continuous reference

Dark radiance a ~ U(.05,.30) and bright radiance b ~ U(.8,1.8) have uniform
(not log-uniform) marginals. The known training prior mixes a copula component
with weight w=.5 and an independent product of these marginals. The original
component is b=4*a+.6. Both targets are T(l)=sqrt(l/(l+.5)).
Future exposures are [.125,.25,.5,1,2,4,8]. Independently per pixel/capture:
N~Poisson(1000*e*l), R~Normal(0,4^2) electrons, followed by max-zero,
full-well clipping at 1000, and rint(4095*q/1000) 12-bit ADC. Ties have
probability zero. Analog/digital gains equal one. Loss is mean squared target
error averaged over the two pixels.

The renderer receives noisy ADC codes and known exposure, never source latents,
component IDs, or future noise. Its conditional target mean uses Gauss-Legendre
integration of the explicit prior. There is no observation binning or fitted
LUT. Independent-component evidence/moments separate into two 1D integrals;
the copula component uses one 1D integral. Nodes 32/64/128/256 and chunk sizes
are configuration. Rotation integrals split at the copula discontinuity.

The actual Poisson+read+ADC likelihood uses internal read kernels truncated at
plus/minus eight sigma. ADC zero contains lower clipping probability and ADC
4095 includes the complete upper Poisson tail. Conservative absolute per-cell
tail error is at most 3*Phi(-8) (about 1.87e-15), excluding floating-point
error. Required tests cover normalization, boundaries, clipped posterior, and
node convergence. This is a numerical physical-chain reference with a stated
tail bound, not a Gaussian approximation to shot noise or an unrestricted
image-model Bayes bound.

## Selection, independence, and controls

Seeds are 19/37/73. Each seed spawns independent train, dev, N0 diagnostic, N1
diagnostic, future-noise, and bootstrap streams. Sources are independent latent
pairs; repeated noisy captures are not independent sources. Each source gets
exactly three past noisy previews at known exposure .25. Policies use only
those previews. Train previews provide fixed candidate quantiles .1 through .9.
Dev expected costs estimated with repeated future noise choose threshold and
two actions. The scalar residual is abs(b_preview_mean-(4*a_preview_mean+.6));
the linear prior is explicitly known even during dependence shift. Dev also
selects one fixed action and equal-capacity two-action thresholds on global
preview brightness and preview temporal noise. Empty groups are excluded.
Rules/fixed action freeze before diagnostic evaluation; node comparisons reuse
rules selected once at the configured selection-node count.

JSON reports per-action expected risk, policy/fixed risk and paired differences,
independent-source bootstrap 95% percentile intervals, source/repetition counts,
node convergence on the same latents/future draws, sensor/software settings,
runtime, and exact train/evaluation prior parameters. Finite future repetitions
leave Monte Carlo error. A minimum of realized future losses is named only
hindsight_min_realized_future_cost: it uses unavailable future noise, is not
an oracle/causal bound, and is excluded from policy claims.

## N0 and N1

N0 uses training dependence. N1 preserves uniform marginals and changes actual
copula with separate train and held-out parameters. Components supported:
linear v=u, reverse v=1-u, rotation v=(u+shift) mod 1, with
a=.05+.25*u, b=.8+v. These maps preserve uniform marginals. Default N1
uses reverse with unchanged w=.5; only changing mixture weight is not the
default generalization test. The renderer always uses training prior and the
same frozen policies apply to both diagnostics. This is known-prior dependence
shift, not unknown full-image prior/real-camera generalization.

## Completion

CLI JSON output refuses overwrite. Generated arrays/caches are not committed.
--smoke is explicitly a small software/numerical smoke; root runs larger
three-seed/node comparisons separately. Tests use NumPy/SciPy/unittest, no
Torch/GPU. Completion requires physical-likelihood/posterior tests, one actual
small CLI smoke, and a local commit with only code, tests, and this protocol.
No positive policy gain is required. Extremely rare observations can underflow
floating-point likelihood/evidence and raise; no Gaussian fallback is used.
