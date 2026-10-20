# History-aware continuous reference and backend cross-evaluation

Status: **FROZEN, 2026-10-09**, before this runner's outcomes. This is a new,
independent simulator diagnostic. It preserves and does not rewrite N0/N1
protocol/results, original capture_tm actions/training/budget/GT/splits, or S24
formal test. It is not real-camera evidence.

## Shared model and conditioning

Reuse continuous_reference.py without changing its physical model, ADCLikelihood,
Prior, quadrature, tone, actions, source generator, or noise generator. Marginals
are a~U(.05,.30), b~U(.8,1.8); default known training prior has weight .5 on
b=4*a+.6 and weight .5 on independent uniforms. The common target is
sqrt(l/(l+.5)). Future exposures are [.125,.25,.5,1,2,4,8]. Sensor means are
1000*e*l Poisson electrons plus independent Normal(0,4^2) read noise, max-zero,
1000-electron clipping, and rint to 0..4095 ADC codes. Known exposure is metadata.

H is exactly the three already-available, independent noisy two-pixel previews
at exposure .25. H-aware rendering estimates E_train[T(L)|H,Y_future,do(e)],
multiplying all history and future likelihoods in the mixed independent/copula
continuous integrals. H-blind is the existing E_train[T(L)|Y_future,do(e)] known
training-prior conditional reference. It intentionally omits H and is not a
claim of Bayes optimality conditional on a history-selected observational e.

Both prior and noise are known toy assumptions. Given H, a deterministic
policy's selected action adds no information about L; e is an acquisition
intervention, never an additional latent/component observation. Renderer API
takes only ADC observations and known exposure metadata. It does not accept
source latents, targets, component labels, selected-policy identity, or unseen
future observations. N1 changes the true copula (default reverse, same weight
.5 and uniform marginals), while both renderers retain the training prior.
Neither renderer is claimed optimal under unknown priors or N1 prior shift.

Individual ADC likelihoods retain the old plus/minus-eight-sigma read-kernel
bound and complete boundary Poisson tail. Products are accumulated in log space
and rescaled jointly across independent/copula grids per pixel, preserving
relative component evidence. A rare ADC likelihood may itself underflow in the
shared kernel and raises; there is no Gaussian substitution or silent fallback.
History lookup/index storage plus observation chunks avoids tiling history over
all source-by-future-repeat-by-node combinations. Node convergence remains
numerical evidence, not an analytic exactness claim.

## Frozen selection and controlled comparisons

Reuse the N0 independent streams exactly: seed children 0/1 train sources/H,
2/3/4 dev sources/H/future noise, 5/6/7 N0 sources/H/future noise,
8/9/10 N1 sources/H/future noise, 11/12 source bootstrap. All source counts,
three previews, exposures, target, priors, and future-repeat count are shared
between renderers. Defaults are seeds 19/37/73, nodes 128/256, selection nodes
256, sixteen future-noise repeats; --smoke is a distinctly labelled small run.

Train H determines scalar threshold candidate quantiles .1 through .9. Dev
source-averaged repeated-future costs independently select each backend's fixed
action and each known-linear-residual/global-brightness/temporal-noise
two-action threshold rule. Definitions and selection algorithm are reused from
N0; the residual explicitly knows b=4*a+.6. Rules may choose the same action on
both sides. Policies never inspect source identity, latents, family labels,
targets, or future noise at inference. No policy refit occurs on diagnostics.

Replay each frozen source selector (trained against H-blind or H-aware costs)
against each renderer on the SAME diagnostic sources/H/future ADC cube. Report
the strict 2-selector-by-2-renderer matrix. The source selector's own dev-chosen
fixed action is the comparison within each matrix cell; do not reselect it for
the other renderer. Also compare residual versus brightness within each cell.
Freeze policy selection once at selection_nodes and reuse for every evaluated
node count. Repeated captures of one source remain one source bootstrap unit.

## Evidence and completion

JSON includes physical/model/conditioning scope, priors, independent-stream
mapping, source/repeat/node configuration, per-action risk, matrix point risks,
highest-node paired rule-minus-fixed and residual-minus-brightness 95% percentile
independent-source bootstrap CI, node differences on identical future draws,
software and phase runtimes. NPZ records ordered source IDs, history ADC codes,
future ADC codes, source cost arrays for every node/backend/action, preview
features, both selectors' frozen actions/fixed indices, and dimension labels.
Generated source arrays/caches remain outside Git. Reports and NPZ refuse
overwrite. CIs condition on frozen policies and finite repeated captures and
do not include policy-selection uncertainty.

Required tests: no-history matches old reference; explicit likelihood-product
conditioning; boundary/stable posterior; history permutation; quadrature and
chunk agreement; observation-only interface; source/repeat indexing and invalid
shape rejection; strict cross-evaluation arithmetic with frozen selectors.
Completion requires those numerical tests, one actual small CLI smoke with
source arrays, and a local commit of only the new implementation, tests, and
this protocol. Larger root runs and scientific interpretation remain separate;
no positive result is required.
