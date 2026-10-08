# DNG real-content relative experiment
Frozen before execution,2026-10-09. This implements DNG_PROTOCOL.json and uses dng_audit.py's local green patches. It does not run original neural training or original formal test.

The bounded processed DNG source x is a surrogate latent and target=T(x)=sqrt(x/(x+.5)). Additional Poisson(1000*e*x), Gaussian read SD4, clipping and12-bit ADC match the new N0 chain. Camera NoiseProfile tags are recorded, not used to infer an uncalibrated full-well. Existing noise, denoising, HDR merges and clipping in x cannot be undone by this experiment. There is no true-camera recapture claim.

Seven common relative exposures [.125,.25,.5,1,2,4,8], exactly3 previews at.25, and four independent future repetitions per source are used. No future noise/target enters policies. Original Apple12/Samsung24 timing/action constraints are not reproduced.

Backend1 is exposure inversion followed by fixed target tone transform.
Backend2 is conditional mean of target under a camera-specific, group-balanced training-only marginal prior.256 continuous-density bins and endpoint atoms define the fitted prior; quadrature uses4 Gauss-Legendre nodes per bin, with2-node comparison. A fixed1e-6 total density pseudomass is added before normalization. Every12-bit ADC code gets its own physical-likelihood posterior; ADC observations are not merged into empirical bins. This backend is an optimal point estimator under that estimated marginal prior; it is not a strong arbitrary image renderer or Bayes bound under unknown source distribution.
Backend3 is a simple local Wiener rule after exposure compensation, with spatial windows1/3/5/7. Mean/variance filtering never crosses patches or future repetitions. Noise variance uses shot+read+ADC assumptions. Development selects one window per exposure and camera, then freezes it. The rule is a classical simplicity control, not a learned strong image TM.

Each backend has its own development-selected fixed exposure and equal-capacity two-action threshold policies on preview mean,99th percentile, temporal-noise RMS and adjacent gradient. Candidate thresholds are train-preview quantiles .1 through .9. Development group-balanced risk chooses threshold and left/right actions. All policies freeze before diagnostic evaluation. Both camera samples from a paired scene share a source group, and source/crop/noise splits are never mixed.

Seeds19/37/73 use independent source-specific preview/future random streams. All backends share the same future observations. Pixel errors average within a source; source errors average within declared background group, and groups receive equal weight. Three diagnostic groups from one session provide descriptive conditional bootstrap intervals, not population-level generalization claims. Four repeated future captures reduce noise Monte Carlo error but do not create independent scenes.

Outputs: results.json plus source_metrics.csv with every diagnostic source's per-action risk and frozen policy actions. No pixel arrays, photos or contact sheets are uploaded. Source-cache hashes and protocol/commit identify inputs; private cache remains local.

~~~powershell
& $CapturePython -X utf8 research/ae_tm/diagnostics/dng_relative_experiment.py --cache '<local-pixel-cache-directory>' --audit '<local-report-directory>/dng_audit.json' --out '<new-local-experiment-directory>' --seeds 19 37 73 --repeats 4
~~~
The output directory must not exist. The driver refuses to overwrite a prior run. Positive or negative diagnostics are retained; no positive outcome is a completion condition.
