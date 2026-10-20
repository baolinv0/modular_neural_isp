# Independent ISP / HDR review

Reviewed on 2026-10-05: `capture_tm/capture_plan.py`, `joint_data.py`,
`joint_objective.py`, `joint_algorithm.py`, the physical target/cache path in
`joint_experiment.py`, and the associated focused tests. The review concerns
physical consistency and experimental validity, not measured phone quality.

**Decision: no remaining blocking ISP issue in the reviewed implementation.**
The implementation is suitable for the declared analytic RGB simulator
experiments. It does not establish performance on a calibrated mobile sensor.

## Verified properties

- Apple acquires one selected final exposure; its three AE observations are
  historical previews. Samsung acquires one selected three-exposure plan.
  `run_simulated` does not acquire all policy candidates at inference.
- All default burst plans use the same three slot centers and reference time
  zero. Shutters are ordered, nonoverlapping, and fit the same 100 ms allowed
  window. Equal-EV time/gain alternatives preserve the noise/motion tradeoff.
  Samsung's 24-plan bank includes variable bracket shapes; the rule baseline
  meters only within its fixed-bracket subset.
- Exposure integration occurs in linear signal before shot noise, full-well
  clipping, read noise, analog gain, ADC noise, and quantization. Digital gain
  remains one in the experiment banks. The simulator's calibration parameters
  are declared engineering assumptions.
- HDR fusion divides measurements by their effective exposure scale exactly
  once and divides variance by its square. It returns common-reference
  radiance with `capture_ev=0`. Apple returns capture units plus the effective
  EV used by TM. Actual readback settings, rather than the requested settings,
  are used by `finish`.
- Fusion computes variance and clipping indicators from measured RGB and the
  sensor profile. It does not use simulator latent radiance, oracle variance,
  or oracle saturation masks in its decisions. Integer registration and local
  disagreement rejection are observation-only.
- WB and CCM are applied once after HDR fusion. Both capture paths retain
  positive values above one. The missing-information mask remains in sensor
  channels and is used as a separate information-availability proxy, rather
  than being interpreted as an output-color error map.
- The ground-truth display image is generated from the sharp latent image at
  the common reference time, through a fixed color transform and fixed target
  renderer. It is independent of selected exposure and trainable TM. All four
  groups share candidate measurements, noise repeats, and targets.

## Evidence

- The capture builder reports 11 passing focused physical-capture tests,
  including temporal causality, actual-measurement composition, observed-only
  fusion, camera translation, moving-object rejection, and bracket selection.
- Independent reviewer probe evaluated all 24 HDR plans at three constant
  radiance levels with nonidentity WB/CCM and a noiseless 20-bit sensor.
  Nonmissing fused outputs agreed with the once-transformed reference to a
  maximum absolute error of **1.91e-6**; every HDR output retained zero capture
  EV. This exercises asymmetric brackets and analog gains beyond one fixture.
- Reviewer command:

  ```text
  python -m pytest -q tests/test_capture_tm_joint_algorithm.py tests/test_capture_tm_joint_data.py tests/test_capture_tm_joint_objective.py
  ```

  Result: **13 passed in 8.68 s**. The facade tests cover selected-plan-only
  inference, actual effective exposure compensation, and one HDR normalization.

## Resolved clarification and limits

The initial objective docstring called the single-frame missing mask a
saturation mask. It now states the actual definition: no accepted observation
has an unclipped signal above two estimated noise standard deviations.
Invalid alignment support and motion rejection also affect the HDR mask.
This is an **observed information-availability proxy**, not latent ground-truth
irreversible information loss. The declared cost can use it, but result tables
must retain that interpretation.

The fixed fusion backend estimates bounded global integer translation. It
does not solve rotation, subpixel registration, parallax, rolling shutter, or
independent motion hidden by a clipped reference frame. Rejection can preserve
the observed reference but cannot recover details absent from reliable
measurements. All-clipped fallback is a measured lower bound, not reconstructed
highlight detail.

The sensor model uses independent RGB noise without CFA, demosaicing,
correlated/row noise, lens effects, or measured mobile sensor calibration.
Analytic temporal interpolation and moving synthetic objects provide controlled
protocol checks rather than a real HDR-video or phone benchmark. These limits
are documented and do not invalidate the declared simulator comparisons.

Image MSE in dark regions includes bias and detail error; it is not a pure
noise estimate. Repeated-noise output variance complements it. A positive
factorial interaction in the declared cost alone cannot identify the full
patent mechanism or establish perceptual quality improvement.
