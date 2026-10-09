# Physical capture plans and fixed HDR fusion

Implementation: `capture_tm/capture_plan.py`. This is the common physical
protocol for all four Apple groups and all four Samsung groups. It is a
research RGB sensor model, not a calibrated Bayer phone pipeline.

## Capture protocol

Apple has twelve single-exposure actions: shutter factors (1/8, 1/2, 1, 4)
relative to the reference shutter capped at 1/120 second, and analog gains
(1, 2, 4). With the default sensor this spans 1/960 to 1/30 second. The short
option retains unsaturated signal just below eight reference-white units;
the longest option uses the full dark-scene shutter allowance. This bank is
fixed before the study and shared by all four groups. Digital gain is one.
Equal-EV shutter/gain alternatives are
intentional: they permit photon noise and blur to change independently of
output brightness. Every final exposure is centered at time zero.

Samsung has twenty-four three-frame plans. Twelve plans use a fixed
(-2, 0, +2) EV bracket. Twelve additional plans offer (-2, 0, +1),
(-1, 0, +2), and (-1, 0, +1) brackets at two center shutters and two gains.
The learned policy therefore chooses both center and bracket shape. The
frozen rule meter is constrained to the fixed-bracket subset. All groups
share the same allowed bank, three-frame count, and physical scheduling
limits. Frame centers are (-1/30, 0, +1/30) seconds, shutters never overlap,
the maximum shutter is 1/30 second, and every plan fits the common
[-0.05, +0.05] second window. The second frame always defines reference
geometry at time zero. Actual summed shutter-open time can differ within
that shared budget.

Three noisy previews use centers (-4/30, -3/30, -2/30) seconds and a shutter
no longer than 1/120 second. All previews end before any final capture can
start. A moving scene needs temporal support covering these observations
and the final burst; the simulator rejects unsupported intervals rather
than substituting future or repeated frames. Static singleton scenes are
supported directly. Preview metadata contains effective physical exposure.

The frozen meter estimates reference radiance from previews and metadata,
meters median intensity toward 18%, limits the high quantile, and breaks
equal-EV ties with an observed-motion estimate. It is scene-adaptive rather
than a constant exposure index.

## Fixed F0 fusion

1. Divide each black-subtracted sensor RGB frame by its effective
   `(t / t_reference) * analog_gain * digital_gain` exactly once.
2. Estimate shot/read/ADC/quantization variance using observed signal and
   the sensor profile, then divide variance by squared exposure scale.
   Simulator oracle variance and oracle saturation masks are never read.
3. Register each side frame to the middle exposure with bounded global
   integer translation. A robust log-intensity matching cost ignores
   observed clipped pixels and trims a minority of moving-object residuals.
4. Reject pixels inconsistent with the reference beyond estimated noise
   and relative-radiometric tolerance. Share rejection across channels to
   avoid combining different object colors. Use the reference where side
   frames disagree, and inverse-variance fuse reliable aligned measurements.
5. Mark information missing where no accepted frame has an unclipped signal
   above two estimated noise standard deviations. All-clipped locations
   retain only the largest measured radiance lower bound for rendering and
   retain an explicit missing-information flag. This is an observed
   useful-signal proxy, not a latent ground-truth unrecoverable-information
   metric; reports must retain that distinction.
6. Apply fixed white balance and the color correction matrix once, after
   fusion. Return HDR radiance to TM with `capture_ev=0`. The Apple path
   instead returns physical capture units plus the effective `capture_ev`.

`compose_captures` is the deployed boundary: it accepts actual measurements,
effective actions, and timestamps, without a scene or target. It validates
finite measurements and ordered nonoverlapping shutters. `render_capture`
uses the same boundary after simulated acquisition. Both paths preserve
HDR values through the fixed front end; only negative color values are
clamped.

F0 has no learned parameters and is identical across the four HDR groups.
Its bounded translation model does not solve rotation, parallax, rolling
shutter, subpixel registration, or independently moving clipped reference
regions. Local motion rejection preserves the observed reference instead
of hallucinating hidden details. These are explicit baseline limitations,
not claims of production-quality HDR deghosting. A stronger frozen fusion
backend can later replace F0 without changing the experimental factors.

Validation covers legal/shared time support, equal-EV alternatives,
scene-adaptive metering, bracket-shape choices, causal previews, radiance
units, clipped-reference highlight recovery, independence from simulator
oracle fields, global translation, motion rejection, and actual-measurement
composition with one color front end.
