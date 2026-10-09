# Independent AE review: single-frame and HDR joint algorithms

Reviewed 2026-10-05. Scope: `learned_policy.py`, `capture_plan.py`,
`joint_algorithm.py`, `joint_experiment.py`, the implementation plan, and their
focused tests. This is a source and experimental-protocol review; the reviewer
did not rerun the complete repository suite or inspect real-camera results.

## Verdict

Pass for the declared finite-candidate, one-shot simulation study after the HDR
candidate-bank correction below. No remaining blocking AE or training-protocol
issue was found. This verdict does not establish real-camera quality,
continuous AE-loop stability, or equivalence to either patent.

## Checked mechanisms

- AE consumes historical bounded sensor observations, their effective shutter
  and gain settings, and legal candidate settings. Neither the latent scene,
  future candidate captures, target image, nor teacher costs enter its inference
  signature. The actual-data `select`/`finish` interface does not require a scene
  or target. Simulated inference acquires one selected plan only.
- Every preview is encoded, and exposure-normalized temporal differences add
  motion evidence beyond the latest frame and per-frame histograms. Shutter,
  analog gain, and digital gain remain separate features. Equal-EV shutter/gain
  alternatives therefore retain physical noise/blur distinctions.
- HDR candidates preserve ordered slots and a common middle reference time.
  Historical preview shutters end before all default capture plans. The plans
  share the same frame count and allowed scheduling window, while actual total
  open-shutter time may differ and is recorded.
- Rule AE is scene-adaptive. Its HDR bracket remains fixed while the learned
  policy can select bracket width/asymmetry. All four groups share the same
  bank, front end, fusion backend, preview observations, and cached sensor
  samples. Frozen TM remains image-adaptive.
- Hard warm-start targets use costs averaged over configured noise repeats.
  Training scenes update parameters; validation scenes choose checkpoints.
  Test scenes do not enter either process. Manifest validation rejects known
  cross-split source/content identities.
- Expected cost is evaluated on separately captured candidates. The split
  backward pass uses detached probabilities for TM gradients and detached
  costs for AE gradients, giving the exact gradient for disjoint AE/TM
  parameters without mixing candidate images or double-counting gradients.
- Learned AE groups share a warm start; active modules have matching per-scene,
  per-noise, per-epoch update budgets. Validation may select different epochs,
  which is explicit in saved results. The paired contrast averages repeats and
  training seeds within scene before scene bootstrap.
- Final rendering uses effective readback settings rather than the requested
  action. Single-frame compensation occurs once; HDR fusion normalizes each
  observation once before TM.

## Finding resolved during review

The initial HDR bank varied only center shutter/gain and kept all bracket
offsets at `(-2, 0, +2)` EV. That could not test learned bracket selection.
The corrected bank contains twelve fixed-bracket and twelve narrow/asymmetric
plans, with unchanged reference time and legal shutter slots. The rule meter
uses the fixed-bracket subset; learned AE scores the full shared bank. The
focused test verifies multiple bracket shapes and the rule's fixed bracket.

## Test evidence inspected

Tests cover candidate-bank permutation, within-plan order, temporal motion
with identical histograms/latest frame, effective metadata sensitivity,
masked exact expected gradients, joint AE/TM gradients, small-gap warm starts,
and fitting scene-dependent action labels. Integration tests check all eight
groups update only assigned modules, selected-plan-only inference, effective
readback compensation, and HDR normalization. These are meaningful mechanism
tests; their passing status is recorded by the integration owner.

## Limits and reporting requirements

The policy makes one decision from three fixed historical previews. It is not
an implemented continuous feedback controller, sequential adaptive bracket
policy, or arbitrary continuous shutter optimizer. The sensor is an assumed
independent-RGB model, and the default plan bank and preview timing need real
camera calibration. The analytic scenes and small studies establish execution
and mechanism evidence only. The finite plan bank also bounds the best
achievable policy.

Dark/highlight region errors should be aggregated in group summaries, not only
kept in per-noise rows. Capture/fusion diagnostics should accompany the primary
cost so gains in appearance do not conceal information loss or motion failures.
This reporting improvement was sent to the integration owner; it does not
change the reviewed AE gradient or causal decision path.
