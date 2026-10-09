# Independent TM review: joint single-frame and HDR algorithms

Review date: 2026-10-05. Scope: `learned_tone.py`, `joint_objective.py`,
`joint_experiment.py`, their focused tests, and `tone_implementation.md`.
This review did not modify production code.

## Verdict

No blocking algorithm defect found in the reviewed TM or joint-gradient paths,
including the Samsung v2 virtual-gain correction reviewed before its study run.
The implementation is suitable for the stated simulator factorial experiment,
subject to the experiment interpretation below. This is parameter-efficient
adaptation of a real pretrained ISP, not full-backbone training and not a
claim-by-claim reproduction of either patent.

## Mechanisms checked

- The real repository S24 style-0 checkpoint is loaded strictly, with SHA-256
  `5137b1a9da936814544a0259add95530e124d954fac8e08ece61333be630c09f`.
  Stock gain, GTM, LTM, chroma and gamma predictors remain frozen and
  image-adaptive. There are 2,663 trainable adapter parameters.
- Apple converts capture values to reference exposure units exactly once.
  The later coordinated gain contains the stock prediction, render intent and
  learned residual; it does not call a second full gain pipeline.
- Samsung accepts already normalized fusion radiance and does not divide it
  by a reference capture exposure again. `capture_ev` is deliberately ignored
  for this renderer.
- HDR radiance is smoothly compressed before the original bounded GTM
  equation. The local gain uses a bounded odds mapping, avoiding the stock
  local multiplication followed by destructive clipping above one.
- Samsung v2 computes the preliminary target PDF using the stock GainNet gain
  times explicit rendering intent. The conditioner predicts a residual once,
  then a final target PDF uses the complete effective gain, including that
  residual, to construct the shoulder-knee term. The same complete gain is
  applied once to radiance. This is an acyclic histogram-conditioned neural
  adaptation, not the exact inverse-CDF patent construction.
- Frozen and trainable TM variants have exactly the same initial output.
  Freezing coefficients does not impose `no_grad` on the input path.
- `prepare` executes no trainable adapters. An independent probe randomly
  perturbed every trainable adapter parameter: every prepared tensor remained
  bit-identical, and cached and direct rendering remained bit-identical after
  the update for both schemes. Detached caches are therefore valid for this
  finite-discrete-candidate training objective.
- The runner constructs the target once from sharp latent radiance at fixed
  time zero, with fixed WB/CCM and an analytic rendering target. It does not
  use an action-dependent or current-TM-dependent target.
- The split backward pass uses detached AE probabilities for TM gradients,
  and detached candidate costs for AE gradients. For disjoint policy and TM
  parameters these sum to the exact gradient of the declared discrete
  expectation. Candidates remain separately rendered physical captures.
- All four groups within a scheme load the same initial TM state and capture
  cache; group flags control which parameters update. Validation cost selects
  checkpoints, and test scenes do not participate in optimization or selection.

## Evidence

Focused execution in the task Python environment, with one CPU thread:

```text
pytest -q tests/test_capture_tm_learned_tone.py tests/test_capture_tm_joint_objective.py
11 passed in 16.38 s

pytest -q tests/test_capture_tm_joint_experiment.py::test_chunked_expected_gradient_matches_direct_joint_gradient tests/test_capture_tm_joint_experiment.py::test_scene_bootstrap_keeps_paired_factorial_contrast
2 passed in 2.93 s
```

Independent two-scheme perturbation probe:

| Check | Apple | Samsung |
|---|---:|---:|
| Frozen versus trainable initial output exactly equal | Yes | Yes |
| Maximum prepared-tensor change after adapter perturbation | 0 | 0 |
| Maximum cached/direct output difference after perturbation | 0 | 0 |
| Trainable parameters | 2,663 | 2,663 |

## Samsung v2 scoped re-review

**Resolved conceptual issue:** The first version constructed its target PDF
from explicit user intent alone. At the default zero intent this left the
PDF-derived curve inactive, even when the stock GainNet or learned residual
brightened the image. Samsung's image-statistics-derived virtual-gain idea is
better represented by the total estimated rendering gain. The v2 construction
above resolves this mismatch without a recursive predictor or duplicated image
gain.

Code and tests were inspected after the builder completed the correction. The
builder's dedicated TM suite reported **11 passed in 20.97 s**. The scoped
review did not duplicate the concurrently running integration suite.

- A PDF-only ablation keeps the same prepared image, gain, coefficient maps
  and weights. At stock gain two and zero explicit intent, removing the PDF
  term changes the output. At effective gain one, the outputs match exactly.
- A residual-equivalence test compares stock gain two plus a learned half-EV
  residual with stock gain `2 * sqrt(2)` and zero residual. Equal final gains
  yield equal curves and outputs, detecting an incorrect preliminary-only PDF.
  It also verifies a finite nonzero residual-gain gradient.
- The builder's before/after Apple probe used nonzero randomized adapters and
  reported bit-identical output, maximum absolute difference zero. Inspection
  confirms Apple's conditioner inputs and gain arithmetic are unchanged.
- `prepare` is unchanged. Only Samsung's renderer identity advances to v2;
  Apple retains v1. Existing Apple results therefore remain applicable.

This correction was accepted before Samsung model training or quality results
were observed. The experiment owner reported interrupting only initial cache
construction. The change addresses algorithm fidelity rather than selecting
an implementation using observed Samsung test quality.

## Experiment interpretation

Samsung v2's virtual-gain curve can be active at zero explicit intent because
its automatically estimated gain is generally nonunit. The focused tests prove
that this dataflow changes the curve. The four AE/TM factorial groups do not
isolate the PDF term's **quality benefit**: that conclusion still requires a
separate, controlled PDF-only ablation with identical total gain and budgets.

The fixed analytic target and simulator metrics are transparent research
surrogates. They do not establish real-camera aesthetics, perceptual superiority,
or that the exact patented traditional curve is superior. The scene-bootstrap
interval is conditional on the executed training seeds and scene distribution.
