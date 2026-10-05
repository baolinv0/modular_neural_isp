# Independent TM review: joint single-frame and HDR algorithms

Review date: 2026-10-05. Scope: `learned_tone.py`, `joint_objective.py`,
`joint_experiment.py`, their focused tests, and `tone_implementation.md`.
This review did not modify production code.

## Verdict

No blocking algorithm defect found in the reviewed TM or joint-gradient paths.
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
- Samsung virtual gain affects a target PDF and a shoulder-knee term, as well
  as the single direct rendering gain. This is a disclosed histogram-conditioned
  neural adaptation, not the exact inverse-CDF patent construction.
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

## Experiment interpretation

The standard runner defaults to `render_ev=0`. In that configuration Samsung's
explicit virtual gain is one, so its source and virtual-target PDFs coincide.
The four groups still test capture/TM learning and their interaction, but cannot
by themselves establish the value of a nonidentity virtual-gain curve. A
controlled nonzero-intent probe, with direct gain held equal across a
PDF-construction ablation, is needed to isolate that mechanism. This issue was
communicated to the implementer and experiment owner.

The fixed analytic target and simulator metrics are transparent research
surrogates. They do not establish real-camera aesthetics, perceptual superiority,
or that the exact patented traditional curve is superior. The scene-bootstrap
interval is conditional on the executed training seeds and scene distribution.
