# Single-frame and HDR AE–TM implementation

**Goal:** Implement both algorithms before running their four-group experiments.

**Architecture:** A shared causal temporal action scorer controls either one exposure (Apple-inspired) or a three-exposure plan (Samsung-inspired). A physical simulator feeds a fixed front end and, for HDR, a fixed observational alignment/fusion backend. Conditional adapters coordinate the original Modular Neural ISP gain, GTM and LTM. Training enumerates real candidate captures and differentiates expected final-image cost through the policy probabilities and TM.

**Tech stack:** Existing Python/PyTorch repository and original photofinishing checkpoint; CPU execution supported. No changes to the original model/checkpoint required.

**Specification:** User-approved single-frame/HDR designs and four-group factorial experiments; prior patent/source audit in `docs/capture_tm/SOURCES.md`.

## Global constraints

- A00/A10/A01/A11 and S00/S10/S01/S11 mean rule/learned AE × frozen/learned TM.
- Rule AE is scene-adaptive. Frozen TM remains image-adaptive. All groups share architecture, initialization, legal actions, observed previews and physical budget.
- Only historical captured previews and effective metadata enter AE. Latent scene/targets/candidate outputs are training/evaluation-only information.
- Apple compensates capture exposure once. HDR inputs normalize to radiance once; no reference exposure correction after fusion.
- Final reference time is fixed at zero, HDR frame centers and frame count identical across groups. Shutters fit non-overlapping slots.
- Target appearance is independent of current TM, candidate and group. Training, validation and test scenes are separate.
- Discrete expected loss preserves gradients to policy and TM. Inference captures one selected physical plan; it never mixes candidate images.
- Trainable adapters, fixed fusion and RGB sensor approximation are disclosed. Synthetic experiments establish execution/mechanism evidence only.

## Tasks and ownership

1. Root: fixed target/objective, diverse synthetic scenes, algorithm facade, integration and documentation.
2. TM implementer: `capture_tm/learned_tone.py` and focused tests. Real checkpoint, unified compensation/gain, conditional GTM/LTM, valid gradients.
3. Capture implementer: `capture_tm/capture_plan.py` and focused tests. Action bank, causal previews, scene-adaptive meter, observation-only HDR fusion.
4. AE implementer: `capture_tm/learned_policy.py` and focused tests. Temporal observations, ordered plans, masking, differentiable expectation and warm start.
5. Experiment implementer: `capture_tm/joint_experiment.py`, `joint_cli.py` and integration tests. Matched four-group loops, validation selection, saved checkpoints and scene-paired results.
6. Independent AE, TM and ISP reviews after integration; fix concrete findings before study execution.
7. Run both four-group CPU studies, record measured results and limitations, update existing draft branch/PR where available.

## Review focus and validation

Meaningful tests cover causal timing, legal physical controls, exposure/radiance units, static HDR recovery and motion fallback, real TM and AE gradients, immutable fixed targets, group-specific parameter updates and runnable inference. Run focused tests during construction, then the repository test suite once integrated. Run a small study on all eight groups with repeated noise seeds and training seeds if CPU runtime permits. Report exact data scale and training schedule. Measure lower-is-better fixed cost J and interaction I = J10 + J01 − J00 − J11 with scene bootstrap uncertainty. Do not infer real-camera quality or patent equivalence from these runs.

## Progress

- Initial inspection: existing branch clean at 5a09417; old frozen renderer is not a joint training implementation.
- Parallel implementation started with disjoint module/test ownership. Root retains integration and git writes.
- Completed both algorithms, deployed-observation APIs, parameter-efficient TM training, temporal AE, physical HDR capture/fusion, and the factorial runner before final experiments.
- Independent AE/TM/ISP reviews completed. Samsung target-PDF handling was corrected to include actual GainNet/residual virtual gain before its first training run; Apple computations were preserved.
- Full regression on final algorithm code: 195 passed in 149.32 seconds.
- Completed all eight groups with training seeds 0/1/2, 10 warmup and 10 main epochs, 40 synthetic 32x32 scenes (24/8/8), and two noise repeats. No real-camera validation.
- Verified selected trained Apple and Samsung checkpoints through actual capture readback inference against cached evaluation within 1e-5 float32 tolerance.
- Results and independent result review are in `docs/capture_tm/JOINT_RESULTS.md`, `JOINT_RESULTS_REVIEW.md`, and `joint_results.json`. Apple AE collapse/saturation and Samsung joint policy concentration are reported; neither scheme establishes joint superiority over TM-only.
