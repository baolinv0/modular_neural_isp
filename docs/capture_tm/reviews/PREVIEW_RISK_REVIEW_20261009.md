# Independent preview-risk diagnostic review, 2026-10-09

Read-only reviewer inspected code, saved artifacts, analytic reference and the
censoring witness independently of implementation. This is a software/mechanism
review, not a blind image-quality study or independent test-set validation.

Issues found and addressed:

1. External Samsung feasible subsets may lack a fixed bracket. Deployment now
   selects a minimum predicted-risk feasible reference, intersects both masks,
   and has a regression test. Full-bank behavior is unchanged.
2. Empty bright regions must not become zero-valued quality evidence. Dedicated
   means/intervals exclude absent regions and expose the independent scene count;
   legacy fields remain traceable. Fresh main-script invariance checks restrict
   equality to rule image/information outcomes rather than changed oracle sets.
3. Saved reports were regenerated without re-training or threshold adjustment.
   A fresh 12x16, seed0, three eight-group smoke also completed with current code.

Final control review found no blocking defect. It confirmed:

- Analytic TM consumes observed captures, not GT/latent. However, it knows the
  exact target curve and is a post-hoc toy control, not a capacity-matched trained
  or expert-style baseline. Rendering cannot change radiance information.
- Apple guarded AE-only + analytic TM beats guarded joint on reported global
  MSE, gradients, radiance and conditional bright-region MSE. Different AE
  choices explain the capture-information difference.
- Samsung analytic controls do not dominate bright-region quality: guarded
  joint MSE 0.010408 versus analytic control 0.024366 on two bright scenes.
- The Samsung guard is structurally inactive for this bank: each plan has a
  minimum frame scale <=0.94 <0.98. All bounded previews predict zero all-frame
  clipping. Original/guarded equality cannot assess an effective HDR constraint.
- The witness is valid with actual rule index3 as well as the placeholder
  reference: all 24 plans pass while plan11 actually saturates everywhere.
  Noise-free final capture isolates clipping; it is not a quality benchmark.
- Validation intervals are descriptive, conditional on reused eight validation
  scenes and these runs. Only two scenes contain the fixed bright region.

Full suite verified: **394 passed, 9 skipped**, no failures/errors (185.85s).
Skipping CUDA/OpenEXR tests leaves those execution paths unverified here.
Review supports withholding any independent joint-benefit claim.
