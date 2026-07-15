# TM Baseline Evolution MVP — Design Specification

## Goal

Prove, without image generation, that a frozen IQA can select scene-conditioned outputs from a controllable TM and that those choices can be distilled into an automatic baseline candidate that improves a held-out target slice without unacceptable regression.

## Pipeline

```text
baseline + controllable alpha candidates
→ candidate-level IQA and hard gates
→ one safe teacher per scene or alpha-zero anchor
→ deterministic train/validation/test manifest
→ automatic alpha-policy student and/or fixed baseline checkpoint
→ independent held-out adjudication
```

## Teacher certification

A non-zero candidate is eligible only when it:

- has `KEEP` action;
- has no hard failures;
- meets minimum IQA confidence;
- has known in-domain or explicitly allowed boundary status;
- exceeds baseline by the configured score margin.

Ties choose the smallest absolute alpha. All unsafe, OOD, unknown or low-margin cases fall back to baseline and become anchors.

## Students

1. Weighted ridge alpha policy over a frozen, interpretable scene-feature vector. Training uses only the train split; OOD inference returns alpha zero.
2. Fixed `LuminanceOnlyBaseline` checkpoint distilled from selected teacher images. Gain/GTM are trainable by default, LTM is frozen, and anchors preserve original behavior.

## Acceptance

Acceptance uses held-out scenes and compares exact baseline against student output. Required gates cover target mean delta, target win rate, overall mean delta, anchor regression, and new hard failures. Teacher and final evaluator identities must be independently traceable; overlapping or unknown evaluators cannot produce a formal acceptance.

## Non-goals

- No image generation or editing model.
- No automatic deployment.
- No claim of visual superiority before real checkpoints, real Dataset V1, independent IQA and human review are run.
- No AE/AWB/HDR joint optimization.
