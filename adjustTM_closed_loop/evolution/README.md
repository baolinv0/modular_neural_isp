# TM Baseline Evolution — Phase 1

This package implements the first non-generative baseline-evolution loop:

```text
frozen baseline output
→ controllable TM alpha candidates
→ candidate-level IQA
→ one safe teacher per scene
→ automatic scene-conditioned distillation
→ held-out regression adjudication
```

The phase has two student forms:

1. **Automatic alpha policy**: a lightweight scene policy predicts one alpha and invokes the frozen controllable TM. This is the fastest way to test whether IQA-selected scene policies are automatable.
2. **Fixed baseline checkpoint**: the selected teacher images are distilled into one `LuminanceOnlyBaseline` checkpoint. Gain/GTM are trainable by default; LTM stays frozen.

No image generation or editing model is used.

## What this phase proves

A valid Phase-1 result requires all of the following on a held-out split:

- IQA-selected teachers outperform alpha-zero baseline by a configured margin;
- an automatic student reproduces enough of that improvement;
- baseline-anchor scenes do not regress beyond the configured tolerance;
- no new hard defect appears;
- final acceptance uses an evaluator independent from the teacher evaluator.

Using the same IQA for teacher selection and final validation is circular. The adjudicator therefore reports evaluator provenance and refuses a formal `ACCEPT` unless the evaluator identities are known and disjoint. Self-evaluation can only be enabled explicitly and should be treated as provisional evidence.

## Input contracts

### Controllable candidate records

The existing generator writes `inference_records.jsonl` with:

```json
{
  "scene_id": "scene.png",
  "method": "param_residual",
  "level": "a_p025",
  "alpha": 0.25,
  "output_path": "/runs/candidates/param_residual/a_p025/scene.png"
}
```

### Candidate-level IQA records

The IQA backend must emit one record per scene/method/level:

```json
{
  "scene_id": "scene.png",
  "method": "param_residual",
  "level": "a_p025",
  "overall_score": 0.84,
  "confidence": 0.93,
  "action": "KEEP",
  "hard_failures": [],
  "distribution_status": "in_domain",
  "evaluator_id": "tmqa-teacher-v1"
}
```

`REVIEW`, low-confidence, unknown-distribution, OOD, hard-failure and non-KEEP candidates cannot become pixel teachers by default.

## End-to-end commands

### 1. Generate multi-level candidates

Use the existing adjustTM benchmark generator:

```bash
python -m adjustTM.benchmark.generate_outputs \
  --manifest /data/test_manifest.json \
  --protocol /configs/protocol.json \
  --methods /configs/methods.json \
  --output-root runs/phase1/candidates \
  --dense-steps 41 \
  --device cuda
```

### 2. Score every candidate with the frozen teacher IQA

The evaluator should write `candidate_iqa.jsonl` using the contract above.

### 3. Merge generation provenance and IQA

```bash
python -m adjustTM_closed_loop.evolution merge-scores \
  --inference-records runs/phase1/candidates/inference_records.jsonl \
  --iqa-scores runs/phase1/candidate_iqa.jsonl \
  --method param_residual \
  --output runs/phase1/candidate_scores.jsonl
```

### 4. Select safe teachers and freeze scene splits

```bash
python -m adjustTM_closed_loop.evolution select-teachers \
  --scores runs/phase1/candidate_scores.jsonl \
  --input-dir /data/input_linear \
  --output-dir runs/phase1/teachers \
  --teacher-evaluator-id tmqa-teacher-v1 \
  --min-improvement 0.03 \
  --min-confidence 0.70
```

Outputs:

```text
teachers/
├── selections.jsonl
├── teacher_manifest.jsonl
└── selection_summary.json
```

Every rejected or unsafe scene becomes an alpha-zero baseline anchor rather than disappearing from training.

### 5A. Distill an automatic alpha policy

```bash
python -m adjustTM_closed_loop.evolution fit-policy \
  --teacher-manifest runs/phase1/teachers/teacher_manifest.jsonl \
  --output-policy runs/phase1/policy/alpha_policy.json \
  --output-report runs/phase1/policy/policy_report.json
```

Only `train` records are fitted. Validation and test records are reported independently. The policy uses a frozen 19-dimensional scene feature contract and falls back to alpha zero when its RMS standardized feature distance exceeds the OOD threshold.

Render exact baseline and automatic-policy images:

```bash
python -m adjustTM_closed_loop.evolution render-policy \
  --teacher-manifest runs/phase1/teachers/teacher_manifest.jsonl \
  --policy runs/phase1/policy/alpha_policy.json \
  --methods-config /configs/methods.json \
  --method-name param_residual \
  --output-dir runs/phase1/policy_render \
  --device cuda
```

### 5B. Distill one fixed baseline checkpoint

```bash
python -m adjustTM_closed_loop.evolution distill-baseline \
  --teacher-manifest runs/phase1/teachers/teacher_manifest.jsonl \
  --baseline-checkpoint /models/luminance_only_baseline.pth \
  --output-dir runs/phase1/fixed_baseline \
  --train-modules gain,gtm \
  --epochs 10 \
  --device cuda \
  --amp
```

The output `best_fixed_baseline.pth` is compatible with the original `load_baseline_checkpoint`. The trainer verifies that frozen parameters have zero drift and re-loads the saved checkpoint before reporting success.

### 6. Score baseline and student with an independent evaluator

Create two scene-level score files with the same held-out scene IDs:

```json
{"scene_id":"scene.png","score":0.78,"hard_failures":[],"evaluator_id":"tmqa-arbiter-v1"}
```

### 7. Adjudicate the evolution

```bash
python -m adjustTM_closed_loop.evolution adjudicate \
  --teacher-manifest runs/phase1/teachers/teacher_manifest.jsonl \
  --baseline-scores runs/phase1/eval/baseline_scores.jsonl \
  --student-scores runs/phase1/eval/student_scores.jsonl \
  --evaluation-splits test \
  --output runs/phase1/eval/adjudication.json
```

Formal `ACCEPT` requires:

- target mean score gain and win-rate gates;
- non-negative overall mean gain;
- bounded anchor mean regression and regression rate;
- zero new hard failures;
- independent evaluator provenance.

## Safety behavior

- Exactly one alpha-zero baseline is required per scene.
- Ties select the smallest absolute alpha.
- Improvement below the minimum margin returns baseline.
- Unknown distribution is rejected, not treated as in-distribution.
- OOD policy inputs render the exact baseline path with alpha zero.
- Train, validation and test are scene-disjoint and deterministic.
- Fixed-checkpoint training uses baseline anchors and parameter anchoring.
- The final gate defaults to the held-out `test` split.

## Verification

Core tests do not require model checkpoints:

```bash
PYTHONPATH=. pytest adjustTM_closed_loop/tests/test_evolution_*.py -q
PYTHONPATH=. python -m compileall -q adjustTM_closed_loop/evolution
```

Rendering and fixed-checkpoint training require the existing adjustTM PyTorch environment and real external checkpoints. The repository tests verify their contracts with fake runners and a reloadable tiny baseline; they do not claim real camera-quality improvement without real data and an independent evaluator.
