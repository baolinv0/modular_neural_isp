# adjustTM Closed Loop V1

`adjustTM_closed_loop` is an isolated evaluation-driven development layer built on top of the existing `adjustTM/` experiment. It does not modify the frozen baseline, the four control methods, the precision benchmark, or the original 59-test suite.

The V1 loop is intentionally narrow:

```text
Dataset V1 gate
→ param_residual training handoff
→ Qwen-TMQA evaluation
→ F1–F8 failure taxonomy
→ training-data prescription
→ target-slice and regression gates
```

## Directory responsibilities

- `dataset_gate.py`: validates the 16-bit linear input and complete nine-level GT trajectory; classifies each scene as `clean`, `boundary`, or `invalid`.
- `qwen_tmqa_adapter.py`: invokes the independently maintained Qwen-TMQA CLI and normalizes its scene reports.
- `failure_taxonomy.py`: maps Qwen-TMQA failure reasons into the frozen TM V1 taxonomy F1–F8.
- `data_prescription.py`: converts repeated failures into module-targeted data tasks with positives, boundary cases, hard negatives, regression anchors, supervision, and acceptance gates.
- `runner.py`: builds the frozen-scope `param_residual` training command and writes a reproducible handoff manifest.
- `bootstrap_qwen_tmqa.py`: extracts and installs the verified Qwen-TMQA source archive from the separate `baolinv0/IQA` project.

## Why Qwen-TMQA remains separate

The evaluator is treated as an independently versioned backend rather than copied into the TM model package. This prevents evaluator upgrades from silently changing model code and makes it possible to freeze one evaluator version while training a candidate TM model.

## 1. Install Qwen-TMQA

Copy or download `artifacts/qwen_tmqa_initial_system.zip` from the `baolinv0/IQA` repository, then run:

```bash
python -m adjustTM_closed_loop.bootstrap_qwen_tmqa \
  --archive /path/to/qwen_tmqa_initial_system.zip \
  --destination adjustTM_closed_loop/vendor
```

This installs the extracted package in editable mode and exposes the `qwen-tmqa` CLI.

## 2. Prepare Dataset V1 and the first closed-loop handoff

```bash
python -m adjustTM_closed_loop prepare \
  --input-dir /data/input_linear \
  --gt-root /data/gt_levels \
  --baseline-checkpoint /models/luminance_only_baseline.pth \
  --output-dir runs/closed_loop_v1 \
  --qwen-config adjustTM_closed_loop/configs/qwen_tmqa.yaml \
  --batch-size 18 \
  --epochs 30 \
  --seed 42 \
  --amp
```

Generated files:

```text
runs/closed_loop_v1/
├── dataset_gate.json
├── qwen_tmqa_runtime.yaml
├── closed_loop_manifest.json
└── commands.sh
```

The generated Qwen-TMQA runtime config points `dataset.source_dir` to the 16-bit linear input directory, so source fidelity is evaluated rather than only RGB sequence consistency.

## 3. Enable Qwen3-VL and an independent arbiter

```bash
python -m adjustTM_closed_loop prepare \
  ... \
  --primary-url http://127.0.0.1:8000/v1 \
  --primary-model Qwen/Qwen3-VL-8B-Instruct \
  --arbiter-url http://127.0.0.1:8001/v1 \
  --arbiter-model OpenGVLab/InternVL3_5-38B
```

Without these flags, Qwen-TMQA still runs its deterministic objective, fixed-region, source-fidelity, and control-trajectory checks.

## 4. Train only the first baseline method

V1 deliberately generates a command for:

```text
control_method = param_residual
```

It preserves the original adjustTM constraints: the baseline Gain/GTM/LTM weights remain frozen, the control batch size is a positive multiple of 18, and `alpha=0` must reproduce the baseline.

Run the generated command in `commands.sh` after the real checkpoint smoke gate passes.

## 5. Evaluate generated nine-level outputs

The Qwen-TMQA dataset root must contain:

```text
a_m100/ ... a_000 ... a_p100/
```

Use the existing `adjustTM.benchmark.generate_outputs` flow to create this layout for the trained `param_residual` checkpoint, then run the corresponding `qwen-tmqa evaluate` command with the same source directory and frozen evaluator config.

## 6. Generate a training-data prescription

```bash
python -m adjustTM_closed_loop prescribe \
  --qwen-output runs/closed_loop_v1/qwen_tmqa_model \
  --output runs/closed_loop_v1/data_prescription.json
```

The prescription does not merely say “add more difficult images.” It specifies:

- target failure mechanism and module owner;
- positive failure scenes;
- boundary/REVIEW scenes;
- hard-negative `REGENERATE` or `REJECT` scenes;
- KEEP scenes used as regression anchors;
- required supervision;
- acceptance gates for the next training round.

## Frozen V1 failure taxonomy

| Code | Failure |
|---|---|
| F1 | brightness insufficient |
| F2 | brightness excessive |
| F3 | highlight clipping |
| F4 | shadow crushing |
| F5 | chroma drift |
| F6 | non-monotonic, discontinuous, dead-zone, or saturated control curve |
| F7 | regional/semantic inconsistency |
| F8 | halo, banding, alignment, edge, or structural artifact |

## Verification

```bash
PYTHONPATH=. python -m compileall -q adjustTM_closed_loop
PYTHONPATH=. pytest adjustTM_closed_loop/tests -q
```

V1 is a handoff and diagnosis layer. It does not automatically execute training, update model weights, or accept an IQA-driven model change without a held-out regression gate.
