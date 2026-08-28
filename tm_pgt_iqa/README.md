# TM PGT NR-IQA

No-reference front-camera portrait Tone Mapping pseudo-GT selector with deterministic TM metrics and an optional Qwen3.8-27B semantic judgment branch.

## Inputs

Each candidate image must have a semantic label map with the same stem in the mask directory.
Default labels:

- `0`: background
- `1`: face (non-skin face region)
- `2`: skin

The evaluator consumes segmentation results; it does not run a segmentation model.

For the semantic branch, also provide the original Source image before candidate Tone Mapping. Source is used to understand scene lighting and to verify that Source -> Candidate remains a plausible Tone Mapping operation.

## Qwen3.8-27B

Model weights are expected to be downloaded before runtime. The evaluator never downloads model files.
Start a local OpenAI-compatible server, for example:

```bash
vllm serve /models/Qwen3.8-27B \
  --served-model-name qwen3.8 \
  --host 127.0.0.1 \
  --port 8000
```

The semantic client uses `/v1/chat/completions` with local `image_url` data URLs.

## Semantic judgment branch

Qwen receives:

1. Source image
2. Candidate image
3. Candidate semantic overlay
4. Deterministic TM evidence

It returns a compact structured judgment:

- scene intent: normal/backlight/low-light/side-light/high-DR etc.
- TM naturalness: flat face, over-lift, over-HDR, shadow lift, highlight compression, lighting-causality break, face/background disconnect, global tone unnaturalness
- TM-only: `PASS | SUSPICIOUS | FAIL`
- semantic quality: `GOOD | ACCEPTABLE | POOR`

High-confidence `TM-only=FAIL` or `semantic_quality=POOR` can reject a candidate. Qwen does not modify the deterministic Overall score.

After deterministic eligibility filtering, the top-K candidates are compared pairwise by Qwen:

```text
A_BETTER | B_BETTER | EQUIVALENT
```

The tournament produces a semantic winner and an `equivalent_top_set`. This is used for final PGT selection rather than creating an arbitrary Qwen 0-100 score.

## Run with semantic branch

```bash
python -m tm_pgt_iqa \
  --images ./candidates \
  --masks ./semantic_masks \
  --source ./source.jpg \
  --config tm_pgt_iqa.example.yaml \
  --output ./pgt_report.json
```

## Deterministic metrics only

```bash
python -m tm_pgt_iqa \
  --images ./candidates \
  --masks ./semantic_masks \
  --output ./pgt_report.json \
  --no-vlm
```

If `--source` is omitted but VLM is enabled, the legacy candidate-only VLM review remains available for backward compatibility.

## Output

Each image contains five deterministic TM scores:

- exposure
- dynamic range
- face tone
- face/background relation
- tone naturalness

The report additionally contains `semantic_ranking` when Source-aware semantic ranking is enabled.

Classification:

- `CERTIFIED_PGT`: training weight 1.0
- `USABLE_PGT`: training weight 0.5
- `REJECT`: training weight 0.0

The numeric defaults are initial engineering priors and should be calibrated on accepted/rejected portrait Tone Mapping samples from the target product domain.
