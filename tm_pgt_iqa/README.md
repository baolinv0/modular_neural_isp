# TM PGT NR-IQA

No-reference front-camera portrait Tone Mapping pseudo-GT selector.

## Inputs

Each candidate image must have a semantic label map with the same stem in the mask directory.
Default labels:

- `0`: background
- `1`: face (non-skin face region)
- `2`: skin

The evaluator does not run a segmentation model.

## Qwen3.8-27B

Model weights are expected to be downloaded before runtime. The evaluator never downloads model files.
Start a local OpenAI-compatible server, for example:

```bash
vllm serve /models/Qwen3.8-27B \
  --served-model-name qwen3.8 \
  --host 127.0.0.1 \
  --port 8000
```

Use a vLLM version with Qwen3.8 multimodal support.

## Run

```bash
python -m tm_pgt_iqa \
  --images ./candidates \
  --masks ./semantic_masks \
  --config tm_pgt_iqa.example.yaml \
  --output ./pgt_report.json
```

For deterministic metrics only:

```bash
python -m tm_pgt_iqa \
  --images ./candidates \
  --masks ./semantic_masks \
  --output ./pgt_report.json \
  --no-vlm
```

## Output

Each image contains five deterministic TM scores:

- exposure
- dynamic range
- face tone
- face/background relation
- tone naturalness

Hard guards cover highlight/shadow catastrophe, local face-lift halo, gross skin-color abnormality, and candidate-pool outliers. Qwen reviews only perceptual TM failures that scalar statistics do not capture well.

Classification:

- `CERTIFIED_PGT`: training weight 1.0
- `USABLE_PGT`: training weight 0.5
- `REJECT`: training weight 0.0

The numeric defaults are initial engineering priors and should be calibrated later on accepted/rejected portrait Tone Mapping samples from the target product domain.
