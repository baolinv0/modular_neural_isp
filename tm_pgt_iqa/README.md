# TM PGT IQA V2

V2 generates and selects front-camera portrait Tone-Mapping pseudo-GT. It evaluates exposure, tone mapping, dynamic range, local face exposure, and the face/background tone relation. It does not score focus, texture, noise, resolution, bokeh, or beautification. Source-preservation checks reject clear content edits outside the Tone-Mapping scope.

The production chain is:

`Source + semantic masks -> candidate pool -> objective guards -> family-balanced Top-K -> optional Qwen semantic judgment/tournament -> PGT`.

## Generate candidates

Label maps use `0=background`, `1=face/non-skin`, `2=skin`, `3=human/body`. Soft face/skin/human maps are optional. The generator writes one directory per source image, with a PNG and authoritative JSON manifest for every candidate.

```bash
python -m tm_pgt_iqa.candidate_generation.generate_candidates \
  --input source/ \
  --masks masks/ \
  --output candidates/ \
  --config tm_pgt_iqa.example.yaml
```

Deterministic generation creates eight Retinex candidates, three local-face lifts, two face-tone shapes, and one gain baseline. The optional Qwen image-edit adapter may append `qwen_normal` and `qwen_strong`; its absence never blocks the deterministic pool.

## Batch evaluation and selection

Use `--candidates` in production. Its JSON manifests are the authority for candidate ID, family, parameters, and source; the selector does not infer a family from filenames.

```bash
python -m tm_pgt_iqa \
  --source source/ \
  --candidates candidates/ \
  --masks masks/ \
  --config tm_pgt_iqa.example.yaml \
  --output results/
```

The source argument can be one image or a directory. Candidate pools may be flat for one source or organized as `candidates/<source-stem>/`. Source masks are resolved from `<stem>.png`, `<stem>_mask.png`, or the exact source filename; use `--source-mask` to specify a file or mask directory explicitly.

For deterministic debugging with no local VLM service:

```bash
python -m tm_pgt_iqa \
  --source source/ --candidates candidates/ --masks masks/ \
  --config tm_pgt_iqa.example.yaml --output results/ --no-vlm
```

The output root contains `report.json`, `summary.csv`, selected images, copied candidates/manifests, per-scene semantic JSON, and candidate/ranking/failure contact sheets. `pgt_class` and `selection_confidence` are independent: CERTIFIED images retain training weight `1.0` even when the winner is weakly separated from another valid candidate.

`--images` remains a compatibility alias for the former single-pool interface. It accepts legacy minimal sidecars, but does not provide the V2 production manifest guarantees.

## Qwen semantic branch

The optional local Qwen3.8-27B service receives source, candidate, semantic overlay, and objective evidence. It runs scene understanding once per source, then naturalness and TM-only review on the deterministic family-balanced Top-K, followed by a pairwise tournament. Qwen provides semantic labels and pairwise preferences, never an arbitrary 0–100 quality score.

```bash
vllm serve /models/Qwen3.8-27B --served-model-name qwen3.8 --host 127.0.0.1 --port 8000
```

**Live Qwen Conformance: NOT COMPLETE.** Local parser and offline tests do not validate JSON stability, pairwise direction, repeatability, or non-TM-edit detection on the deployment server. Run the required 20–30 real-scene conformance set before treating Qwen output as production-validated.

## Human validation and ablations

Human annotations are JSON scene records with `source`, ranked candidate IDs in `ranking`, and optional boolean `accepted` for certified outputs. Calculate the required metrics:

```bash
python -m tm_pgt_iqa.validation \
  --report results/report.json --annotations human_rankings.json \
  --output validation.json
python -m tm_pgt_iqa.validation --write-ablation-template ablations.json
```

The validation report contains mean Kendall tau, Top-2 accuracy, and Certified Precision. The emitted ablation template records the mandatory Objective/Qwen and candidate-family experiment matrix; collecting real human annotations and running those experiments remains an offline validation task.
