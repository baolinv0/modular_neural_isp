# TM PGT NR-IQA Design

## Goal
Build a frozen no-reference front-camera Tone Mapping IQA subsystem that consumes a rendered candidate image plus externally provided semantic masks, computes deterministic TM-quality evidence, optionally asks a local Qwen3.8-27B vision-language model for a compact perceptual review, and classifies each candidate as CERTIFIED_PGT, USABLE_PGT, or REJECT.

## Scope
- Tone Mapping quality only.
- Semantic masks are provided by the caller; this project does not run segmentation.
- Qwen3.8-27B is pre-downloaded and served locally through an OpenAI-compatible `/v1/chat/completions` endpoint.
- No CodeAgent integration and no TM training loop.
- No hash/SHA256 logic.
- Validation is limited to inputs required for correct execution; no extreme-case defensive framework.

## Quality dimensions
1. Face Exposure Quality
2. Dynamic Range Allocation
3. Face Tone Structure
4. Face/Background Relation
5. Tone Naturalness

## Guards
1. Highlight/shadow catastrophe
2. Local face-lift/halo artifact
3. Gross skin-color abnormality
4. Candidate-pool outlier

## Scoring
For candidates that pass hard guards:

`Q = 0.20*Exposure + 0.20*DynamicRange + 0.25*FaceTone + 0.15*FaceBackground + 0.20*Naturalness`

The deterministic score is primary. Qwen is a perceptual review of failure types that are difficult to represent by scalar statistics: face flattening, over-HDR appearance, unnatural face lift, broken lighting structure, unnatural tone, and severe TM artifacts. Qwen does not produce the main continuous quality score.

## Inputs
Per candidate:
- RGB image: JPEG/PNG, display-referred sRGB.
- semantic label map: PNG with configurable integer labels. Default: 0 background, 1 face, 2 skin.

The face mask is `face | skin`; skin is the skin label; background is the background label.

## Qwen input
- candidate full image
- visualization overlay of semantic regions
- deterministic evidence serialized as compact JSON

Qwen returns compact JSON:
- `decision`: ACCEPT | REVIEW | REJECT
- `failures`: zero or more supported failure labels
- `confidence`: 0..1
- `summary`: one short sentence

## PGT classification
- REJECT when a deterministic hard guard fails or Qwen returns REJECT with high confidence.
- CERTIFIED_PGT when quality score is high and Qwen accepts (or VLM is disabled).
- USABLE_PGT for the remaining valid candidates.
- training weights: CERTIFIED=1.0, USABLE=0.5, REJECT=0.0.

Candidate-pool consistency is used only as a confidence/guard signal; it never defines the target style.

## Runtime
Offline by default. Model weights are never downloaded by evaluator code. Example server:

`vllm serve /models/Qwen3.8-27B --served-model-name qwen3.8 --host 127.0.0.1 --port 8000`

The client sends base64 data URLs for local images to the OpenAI-compatible endpoint.
