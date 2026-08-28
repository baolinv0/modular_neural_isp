# Qwen3.8-27B TM Semantic Judgment Branch

## Goal
Add a semantic VLM branch to the existing TM PGT IQA pipeline. Deterministic metrics remain responsible for measurable luminance facts; Qwen3.8-27B is responsible for scene intent, perceptual TM naturalness, TM-only preservation, and Top-K pairwise preference.

## Inputs
For single-candidate semantic review:
- Source image
- Candidate image
- Candidate semantic overlay (background / face / skin)
- Existing deterministic metrics, features, and guards

For pairwise ranking:
- Source image
- Candidate A
- Candidate B
- Their semantic overlays and deterministic evidence

## Outputs
Single-candidate review:
- scene_type: NORMAL | BACKLIGHT | LOW_LIGHT | SIDE_LIGHT | HIGH_DR | BRIGHT_BACKGROUND | DARK_BACKGROUND | MIXED_LIGHT
- scene_intent: face_lift_needed, background_preservation, shadow_atmosphere, highlight_priority
- naturalness severities for FACE_TOO_FLAT, FACE_OVER_LIFTED, OVER_HDR_LOOK, SHADOW_OVER_LIFTED, HIGHLIGHT_OVER_COMPRESSED, LIGHTING_CAUSALITY_BROKEN, FACE_BACKGROUND_DISCONNECTED, UNNATURAL_GLOBAL_TONE
- tm_only: PASS | SUSPICIOUS | FAIL
- semantic_quality: GOOD | ACCEPTABLE | POOR
- confidence
- summary

Pairwise review:
- preference: A_BETTER | B_BETTER | EQUIVALENT
- primary_reason
- confidence

## Decision policy
- Deterministic hard guards remain first.
- A high-confidence TM-only FAIL rejects a candidate.
- High-confidence POOR semantic quality rejects a candidate.
- Semantic judgments do not create arbitrary continuous IQA scores.
- After deterministic filtering, Top-K candidates are ranked by Qwen pairwise comparisons. Pairwise ranking affects selection, not the deterministic quality score.
- Near-tied pairwise candidates may remain in an equivalent_top_set.

## Runtime
Use the local Qwen3.8-27B OpenAI-compatible `/v1/chat/completions` endpoint. Model weights are assumed to be pre-downloaded. No model download logic is added.

## Constraints
- Functionality first; only normal input validation.
- No hashes or SHA256 logic.
- No defensive handling for implausible edge cases.
- Keep the semantic rubric compact and qualitative.
- Preserve the existing `vlm.py` API for backward compatibility.
