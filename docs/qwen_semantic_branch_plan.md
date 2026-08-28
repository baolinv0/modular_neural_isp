# Qwen Semantic Branch Implementation Plan

**Goal:** Implement the approved Qwen3.8-27B semantic judgment branch for TM PGT IQA.

**Architecture:** Keep deterministic IQA unchanged. Add a focused semantic client that sends Source/Candidate/Overlay plus compact evidence to the local OpenAI-compatible Qwen endpoint. Extend evaluator and CLI to run single-candidate semantic review and Top-K pairwise ranking.

**Tech Stack:** Python, NumPy, Pillow, urllib, pytest, local vLLM/OpenAI-compatible API.

**Spec:** `docs/qwen_semantic_branch_design.md`

## Global Constraints
- Functionality first; avoid over-defensive code.
- No hash/SHA256 logic.
- Compact qualitative rubric only.
- Pre-downloaded local Qwen3.8-27B; no download code.

### Task 1: Semantic schemas, prompts, parser and Qwen client
- Create `tm_pgt_iqa/semantic_judge.py`.
- Add dataclasses for SceneIntent, SemanticReview, PairwiseReview.
- Build Source + Candidate + Overlay payloads.
- Build pairwise Source + A + B payloads.
- Parse compact JSON outputs.
- Add tests for payload image ordering, parser, and supported labels.

### Task 2: Evaluator integration
- Extend `CandidateResult` with `semantic`.
- Extend `evaluate_one(..., source_path=None)`.
- Reject high-confidence `tm_only=FAIL` or `semantic_quality=POOR`.
- Add `rank_candidates(results, source_path, masks_by_candidate, top_k)` pairwise tournament.
- Keep deterministic quality score unchanged.
- Add tests for semantic reject and pairwise winner/equivalent behavior.

### Task 3: CLI and config
- Add semantic configuration fields: enabled, top_k, reject_confidence, pairwise_confidence, equivalent_margin.
- Add `--sources` option. A source image is matched by scene key before candidate suffix; if source is not supplied, deterministic-only behavior remains available.
- Run pairwise ranking only when source and semantic branch are enabled.
- Include semantic ranking fields in report.

### Task 4: Verification
- Run `pytest tests/test_tm_pgt_iqa.py tests/test_qwen_semantic_branch.py -q`.
- Run `python -m compileall tm_pgt_iqa`.
- Run CLI help.
- Verify GitHub diff contains only semantic-branch related changes plus prior faceloss work.
