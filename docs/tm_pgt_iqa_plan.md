# TM PGT NR-IQA Implementation Plan

## Goal
Implement a complete, offline-first no-reference Tone Mapping pseudo-GT selector using deterministic semantic-region metrics plus optional local Qwen3.8-27B visual review.

## Files
- `tm_pgt_iqa/__init__.py`: public API
- `tm_pgt_iqa/config.py`: compact configuration dataclasses
- `tm_pgt_iqa/segmentation.py`: load label map and build face/skin/background masks
- `tm_pgt_iqa/metrics.py`: five quality dimensions and four guards
- `tm_pgt_iqa/vlm.py`: local OpenAI-compatible Qwen client, prompt, JSON parser
- `tm_pgt_iqa/evaluator.py`: per-image evaluation and candidate-pool classification
- `tm_pgt_iqa/cli.py`: batch CLI and JSON report
- `tm_pgt_iqa/__main__.py`: module entry point
- `tm_pgt_iqa.example.yaml`: example offline configuration
- `tests/test_tm_pgt_iqa.py`: deterministic metrics, guards, segmentation, scoring, pool ranking, VLM parser/client payload tests

## Implementation order
1. Write tests for mask loading and identity metric behavior.
2. Implement semantic-mask loading and luminance/statistics helpers.
3. Write tests for exposure, DR, face-tone, face/background, naturalness, halo/color guards.
4. Implement deterministic metric engine.
5. Write tests for VLM JSON parsing and request payload construction without a live server.
6. Implement Qwen3.8-27B OpenAI-compatible client.
7. Write tests for PGT classification and candidate-pool outlier logic.
8. Implement evaluator and CLI.
9. Run `pytest -q tests/test_tm_pgt_iqa.py` and `python -m compileall tm_pgt_iqa`.

## Constraints
- Functional implementation first; no excessive defensive framework.
- No hash/SHA256.
- No segmentation model; caller supplies semantic label map.
- No CodeAgent interface.
- Main continuous score is deterministic; Qwen only reviews perceptual TM failures.
- Model path/served model name/base URL are configuration only; evaluator never downloads model weights.
