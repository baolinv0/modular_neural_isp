# adjustTM Benchmark Design

## Goal

Build a cache-driven benchmark that separately measures: (1) brightness controllability, (2) comparison among four learned controllers, and (3) gain over frozen/simple baselines.

## Ground-truth semantics

- Input: 16-bit linear RGB after WB and CCM, before tone mapping.
- `a_000`: real camera RGB.
- `a_m100`, `a_p100`: fixed, pre-generated Flux endpoints.
- Six nonzero intermediate levels: Retinex interpolation.

Consequently, only `a_000` supports claims about real camera reconstruction quality. Endpoint/intermediate metrics measure teacher fidelity, not absolute image quality.

## Methods

Main ranking: frozen baseline, globally calibrated exposure, globally calibrated luminance gamma, param residual, parallel adapter, FiLM, dual LoRA. Diagnostic-only: per-image exposure oracle and gamma oracle.

## Architecture

A manifest freezes test files and hashes. Inference is run once and cached. Reference metrics, dense-control metrics, VLM tasks, human-study tasks, statistics, and reports consume the same cache. Every stage records protocol hashes and fails on incompatible cache reuse.

## Metric groups

1. Real camera quality at `a_000`.
2. Flux endpoint fidelity at ±1.
3. Retinex intermediate fidelity.
4. Nine-level trajectory fidelity.
5. Dense, GT-independent control monotonicity, dead zones, jumps, smoothness, and range balance.
6. VLM target-match and naturalness as separate evidence.
7. Human target-match and naturalness as separate blinded studies.
8. Efficiency and parameter cost.

## Statistical unit

The independent unit is scene. Level values are first aggregated within scene, then scene bootstrap confidence intervals, paired deltas, win/tie/loss, worst-5% CVaR, paired permutation tests, and Holm correction are computed.

## Safety and reproducibility

The benchmark records code/checkpoint/file hashes, preserves aspect ratio in `fit_pad` mode, never uses test GT for global calibration, keeps oracle results out of main ranking, and does not form a single opaque weighted score.
