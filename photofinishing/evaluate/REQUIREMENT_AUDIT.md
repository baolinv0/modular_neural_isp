# Final Requirement Audit

Scope audited against `IMPLEMENTATION_PLAN.md` and the user-requested Stage-1/Stage-2 evaluation objectives.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Input multiple pretrained, Stage-1, and Stage-2 model groups | PASS | `config.py` supports multiple groups and multiple named Stage-2 variants; duplicate labels fail closed. |
| 2 | Evaluate same-scene, non-pixel-aligned reference images | PASS | `data.py` loads independent target/reference images; `metrics.py` uses distribution/quantile metrics and never creates pixel correspondence. |
| 3 | Determine whether Stage 1 improves brightness and tone | PASS | Absolute/signed EV, luminance quantiles, exposure-normalized tone shape, log-luma W1, shadow/highlight and clipping metrics. |
| 4 | Determine whether Stage 2 improves color independently of Stage 1 | PASS | Stage-2 evidence is `Stage1 distance - Stage2 distance`; primary metric is luminance-conditioned CbCr SWD with luminance regression gates. |
| 5 | Compare pretrained -> Stage1 and Stage1 -> Stage2 increments | PASS | `incremental_improvements.csv` and paired transition summaries. Signed EV direction is diagnostic and excluded from lower-is-better improvement statistics. |
| 6 | Measure whether Stage 2 is necessary | PASS | Optional repeated reference establishes a capture/non-alignment noise floor; without it necessity is explicitly `undetermined`. |
| 7 | Handle common FoV, dynamic regions, and semantic composition | PASS | Independent valid/ignore masks; paired skin/sky/vegetation masks; semantic Lab SWD and semantic area-gap diagnostics. Automatic segmentation is intentionally out of scope. |
| 8 | Report mean, median, win rate, negative rate, p10, and bootstrap CI | PASS | `statistics.py` implements deterministic paired bootstrap summaries. |
| 9 | Decide Stage-1 effectiveness, Stage-2 necessity/effectiveness, and Stage-2 capacity | PASS | `decision.py` applies evidence, auxiliary-color, luminance-safety, and negative-tail gates. |
| 10 | Decide whether to expand the data set | PASS | Separate evaluation-data and training-data recommendations use sample count, inconclusive CI, cross-group sign stability, noise-floor evidence, and failure slices. |
| 11 | Produce observable and machine-readable outputs | PASS | Per-scene CSVs, summary CSV/JSON, decisions JSON, Markdown report, scene-tag slices, visual panels, and optional individual PNG outputs. |
| 12 | Preserve existing code and affine provenance rules | PASS | Existing evaluator/training files remain usable; new loader reuses strict affine reconstruction/SHA validation; old tests and real model canaries pass. |

## Verification Boundary

The implementation is mechanically complete and tested. It has not been run on the user's private real checkpoint/image paths in this environment. Real-data scientific conclusions are produced only after the user supplies valid local paths and executes the documented command.
