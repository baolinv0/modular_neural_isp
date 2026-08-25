# Front Portrait IQA Evaluator

Target-conditioned semantic portrait rendering evaluator for CodeAgent experiments.

This checkpoint contains the frozen evaluator foundation already implemented for:

- Candidate / Baseline / Reference manifest parsing
- configurable Brightness / Color / Tone policy and gate defaults
- image decoding, linear luminance, Lab conversion, CIEDE2000 helpers
- main-face ROI detection
- semantic face / skin / face-ring / background masks
- non-pixel-aligned scene validation using normalized face position and scale

The remaining scoring/VLM/aggregation/decision/CLI layers are intentionally not represented as complete in this checkpoint. They should be added on this `FrontPortrait` branch before the evaluator is used as the final CodeAgent decision boundary.

Recommended import setup from repository root:

```bash
export PYTHONPATH="$PWD/evaluation:$PYTHONPATH"
```

Dependencies used by this foundation: `numpy`, `PyYAML`, and `opencv-python` (or `opencv-python-headless`).
