from __future__ import annotations

LEVELS: tuple[tuple[str, float], ...] = (
    ("a_m100", -1.00),
    ("a_m075", -0.75),
    ("a_m050", -0.50),
    ("a_m025", -0.25),
    ("a_000", 0.00),
    ("a_p025", 0.25),
    ("a_p050", 0.50),
    ("a_p075", 0.75),
    ("a_p100", 1.00),
)
LEVEL_TO_ALPHA = dict(LEVELS)
ALPHA_TO_LEVEL = {alpha: name for name, alpha in LEVELS}
CONTROL_METHODS = ("param_residual", "parallel_adapter", "film", "dual_lora")

# Defaults chosen so all four methods are close in trainable-parameter count
# for the repository's default Gain/GTM feature widths (20 and 40).
PARAM_RESIDUAL_HIDDEN = 8
PARALLEL_ADAPTER_BOTTLENECK = 4
FILM_HIDDEN = 4
DUAL_LORA_RANK = 8
DEFAULT_TARGET_CONTROL_PARAMS = 1040
DEFAULT_PARAMETER_TOLERANCE = 0.10
