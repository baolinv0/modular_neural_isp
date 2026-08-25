"""No-reference Tone Mapping pseudo-GT IQA.

The package scores front-camera portrait Tone Mapping candidates using
semantic-region statistics and an optional local Qwen3.8-27B visual judge.
"""

from .config import IQAConfig, load_config
from .evaluator import TMPGTEvaluator

__all__ = ["IQAConfig", "TMPGTEvaluator", "load_config"]
