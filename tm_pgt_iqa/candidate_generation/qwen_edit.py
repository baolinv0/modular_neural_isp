from __future__ import annotations

from typing import Protocol
import numpy as np

from ..config import QwenEditConfig
from ..segmentation import SemanticMasks


class QwenEditAdapter(Protocol):
    """Optional image-edit service isolated from the Qwen VLM judge."""

    def generate(self, rgb: np.ndarray, masks: SemanticMasks, strength: str) -> np.ndarray: ...


def generate_qwen_edits(
    rgb: np.ndarray,
    masks: SemanticMasks,
    cfg: QwenEditConfig,
    adapter: QwenEditAdapter | None = None,
) -> dict[str, np.ndarray]:
    """Return no candidates unless an explicitly configured edit adapter exists."""
    if not cfg.enabled or adapter is None:
        return {}
    return {
        "qwen_normal": np.asarray(adapter.generate(rgb, masks, "normal"), dtype=np.float32),
        "qwen_strong": np.asarray(adapter.generate(rgb, masks, "strong"), dtype=np.float32),
    }
