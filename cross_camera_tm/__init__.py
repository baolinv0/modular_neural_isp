"""Cross-camera Samsung-style tone-mapping domain adaptation."""

from .canonicalization import CanonicalizationConfig, DeviceCanonicalizer
from .contracts import CanonicalizationResult, ConfidenceSummary, LinearMetadata, canonical_tensor_sha256

# ``phase1_training.train_phase1`` was the superseded implementation that
# computed pair targets before fold isolation. Import the helper module once and
# remove that public symbol so every supported caller must use the authoritative
# ``phase1_protocol.train_phase1`` entry point.
from . import phase1_training as _phase1_training

if hasattr(_phase1_training, "train_phase1"):
    delattr(_phase1_training, "train_phase1")


__all__ = [
    "CanonicalizationConfig",
    "CanonicalizationResult",
    "ConfidenceSummary",
    "DeviceCanonicalizer",
    "LinearMetadata",
    "canonical_tensor_sha256",
]
