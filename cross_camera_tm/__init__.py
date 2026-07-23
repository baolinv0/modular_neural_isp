"""Cross-camera Samsung-style tone-mapping domain adaptation."""

from .canonicalization import CanonicalizationConfig, DeviceCanonicalizer
from .contracts import CanonicalizationResult, ConfidenceSummary, LinearMetadata, canonical_tensor_sha256

__all__ = [
    "CanonicalizationConfig",
    "CanonicalizationResult",
    "ConfidenceSummary",
    "DeviceCanonicalizer",
    "LinearMetadata",
    "canonical_tensor_sha256",
]
