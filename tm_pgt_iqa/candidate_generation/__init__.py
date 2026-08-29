"""Deterministic, source-preserving Tone Mapping candidate generation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Keep ``python -m ...generate_candidates`` warning-free.
    from .generate_candidates import CandidateManifest, GeneratedCandidate, generate_pool, write_pool

__all__ = ["CandidateManifest", "GeneratedCandidate", "generate_pool", "write_pool"]


def __getattr__(name: str):
    if name in __all__:
        from . import generate_candidates
        return getattr(generate_candidates, name)
    raise AttributeError(name)
