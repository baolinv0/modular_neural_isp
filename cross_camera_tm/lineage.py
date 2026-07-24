from __future__ import annotations

import re
from dataclasses import dataclass


def _valid_hash(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


@dataclass(frozen=True)
class LineageNode:
    artifact_sha256: str
    kind: str
    parent_sha256s: tuple[str, ...]
    transformations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _valid_hash(self.artifact_sha256) or any(
            not _valid_hash(parent) for parent in self.parent_sha256s
        ):
            raise ValueError("lineage nodes require SHA-256 artifact identifiers")
        if not self.kind or not self.transformations:
            raise ValueError("lineage kind and transformations are required")


class LineageStore:
    def __init__(self):
        self._nodes: dict[str, LineageNode] = {}

    def add(self, node: LineageNode) -> None:
        if node.artifact_sha256 in self._nodes:
            raise ValueError("duplicate lineage artifact")
        missing = [parent for parent in node.parent_sha256s if parent not in self._nodes]
        if missing:
            raise ValueError("lineage parent is missing: " + ",".join(missing))
        self._nodes[node.artifact_sha256] = node

    def ancestors(self, artifact_sha256: str) -> tuple[str, ...]:
        if artifact_sha256 not in self._nodes:
            raise KeyError(artifact_sha256)
        ordered: list[str] = []

        def visit(current: str) -> None:
            for parent in self._nodes[current].parent_sha256s:
                if parent not in ordered:
                    ordered.append(parent)
                    visit(parent)

        visit(artifact_sha256)
        return tuple(ordered)

    def get(self, artifact_sha256: str) -> LineageNode:
        return self._nodes[artifact_sha256]
