from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .certification import CRITICAL_GATE_NAMES


def _valid_hash(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


@dataclass(frozen=True)
class ManifestRecord:
    artifact_sha256: str
    input_sha256: str
    parent_sha256s: tuple[str, ...]
    model_sha256: str
    config_sha256: str
    profile_sha256: str
    transformations: tuple[str, ...]
    gates: tuple[dict[str, Any], ...]
    supervision_type: str
    synthetic: bool
    real_model: bool
    raw_generated: bool
    projected: bool
    fully_certified: bool
    route_reasons: tuple[str, ...]

    _FIELDS = frozenset(
        {
            "artifact_sha256",
            "input_sha256",
            "parent_sha256s",
            "model_sha256",
            "config_sha256",
            "profile_sha256",
            "transformations",
            "gates",
            "supervision_type",
            "synthetic",
            "real_model",
            "raw_generated",
            "projected",
            "fully_certified",
            "route_reasons",
        }
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ManifestRecord":
        unknown = sorted(set(payload) - cls._FIELDS)
        missing = sorted(cls._FIELDS - set(payload))
        if unknown:
            raise ValueError("unknown manifest fields: " + ",".join(unknown))
        if missing:
            raise ValueError("missing manifest fields: " + ",".join(missing))
        return cls(
            artifact_sha256=str(payload["artifact_sha256"]),
            input_sha256=str(payload["input_sha256"]),
            parent_sha256s=tuple(str(value) for value in payload["parent_sha256s"]),
            model_sha256=str(payload["model_sha256"]),
            config_sha256=str(payload["config_sha256"]),
            profile_sha256=str(payload["profile_sha256"]),
            transformations=tuple(str(value) for value in payload["transformations"]),
            gates=tuple(dict(value) for value in payload["gates"]),
            supervision_type=str(payload["supervision_type"]),
            synthetic=bool(payload["synthetic"]),
            real_model=bool(payload["real_model"]),
            raw_generated=bool(payload["raw_generated"]),
            projected=bool(payload["projected"]),
            fully_certified=bool(payload["fully_certified"]),
            route_reasons=tuple(str(value) for value in payload["route_reasons"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ManifestWriter:
    def __init__(self, path: Path):
        self.path = Path(path)

    @staticmethod
    def validate(record: ManifestRecord) -> None:
        hashes = (
            record.artifact_sha256,
            record.input_sha256,
            record.model_sha256,
            record.config_sha256,
            record.profile_sha256,
            *record.parent_sha256s,
        )
        if not all(_valid_hash(value) for value in hashes):
            raise ValueError("manifest requires exact SHA-256 identifiers")
        if record.supervision_type not in {"diagnostic", "reject", "parameter", "range", "preference", "pixel"}:
            raise ValueError("unknown supervision_type")
        if record.raw_generated and record.supervision_type == "pixel":
            raise ValueError("raw generated images can never be pixel targets")
        if record.supervision_type == "pixel" and (not record.projected or not record.fully_certified):
            raise ValueError("pixel targets require projection and full recertification")
        if record.fully_certified:
            gate_names = tuple(gate.get("name") for gate in record.gates)
            if gate_names != CRITICAL_GATE_NAMES:
                raise ValueError("fully certified manifest must contain every critical gate in order")
        if not record.transformations or not record.route_reasons:
            raise ValueError("manifest requires transformations and route reasons")

    def write(self, record: ManifestRecord) -> None:
        self.validate(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
