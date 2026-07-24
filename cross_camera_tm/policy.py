from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import FailureType, GateStatus


@dataclass(frozen=True)
class LocalModelSpec:
    model_id: str
    license: str
    checkpoint_path: str
    remote_api: bool


@dataclass(frozen=True)
class PolicyDecision:
    status: GateStatus
    reason: str


class OpenSourceLocalPolicy:
    ALLOWED_LICENSES = frozenset(
        {"apache-2.0", "apache 2.0", "mit", "bsd-2-clause", "bsd-3-clause"}
    )

    def validate(self, spec: LocalModelSpec) -> PolicyDecision:
        if spec.remote_api:
            return PolicyDecision(GateStatus.FAIL, "remote_api_forbidden")
        if spec.license.strip().lower() not in self.ALLOWED_LICENSES:
            return PolicyDecision(GateStatus.FAIL, "license_not_allowlisted")
        path = Path(spec.checkpoint_path)
        if not path.is_file():
            return PolicyDecision(GateStatus.UNAVAILABLE, "local_checkpoint_unavailable")
        return PolicyDecision(GateStatus.PASS, "open_source_local_checkpoint")

    def require(self, spec: LocalModelSpec) -> None:
        decision = self.validate(spec)
        if decision.status is not GateStatus.PASS:
            raise RuntimeError(decision.reason)


@dataclass(frozen=True)
class DiagnosisResult:
    failure_type: FailureType
    roi_hint: tuple[int, int, int, int] | None
    highlight_risk: bool
    synthetic_mock: bool
    model_id: str


class Qwen3VLDiagnosisAdapter:
    """Local-only semantic diagnosis; deliberately exposes no numerical TM strength."""

    def __init__(
        self,
        infer: Callable[[torch.Tensor], Mapping[str, Any]],
        *,
        model_id: str,
        synthetic_mock: bool,
    ):
        self._infer = infer
        self.model_id = model_id
        self.synthetic_mock = synthetic_mock

    @classmethod
    def from_local(
        cls,
        infer: Callable[[torch.Tensor], Mapping[str, Any]],
        spec: LocalModelSpec,
        policy: OpenSourceLocalPolicy | None = None,
    ) -> "Qwen3VLDiagnosisAdapter":
        (policy or OpenSourceLocalPolicy()).require(spec)
        return cls(infer, model_id=spec.model_id, synthetic_mock=False)

    @classmethod
    def deterministic_mock(
        cls, infer: Callable[[torch.Tensor], Mapping[str, Any]]
    ) -> "Qwen3VLDiagnosisAdapter":
        return cls(infer, model_id="deterministic-qwen3-vl-interface-double", synthetic_mock=True)

    def diagnose(self, image: torch.Tensor) -> DiagnosisResult:
        payload = self._infer(image)
        forbidden = {"ev", "strength", "gain", "correction"}.intersection(payload)
        if forbidden:
            raise ValueError("diagnosis adapter cannot provide numerical correction strength")
        failure_type = FailureType(str(payload["failure_type"]))
        roi = payload.get("roi_hint")
        if roi is not None:
            if not isinstance(roi, Sequence) or len(roi) != 4:
                raise ValueError("roi_hint must be [x0,y0,x1,y1]")
            roi = tuple(int(item) for item in roi)
        return DiagnosisResult(
            failure_type=failure_type,
            roi_hint=roi,  # type: ignore[arg-type]
            highlight_risk=bool(payload.get("highlight_risk", False)),
            synthetic_mock=self.synthetic_mock,
            model_id=self.model_id,
        )


@dataclass(frozen=True)
class EditProposal:
    image: torch.Tensor
    prompt: str
    synthetic_mock: bool
    model_id: str


class QwenImageEditLocalAdapter:
    def __init__(
        self,
        infer: Callable[[torch.Tensor, str], torch.Tensor],
        *,
        model_id: str,
        synthetic_mock: bool,
    ):
        self._infer = infer
        self.model_id = model_id
        self.synthetic_mock = synthetic_mock

    @classmethod
    def from_local(
        cls,
        infer: Callable[[torch.Tensor, str], torch.Tensor],
        spec: LocalModelSpec,
        policy: OpenSourceLocalPolicy | None = None,
    ) -> "QwenImageEditLocalAdapter":
        (policy or OpenSourceLocalPolicy()).require(spec)
        return cls(infer, model_id=spec.model_id, synthetic_mock=False)

    @classmethod
    def deterministic_mock(
        cls, infer: Callable[[torch.Tensor, str], torch.Tensor]
    ) -> "QwenImageEditLocalAdapter":
        return cls(infer, model_id="deterministic-qwen-image-edit-interface-double", synthetic_mock=True)

    def edit(self, image: torch.Tensor, prompt: str) -> EditProposal:
        output = self._infer(image.detach().clone(), prompt)
        if not torch.is_tensor(output) or output.shape != image.shape:
            raise ValueError("local image editor must return the input tensor shape")
        if not torch.isfinite(output).all():
            raise ValueError("local image editor returned non-finite pixels")
        return EditProposal(output.clamp(0.0, 1.0), prompt, self.synthetic_mock, self.model_id)


@dataclass(frozen=True)
class AnonymousABResult:
    available: bool
    preference: str
    order_consistent: bool
    synthetic_mock: bool
    model_id: str


class InternVLAnonymousABAdapter:
    """Anonymous AB/BA interface. Its result is evidence, never an acceptance decision."""

    def __init__(self, compare: Callable[[torch.Tensor, torch.Tensor], str], *, model_id: str, synthetic_mock: bool):
        self._compare = compare
        self.model_id = model_id
        self.synthetic_mock = synthetic_mock

    @classmethod
    def deterministic_mock(
        cls, compare: Callable[[torch.Tensor, torch.Tensor], str]
    ) -> "InternVLAnonymousABAdapter":
        return cls(compare, model_id="deterministic-internvl-interface-double", synthetic_mock=True)

    @classmethod
    def from_local(
        cls,
        compare: Callable[[torch.Tensor, torch.Tensor], str],
        spec: LocalModelSpec,
        policy: OpenSourceLocalPolicy | None = None,
    ) -> "InternVLAnonymousABAdapter":
        (policy or OpenSourceLocalPolicy()).require(spec)
        return cls(compare, model_id=spec.model_id, synthetic_mock=False)

    def arbitrate(self, projected: torch.Tensor, baseline: torch.Tensor) -> AnonymousABResult:
        first = self._compare(projected, baseline).strip().upper()
        second_raw = self._compare(baseline, projected).strip().upper()
        if first not in {"A", "B", "TIE"} or second_raw not in {"A", "B", "TIE"}:
            return AnonymousABResult(False, "UNAVAILABLE", False, self.synthetic_mock, self.model_id)
        second = "B" if second_raw == "A" else "A" if second_raw == "B" else "TIE"
        consistent = first == second
        preference = "PROJECTED" if first == "A" and consistent else "BASELINE" if first == "B" and consistent else "TIE"
        return AnonymousABResult(True, preference, consistent, self.synthetic_mock, self.model_id)


@dataclass(frozen=True)
class OvisInspectionResult:
    available: bool
    hard_defect: bool
    reason: str
    synthetic_mock: bool
    model_id: str


class OvisLocalInspector:
    """Optional independent local inspection interface; unavailability remains explicit."""

    def __init__(
        self,
        inspect: Callable[[torch.Tensor], Mapping[str, Any]],
        *,
        model_id: str,
        synthetic_mock: bool,
    ):
        self._inspect = inspect
        self.model_id = model_id
        self.synthetic_mock = synthetic_mock

    @classmethod
    def from_local(
        cls,
        inspect: Callable[[torch.Tensor], Mapping[str, Any]],
        spec: LocalModelSpec,
        policy: OpenSourceLocalPolicy | None = None,
    ) -> "OvisLocalInspector":
        (policy or OpenSourceLocalPolicy()).require(spec)
        return cls(inspect, model_id=spec.model_id, synthetic_mock=False)

    @classmethod
    def deterministic_mock(
        cls, inspect: Callable[[torch.Tensor], Mapping[str, Any]]
    ) -> "OvisLocalInspector":
        return cls(inspect, model_id="deterministic-ovis-interface-double", synthetic_mock=True)

    def inspect(self, image: torch.Tensor) -> OvisInspectionResult:
        payload = self._inspect(image)
        return OvisInspectionResult(
            available=bool(payload.get("available", True)),
            hard_defect=bool(payload.get("hard_defect", False)),
            reason=str(payload.get("reason", "ok")),
            synthetic_mock=self.synthetic_mock,
            model_id=self.model_id,
        )
