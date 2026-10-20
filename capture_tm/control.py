"""Physical-action constraints and an auditable simulated delayed-control ledger.

This controller calls the capture-domain simulator. It does not drive hardware
or fabricate device execution metadata. Anti-flicker is only a duration rule for
an ideal light source modulated at twice the configured mains frequency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch

from .types import CaptureAction, CaptureResult, Scene, SensorProfile


def _action_dict(action: CaptureAction) -> dict:
    return {name: float(getattr(action, name)) for name in ("exposure_s", "analog_gain", "digital_gain")}


@dataclass(frozen=True)
class ExposureConstraints:
    min_exposure_s: float = 1 / 2000
    max_exposure_s: float = 1 / 30
    min_analog_gain: float = 1.
    max_analog_gain: float = 128.
    max_total_capture_s: float = .1
    readout_s: float = 0.
    control_delay_frames: int = 0
    anti_flicker_hz: float | None = None

    def __post_init__(self):
        for name in ("min_exposure_s", "max_exposure_s", "min_analog_gain", "max_analog_gain", "max_total_capture_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_exposure_s > self.max_exposure_s or self.min_analog_gain > self.max_analog_gain:
            raise ValueError("constraint minimum cannot exceed maximum")
        if not math.isfinite(self.readout_s) or self.readout_s < 0:
            raise ValueError("readout_s must be nonnegative and finite")
        if isinstance(self.control_delay_frames, bool) or not isinstance(self.control_delay_frames, int) or self.control_delay_frames < 0:
            raise ValueError("control_delay_frames must be a nonnegative integer")
        if self.anti_flicker_hz is not None and (not math.isfinite(self.anti_flicker_hz) or self.anti_flicker_hz <= 0):
            raise ValueError("anti_flicker_hz must be positive and finite")

    def feasible_mask(self, actions: Sequence[CaptureAction], *, remaining_capture_s=None) -> torch.Tensor:
        """Return a CPU Bool[K] mask without modifying actions or any budget.

        ``anti_flicker_hz`` explicitly denotes mains frequency (e.g. 50/60 Hz).
        Only positive integer multiples of 1/(2*f) are legal under that option;
        it does not promise protection for arbitrary LEDs or rolling shutters.
        Digital gain must be positive/finite; its candidate family is supplied
        by the caller and shared across all compared baselines.
        """
        budget = self.max_total_capture_s
        if remaining_capture_s is not None:
            if not math.isfinite(remaining_capture_s) or remaining_capture_s < 0:
                raise ValueError("remaining capture budget must be nonnegative and finite")
            budget = min(budget, remaining_capture_s)
        feasible = []
        for action in actions:
            values = _action_dict(action)
            valid = all(math.isfinite(value) and value > 0 for value in values.values())
            if valid:
                t, analog = values["exposure_s"], values["analog_gain"]
                valid = (self.min_exposure_s <= t <= self.max_exposure_s and
                         self.min_analog_gain <= analog <= self.max_analog_gain and
                         t + self.readout_s <= budget + 1e-12)
                if valid and self.anti_flicker_hz is not None:
                    cycles = 2. * self.anti_flicker_hz * t
                    valid = round(cycles) >= 1 and math.isclose(cycles, round(cycles), rel_tol=1e-6, abs_tol=1e-6)
            feasible.append(valid)
        return torch.tensor(feasible, dtype=torch.bool)


@dataclass(frozen=True)
class ControlRequest:
    requested_action: CaptureAction
    requested_frame: int
    effective_frame: int
    source_observation_frame: int | None = None


@dataclass(frozen=True)
class FrameControl:
    frame_index: int
    requested_action: CaptureAction | None
    effective_action: CaptureAction
    capture_bias_ev: float
    render_compensation_ev: float
    requested_effective_frame: int | None
    effective_request_frame: int | None
    source_observation_frame: int | None
    total_capture_s: float
    requested_source_observation_frame: int | None
    effective_due_frame: int | None

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "effective_frame": self.frame_index,
            "requested_frame": self.frame_index if self.requested_action is not None else None,
            "requested_effective_frame": self.requested_effective_frame,
            "effective_request_frame": self.effective_request_frame,
            "source_observation_frame": self.source_observation_frame,
            "requested_source_observation_frame": self.requested_source_observation_frame,
            "effective_due_frame": self.effective_due_frame,
            "requested_action": _action_dict(self.requested_action) if self.requested_action is not None else None,
            "effective_action": _action_dict(self.effective_action),
            "capture_bias_ev": self.capture_bias_ev,
            "render_compensation_ev": self.render_compensation_ev,
            "total_capture_s": self.total_capture_s,
        }


@dataclass(frozen=True)
class ControlledCapture:
    result: CaptureResult
    control: FrameControl


class DelayedExposureController:
    """Apply pending actions only at their simulated effective frame.

    Cursor ``next_frame`` starts at 0. request() submits for that next capture
    tick. advance() returns frame 0 first, then 1, 2, etc. For delay=2:

        controller.request(B)  # requested_frame=0, effective_frame=2
        controller.advance()   # frame=0, effective=A
        controller.advance()   # frame=1, effective=A
        controller.advance()   # frame=2, effective=B

    A policy observing frame 0 submits at next_frame=1; it can record
    source_observation_frame=0. Successful effective frames consume exposure
    plus one readout from the sequence budget. Failed simulation is transactional:
    no cursor, pending action, effective state or successful ledger is committed.
    """
    def __init__(self, initial_action: CaptureAction, reference_action: CaptureAction,
                 constraints: ExposureConstraints):
        if not constraints.feasible_mask([initial_action]).item():
            raise ValueError("initial action must be feasible")
        # The reference sets a brightness coordinate and need not be capturable
        # within this session's constraints.
        if not all(math.isfinite(value) and value > 0 for value in _action_dict(reference_action).values()):
            raise ValueError("reference action must have positive finite physical values")
        self.constraints = constraints
        self.reference_action = reference_action
        self.effective_action = initial_action
        self.next_frame = 0
        self.total_capture_s = 0.
        self.ledger: list[FrameControl] = []
        self._requests: dict[int, ControlRequest] = {}
        self._effective_request: ControlRequest | None = None

    def request(self, action: CaptureAction, *, source_observation_frame=None) -> ControlRequest:
        """Schedule a legal action; do not relabel any existing observation."""
        if self.next_frame in self._requests:
            raise ValueError("a request has already been submitted for this frame")
        if source_observation_frame is not None:
            if isinstance(source_observation_frame, bool) or not isinstance(source_observation_frame, int) or not 0 <= source_observation_frame < self.next_frame:
                raise ValueError("source observation must be an already completed frame")
        remaining = max(0., self.constraints.max_total_capture_s - self.total_capture_s)
        if not self.constraints.feasible_mask([action], remaining_capture_s=remaining).item():
            raise ValueError("requested action must be feasible within the remaining budget")
        request = ControlRequest(action, self.next_frame,
                                 self.next_frame + self.constraints.control_delay_frames,
                                 source_observation_frame)
        self._requests[self.next_frame] = request
        return request

    def _plan(self) -> tuple[FrameControl, ControlRequest | None]:
        effective_request = self._effective_request
        due = [request for request in self._requests.values() if request.effective_frame == self.next_frame]
        if due:
            effective_request = due[-1]
        effective = effective_request.requested_action if effective_request is not None else self.effective_action
        remaining = max(0., self.constraints.max_total_capture_s - self.total_capture_s)
        if not self.constraints.feasible_mask([effective], remaining_capture_s=remaining).item():
            raise ValueError("effective action cannot fit the remaining capture budget")
        bias = (math.log2(effective.exposure_s) - math.log2(self.reference_action.exposure_s) +
                math.log2(effective.analog_gain) - math.log2(self.reference_action.analog_gain) +
                math.log2(effective.digital_gain) - math.log2(self.reference_action.digital_gain))
        submitted = self._requests.get(self.next_frame)
        frame = FrameControl(
            self.next_frame, submitted.requested_action if submitted else None, effective,
            bias, -bias, submitted.effective_frame if submitted else None,
            effective_request.requested_frame if effective_request else None,
            effective_request.source_observation_frame if effective_request else None,
            self.total_capture_s + effective.exposure_s + self.constraints.readout_s,
            submitted.source_observation_frame if submitted else None,
            effective_request.effective_frame if effective_request else None,
        )
        return frame, effective_request

    def _commit(self, frame: FrameControl, effective_request: ControlRequest | None) -> None:
        self.effective_action = frame.effective_action
        self._effective_request = effective_request
        self.total_capture_s = frame.total_capture_s
        self.ledger.append(frame)
        self.next_frame += 1
        # Preserve current/future submissions only; the applied provenance is
        # retained independently by _effective_request and the immutable ledger.
        self._requests = {n: request for n, request in self._requests.items()
                          if request.effective_frame >= self.next_frame}

    def advance(self) -> FrameControl:
        """Commit one simulated frame's effective settings without pixel capture."""
        frame, effective_request = self._plan()
        self._commit(frame, effective_request)
        return frame

    def capture_frame(self, scene: Scene, sensor: SensorProfile, *, center_s=0., seed=0,
                      noisy=True) -> ControlledCapture:
        """Recapture latent radiance with the effective action and commit on success.

        Dynamic callers must explicitly advance center_s; the default is useful
        for static fixtures or common-time candidate comparison only.
        """
        from .simulation import capture

        frame, effective_request = self._plan()
        result = capture(scene, frame.effective_action, sensor, center_s=center_s,
                         seed=seed, noisy=noisy)
        if result.action != frame.effective_action:
            raise ValueError("simulator returned action metadata inconsistent with effective control")
        result.metadata["control"] = {**frame.to_dict(), "center_s": float(center_s), "seed": int(seed)}
        self._commit(frame, effective_request)
        return ControlledCapture(result, frame)
