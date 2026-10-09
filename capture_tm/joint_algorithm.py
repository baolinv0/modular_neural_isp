"""Deployable causal AE, physical acquisition, and conditional TM algorithms.

``select`` only requires already captured previews. A camera driver performs
the returned JSON-safe plan, then ``finish`` consumes actual capture readbacks.
``run_simulated`` is the same one-plan inference protocol using the sensor
simulator; it never renders a candidate bank or consumes a training target.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .capture_plan import (CapturePlan, build_plan_bank, compose_captures,
                           observe_previews, plan_features, rule_plan_index)
from .learned_tone import ConditionalToneMapper
from .learned_policy import TemporalExposurePolicy
from .simulation import capture
from .types import CaptureAction, CaptureResult, Scene, SensorProfile


class JointCaptureAlgorithm:
    """One-frame Apple-inspired or three-frame Samsung-inspired AE+TM.

    Input previews use bounded sensor-linear RGB, before WB/CCM. Metadata is
    effective log2 shutter/analog/digital exposure, matching each observation.
    Plans use times relative to the common reference time zero. Camera callers
    must translate those offsets to their physical acquisition clock.
    An optional AcquisitionProfile selects Bayer acquisition and reserves its
    rolling shutter/readout time in the default bank; omission keeps RGB simulation.
    """

    def __init__(self, scheme: str, sensor: SensorProfile, tm=None, policy=None,
                 plans: list[CapturePlan] | None = None, acquisition=None):
        if scheme not in {"apple", "samsung"}:
            raise ValueError("scheme must be apple or samsung")
        if not isinstance(sensor, SensorProfile):
            raise ValueError("sensor must be a SensorProfile")
        self.scheme = scheme
        self.sensor = sensor
        self.acquisition = acquisition
        if acquisition is not None:
            from .acquisition import AcquisitionProfile
            from .pipeline import build_acquisition_plans, validate_capture_protocol
            if not isinstance(acquisition, AcquisitionProfile):
                raise ValueError("acquisition must be an AcquisitionProfile or None")
            if plans is None:
                plans = build_acquisition_plans(scheme, sensor, acquisition)
        self.plans = list(build_plan_bank(scheme, sensor) if plans is None else plans)
        expected_frames = 1 if scheme == "apple" else 3
        if not self.plans or any(not isinstance(plan, CapturePlan) or len(plan.actions) != expected_frames
                                 for plan in self.plans):
            raise ValueError(f"{scheme} plans require {expected_frames} frame(s)")
        if acquisition is not None:
            validate_capture_protocol(self.plans, acquisition)
        self.tm = ConditionalToneMapper(scheme, trainable=False) if tm is None else tm
        if self.tm.scheme != scheme:
            raise ValueError("tone mapper scheme must match the capture algorithm")
        self.tm.eval()
        self.policy = policy
        if self.policy is not None:
            self.policy.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device="cpu", weights=None):
        """Load a runner ``selected.pt`` or ``last.pt`` for inference.

        Restores the exact sensor, optional acquisition profile, ordered plan
        bank, policy architecture, and complete trained TM state. Rule-AE groups
        retain their scene-adaptive rule even though runner checkpoints also
        carry the common AE weights.
        ``weights`` may override a moved source checkpoint path; full trained
        state is subsequently loaded strictly. No optimizer or pickle objects
        are needed. ``checkpoint_render_ev`` records training appearance intent.
        """
        data = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        required = {"scheme", "sensor", "plans", "tm_state", "learned_ae"}
        if not isinstance(data, dict) or data.get("version") != 1 or not required.issubset(data):
            raise ValueError("expected a version 1 joint-experiment group checkpoint")
        if (data.get("manifest_kind") == "capture_tm_acquisition_dataset"
                and data.get("acquisition") is None):
            raise ValueError("acquisition archive checkpoint requires its bound acquisition profile")
        sensor = SensorProfile.from_dict(data["sensor"])
        acquisition = None
        if data.get("acquisition") is not None:
            from .acquisition import AcquisitionProfile
            acquisition = AcquisitionProfile.from_dict(data["acquisition"])
        plans = [CapturePlan(tuple(CaptureAction(**action) for action in plan["actions"]),
                             tuple(plan["centers_s"])) for plan in data["plans"]]
        source_weights = weights if weights is not None else data.get("tm_weights")
        tm = ConditionalToneMapper(data["scheme"], weights=source_weights, trainable=False)
        tm.load_state_dict(data["tm_state"], strict=True)
        tm.requires_grad_(False).to(device)
        policy = None
        if data["learned_ae"]:
            if not {"policy_kwargs", "policy_state"}.issubset(data):
                raise ValueError("learned AE checkpoint is missing its architecture or parameters")
            policy = TemporalExposurePolicy(**data["policy_kwargs"])
            policy.load_state_dict(data["policy_state"], strict=True)
            policy.requires_grad_(False).to(device)
        algorithm = cls(data["scheme"], sensor, tm=tm, policy=policy, plans=plans,
                        acquisition=acquisition)
        algorithm.checkpoint_render_ev = float(data.get("render_ev", 0.))
        algorithm.clip_risk_tolerance = data.get('clip_risk_tolerance')
        algorithm.checkpoint_group = data.get("group")
        algorithm.checkpoint_manifest_kind = data.get("manifest_kind")
        return algorithm

    @torch.no_grad()
    def select(self, previews: torch.Tensor, state: torch.Tensor,
               feasible_mask: torch.Tensor | None = None) -> dict:
        """Return a JSON-safe physical request, using only past observations.

        Exactly three observed previews are used by both rule and learned AE.
        ``algorithm.plans[request['selected_index']]`` exposes the immutable
        selected CapturePlan. No pending request is retained for rendering;
        ``finish`` always uses the camera's effective capture metadata.
        """
        if (not isinstance(previews, torch.Tensor) or previews.ndim != 4
                or previews.shape[:2] != (3, 3) or min(previews.shape[-2:]) < 1
                or not previews.is_floating_point() or not torch.isfinite(previews).all()
                or (previews < 0).any() or (previews > 1).any()):
            raise ValueError("previews must be finite [3,3,H,W] sensor observations in [0,1]")
        if (not isinstance(state, torch.Tensor) or state.shape != (3, 3)
                or not state.is_floating_point() or not torch.isfinite(state).all()):
            raise ValueError("state must contain finite effective capture metadata [3,3]")
        if feasible_mask is None:
            feasible_mask = torch.ones(len(self.plans), dtype=torch.bool, device=previews.device)
        if (not isinstance(feasible_mask, torch.Tensor) or feasible_mask.dtype != torch.bool
                or feasible_mask.shape != (len(self.plans),) or not feasible_mask.any()):
            raise ValueError("feasible_mask must be boolean [K] with a feasible plan")
        if self.policy is None:
            legal_indices = feasible_mask.nonzero(as_tuple=True)[0].tolist()
            legal_plans = [self.plans[index] for index in legal_indices]
            local_index = rule_plan_index(previews, state.to(previews), legal_plans, self.sensor)
            index = legal_indices[local_index]
        else:
            parameter = next(self.policy.parameters())
            observed = previews.to(parameter)
            effective_state = state.to(parameter)
            candidates = plan_features(self.plans, self.sensor).to(parameter)
            tolerance = getattr(self, 'clip_risk_tolerance', None)
            if tolerance is not None:
                from .learned_policy import preview_clipping_guard
                legal_indices = feasible_mask.nonzero(as_tuple=True)[0].tolist()
                local = rule_plan_index(previews, state.to(previews),
                                       [self.plans[i] for i in legal_indices], self.sensor)
                guard, _ = preview_clipping_guard(observed[None], effective_state[None],
                    candidates[None], torch.tensor([legal_indices[local]], device=parameter.device),
                    tolerance=tolerance)
                feasible_mask = feasible_mask.to(parameter.device) & guard[0]
            scores = self.policy(observed[None], effective_state[None], candidates[None],
                                 feasible_mask.to(parameter.device)[None])
            index = int(scores.argmax(-1).item())
        plan = self.plans[index]
        request = {"scheme": self.scheme, "selected_index": index,
                   "plan": {"actions": [action.to_dict() for action in plan.actions],
                            "centers_s": list(plan.centers_s)},
                   "rule_vs_learned": "rule" if self.policy is None else "learned"}
        if self.acquisition is not None:
            request["acquisition"] = self.acquisition.to_dict()
        return request

    @torch.no_grad()
    def finish(self, captures: list[CaptureResult], render_ev: float | None = None) -> dict:
        """Fuse actual measurements and render once using actual effective EV.

        The simulator-only variance and latent saturation fields of
        CaptureResult are not required by fusion. Its confidence is estimated
        from measured samples, effective settings, and the sensor profile.
        RawCaptureResult inherits this readback contract via demosaiced RGB.
        Bound acquisition validates actual rolling exposure/readout ordering,
        accepting timestamps on the camera's absolute clock.
        Omitting render_ev preserves loaded checkpoint appearance intent, or
        uses zero EV for a freshly constructed algorithm.
        """
        expected_frames = 1 if self.scheme == "apple" else 3
        if len(captures) != expected_frames:
            raise ValueError(f"{self.scheme} finish requires {expected_frames} frame(s)")
        if self.acquisition is not None:
            from .acquisition import AcquisitionProfile
            starts, ends = [], []
            for result in captures:
                try:
                    center = float(result.metadata["center_s"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("effective capture center_s must be finite") from error
                if not math.isfinite(center):
                    raise ValueError("effective capture center_s must be finite")
                declared_profile = result.metadata.get("acquisition")
                if (declared_profile is not None
                        and AcquisitionProfile.from_dict(declared_profile) != self.acquisition):
                    raise ValueError("effective capture acquisition profile differs from bound profile")
                half = (result.action.exposure_s + self.acquisition.rolling_shutter_s) / 2
                starts.append(center - half)
                ends.append(center + half + self.acquisition.readout_s)
            if any(end > start + 1e-10 for end, start in zip(ends[:-1], starts[1:])):
                raise ValueError("effective rolling shutter/readout captures overlap")
        if render_ev is None:
            render_ev = getattr(self, "checkpoint_render_ev", 0.)
        fused = compose_captures(captures, self.sensor)
        parameter = next(self.tm.parameters())
        image = fused["image"].to(parameter)
        reliability = fused["reliability"].to(parameter)
        output = self.tm(image[None], capture_ev=fused["capture_ev"], render_ev=render_ev,
                         reliability=reliability[None])[0]
        return {"output": output, "fusion": fused,
                "metadata": {"scheme": self.scheme,
                             "effective_actions": [result.action.to_dict() for result in captures],
                             "effective_centers_s": [float(result.metadata["center_s"]) for result in captures],
                             "render_ev": float(render_ev), "selected_plan_only": True}}

    @torch.no_grad()
    def run_simulated(self, scene: Scene, seed: int = 0, noisy: bool = True,
                      render_ev: float | None = None) -> dict:
        """Observe history, choose once, simulate only that plan, then render.

        Preview noise remains physical in both modes; ``noisy=False`` disables
        noise only for final acquisition, matching the candidate simulator.
        Training oracle evaluation belongs to the experiment runner, never here.
        """
        if self.acquisition is None:
            previews, state, observation_metadata = observe_previews(scene, self.sensor, seed=seed)
        else:
            from .pipeline import observe_raw_previews
            previews, state, observation_metadata = observe_raw_previews(
                scene, self.sensor, self.acquisition, seed=seed)
        request = self.select(previews, state)
        plan = self.plans[request["selected_index"]]
        if self.acquisition is None:
            if min(t - action.exposure_s / 2 for action, t in zip(plan.actions, plan.centers_s)) <= max(
                    observation_metadata["shutter_ends_s"]):
                raise ValueError("selected capture must start after all observed preview shutters")
            captures = [capture(scene, action, self.sensor, center_s=center,
                                seed=seed + 1000003 + frame * 104729, noisy=noisy)
                        for frame, (action, center) in enumerate(zip(plan.actions, plan.centers_s))]
        else:
            from .acquisition import capture_raw
            from .pipeline import validate_capture_protocol
            validate_capture_protocol([plan], self.acquisition, observation_metadata)
            captures = [capture_raw(scene, action, self.sensor, acquisition=self.acquisition,
                                    center_s=center, seed=seed + 1000003 + frame * 104729, noisy=noisy)
                        for frame, (action, center) in enumerate(zip(plan.actions, plan.centers_s))]
        result = self.finish(captures, render_ev=render_ev)
        result.update({"request": request, "captures": captures,
                       "observations": {"previews": previews, "state": state,
                                        "metadata": observation_metadata}})
        return result
