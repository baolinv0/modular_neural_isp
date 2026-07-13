from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from photofinishing.photofinishing_model import (
    GainNet,
    GlobalToneMappingNet,
    LocalToneMappingNet,
    PhotofinishingModule,
)

from .adapters import ControlledPredictor
from .constants import CONTROL_METHODS
from .transfer import linear_to_srgb


class LuminanceOnlyBaseline(nn.Module):
    """Gain -> GTM -> frozen LTM -> fixed sRGB OETF."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        device = torch.device(device)
        act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self._gain_net = GainNet(act_func=act).to(device)
        self._gtm_net = GlobalToneMappingNet(act_func=act).to(device)
        self._ltm_net = LocalToneMappingNet(act_func=act).to(device)

    def forward(self, x: torch.Tensor, training_mode: bool = False) -> dict[str, torch.Tensor]:
        gain = self._gain_net(x)
        x_gain = PhotofinishingModule._apply_gain(x, gain)
        gtm_params = self._gtm_net(x_gain)
        x_gtm = PhotofinishingModule._apply_gtm(x_gain, gtm_params)
        ltm_params = self._ltm_net(x_gain, x_gtm, training_mode=training_mode)
        x_ltm = PhotofinishingModule._apply_ltm(x_gain, x_gtm, ltm_params).clamp(0.0, 1.0)
        return {
            "output": linear_to_srgb(x_ltm),
            "linear_output": x_ltm,
            "gain_factor": gain,
            "gtm_params": gtm_params,
            "ltm_params": ltm_params,
        }


class ControlledBrightnessISP(nn.Module):
    def __init__(
        self,
        control_method: str,
        device: torch.device | str = "cpu",
        baseline: LuminanceOnlyBaseline | None = None,
    ) -> None:
        super().__init__()
        if control_method not in CONTROL_METHODS:
            raise ValueError(f"control_method must be one of {CONTROL_METHODS}")
        self.control_method = control_method
        self.baseline = baseline or LuminanceOnlyBaseline(device=device)
        self.freeze_baseline()
        self.gain_control = ControlledPredictor(
            self.baseline._gain_net,
            sequential_attr="gain",
            control_method=control_method,
            output_kind="gain",
        )
        self.gtm_control = ControlledPredictor(
            self.baseline._gtm_net,
            sequential_attr="gtm",
            control_method=control_method,
            output_kind="gtm",
        )

    def freeze_baseline(self) -> None:
        for parameter in self.baseline.parameters():
            parameter.requires_grad = False

    def control_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and "control" in name:
                yield parameter

    def forward_baseline(self, x: torch.Tensor, training_mode: bool = False) -> dict[str, torch.Tensor]:
        return self.baseline(x, training_mode=training_mode)

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        training_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        gain = self.gain_control(x, alpha)
        x_gain = PhotofinishingModule._apply_gain(x, gain)
        gtm_params = self.gtm_control(x_gain, alpha)
        x_gtm = PhotofinishingModule._apply_gtm(x_gain, gtm_params)
        ltm_params = self.baseline._ltm_net(x_gain, x_gtm, training_mode=training_mode)
        x_ltm = PhotofinishingModule._apply_ltm(x_gain, x_gtm, ltm_params).clamp(0.0, 1.0)
        return {
            "output": linear_to_srgb(x_ltm),
            "linear_output": x_ltm,
            "gain_factor": gain,
            "gtm_params": gtm_params,
            "ltm_params": ltm_params,
        }

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        modules = {
            "gain_control": sum(p.numel() for p in self.gain_control.parameters() if p.requires_grad),
            "gtm_control": sum(p.numel() for p in self.gtm_control.parameters() if p.requires_grad),
            "ltm_trainable": sum(p.numel() for p in self.baseline._ltm_net.parameters() if p.requires_grad),
        }
        return {"total": total, "frozen": total - trainable, "trainable": trainable, "modules": modules}

    def control_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith("gain_control.") or key.startswith("gtm_control.")
        }


def _unwrap_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint does not contain a state_dict")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("baseline."):
            key = key[len("baseline.") :]
        normalized[key] = value
    return normalized


def load_baseline_checkpoint(
    model: LuminanceOnlyBaseline,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> None:
    state = _unwrap_checkpoint(torch.load(checkpoint_path, map_location=map_location))
    allowed_extra_prefixes = ("_gamma_net.", "_lut_net.", "_3d_lut.")
    filtered = {key: value for key, value in state.items() if not key.startswith(allowed_extra_prefixes)}
    result = model.load_state_dict(filtered, strict=False)
    missing = list(result.missing_keys)
    unexpected = [key for key in result.unexpected_keys if not key.startswith(allowed_extra_prefixes)]
    if missing or unexpected:
        raise RuntimeError(f"Incompatible baseline checkpoint; missing={missing}, unexpected={unexpected}")


def load_control_checkpoint(
    model: ControlledBrightnessISP,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("Control checkpoint must be a dictionary")
    method = checkpoint.get("control_method")
    if method is not None and method != model.control_method:
        raise ValueError(f"Checkpoint method {method!r} does not match model method {model.control_method!r}")
    state = checkpoint.get("control_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("Control checkpoint has no control_state_dict")
    current = model.state_dict()
    unknown = sorted(set(state) - set(current))
    if unknown:
        raise RuntimeError(f"Unknown control keys: {unknown}")
    current.update(state)
    model.load_state_dict(current, strict=True)
    return checkpoint
