from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
import numpy as np

from .transforms import exposure_transform, luminance_gamma_transform
from .schemas import sha256_file


class MethodRunner(Protocol):
    def predict(self, image: torch.Tensor, alpha: float) -> Mapping[str, torch.Tensor]: ...
    def metadata(self) -> Mapping[str, Any]: ...


class FrozenBaselineRunner:
    def __init__(self, checkpoint: str | Path, device: str | torch.device):
        from adjustTM.model import ControlledBrightnessISP, load_baseline_checkpoint

        self.device = torch.device(device)
        self.model = ControlledBrightnessISP("param_residual", device=self.device).to(self.device)
        load_baseline_checkpoint(self.model.baseline, checkpoint, map_location=self.device)
        self.model.freeze_baseline()
        self.model.eval()
        self.checkpoint = str(checkpoint)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, alpha: float) -> Mapping[str, torch.Tensor]:
        del alpha
        return self.model.forward_baseline(image.to(self.device))

    def zero_reference(self, image: torch.Tensor) -> torch.Tensor:
        return self.predict(image, 0.0)["output"]

    def metadata(self) -> Mapping[str, Any]:
        return {"type": "frozen_baseline", "checkpoint": self.checkpoint, "checkpoint_sha256": sha256_file(self.checkpoint)}


class ControlledISPRunner:
    def __init__(self, method: str, baseline_checkpoint: str | Path, control_checkpoint: str | Path, device: str | torch.device):
        from adjustTM.model import ControlledBrightnessISP, load_baseline_checkpoint, load_control_checkpoint

        self.device = torch.device(device)
        self.method = method
        self.model = ControlledBrightnessISP(method, device=self.device).to(self.device)
        load_baseline_checkpoint(self.model.baseline, baseline_checkpoint, map_location=self.device)
        self.model.freeze_baseline()
        load_control_checkpoint(self.model, control_checkpoint, map_location=self.device)
        self.model.assert_baseline_frozen()
        self.model.eval()
        self.baseline_checkpoint = str(baseline_checkpoint)
        self.control_checkpoint = str(control_checkpoint)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, alpha: float) -> Mapping[str, torch.Tensor]:
        alpha_tensor = torch.full((image.shape[0],), float(alpha), device=self.device, dtype=image.dtype)
        return self.model(image.to(self.device), alpha_tensor)

    @torch.no_grad()
    def zero_reference(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.forward_baseline(image.to(self.device))["output"]

    def metadata(self) -> Mapping[str, Any]:
        return {
            "type": "controlled_isp",
            "method": self.method,
            "baseline_checkpoint": self.baseline_checkpoint,
            "baseline_checkpoint_sha256": sha256_file(self.baseline_checkpoint),
            "control_checkpoint": self.control_checkpoint,
            "control_checkpoint_sha256": sha256_file(self.control_checkpoint),
            "parameter_report": self.model.parameter_report(),
        }


class SimpleTransformRunner:
    def __init__(self, baseline: MethodRunner, parameters: Mapping[str, float], kind: str):
        self.baseline = baseline
        self.parameters = {str(key): float(value) for key, value in parameters.items()}
        self.kind = kind

    def parameter_for_alpha(self, alpha: float) -> float:
        if all(_is_float(key) for key in self.parameters):
            pairs = sorted((float(key), value) for key, value in self.parameters.items())
        else:
            from adjustTM.constants import LEVEL_TO_ALPHA
            pairs = sorted((float(LEVEL_TO_ALPHA[key]), value) for key, value in self.parameters.items())
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        return float(np.interp(float(alpha), x, y))

    @torch.no_grad()
    def predict(self, image: torch.Tensor, alpha: float) -> Mapping[str, torch.Tensor]:
        result = dict(self.baseline.predict(image, 0.0))
        parameter = self.parameter_for_alpha(alpha)
        output = result["output"]
        if self.kind == "exposure":
            output = exposure_transform(output, parameter)
        elif self.kind == "gamma":
            output = luminance_gamma_transform(output, parameter)
        else:
            raise ValueError(self.kind)
        result["output"] = output
        return result

    def zero_reference(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.baseline, "zero_reference"):
            return self.baseline.zero_reference(image)
        return self.baseline.predict(image, 0.0)["output"]

    def metadata(self) -> Mapping[str, Any]:
        return {"type": f"{self.kind}_global", "parameters": self.parameters}


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_runners(config_path: str | Path, device: str | torch.device) -> dict[str, MethodRunner]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    runners: dict[str, MethodRunner] = {}
    for name, spec in config.items():
        kind = spec["type"]
        if kind == "frozen_baseline":
            runners[name] = FrozenBaselineRunner(spec["baseline_checkpoint"], device)
        elif kind == "controlled_isp":
            runners[name] = ControlledISPRunner(
                spec["control_method"], spec["baseline_checkpoint"], spec["control_checkpoint"], device
            )
        elif kind in {"exposure_global", "gamma_global"}:
            baseline_name = spec.get("baseline_method", "frozen_baseline")
            if baseline_name not in runners:
                raise ValueError(f"Simple method {name} requires previously declared {baseline_name}")
            calibration = json.loads(Path(spec["calibration"]).read_text(encoding="utf-8"))
            key = "exposure_global" if kind == "exposure_global" else "gamma_global"
            runners[name] = SimpleTransformRunner(runners[baseline_name], calibration[key], "exposure" if "exposure" in kind else "gamma")
        else:
            raise ValueError(f"Unknown method type: {kind}")
    return runners
