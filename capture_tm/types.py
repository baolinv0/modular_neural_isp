"""Validated physical capture and explicitly named image-domain contracts."""
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch import Tensor

SCENE_DOMAIN = "sensor_linear_relative_radiance"


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


@dataclass(frozen=True)
class CaptureAction:
    exposure_s: float
    analog_gain: float = 1.
    digital_gain: float = 1.

    def __post_init__(self):
        for name in ("exposure_s", "analog_gain", "digital_gain"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    def scale(self, reference: "CaptureAction") -> float:
        """Brightness ratio only; gains do not create photons or reduce blur."""
        return (self.exposure_s / reference.exposure_s) * (self.analog_gain / reference.analog_gain) * (self.digital_gain / reference.digital_gain)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SensorProfile:
    full_well_e: float = 10000.
    read_noise_e: float = 3.
    adc_noise_dn: float = .5
    bit_depth: int = 12
    black_level_dn: float = 64.
    reference_exposure_s: float = 1 / 120
    wb: tuple = (1., 1., 1.)
    ccm: tuple = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    calibration_status: str = "assumed_engineering"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.provenance,dict):
            raise ValueError("sensor provenance must be a dictionary")
        for name in ("full_well_e", "reference_exposure_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("read_noise_e", "adc_noise_dn", "black_level_dn"):
            object.__setattr__(self, name, _positive(getattr(self, name), name, zero=True))
        if isinstance(self.bit_depth, bool) or int(self.bit_depth) != self.bit_depth or not 2 <= self.bit_depth <= 24:
            raise ValueError("bit_depth must be an integer in [2,24]")
        if self.black_level_dn >= 2 ** self.bit_depth - 1:
            raise ValueError("black_level_dn must be below the ADC maximum")
        wb = torch.as_tensor(self.wb, dtype=torch.float64)
        ccm = torch.as_tensor(self.ccm, dtype=torch.float64)
        if wb.shape != (3,) or not torch.isfinite(wb).all() or not (wb > 0).all():
            raise ValueError("wb must contain three positive finite gains")
        if ccm.shape != (3,3) or not torch.isfinite(ccm).all():
            raise ValueError("ccm must be a finite 3x3 matrix")
        object.__setattr__(self, "wb", tuple(float(x) for x in wb))
        object.__setattr__(self, "ccm", tuple(tuple(float(x) for x in row) for row in ccm))
        if self.calibration_status not in {"assumed_engineering", "measured_sensor"}:
            raise ValueError("unknown sensor calibration_status")
        if self.calibration_status == "measured_sensor" and not self.provenance:
            raise ValueError("measured_sensor requires calibration provenance")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "SensorProfile":
        if not isinstance(value, dict):
            raise ValueError("sensor must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"invalid sensor fields: {error}") from error


@dataclass
class Scene:
    scene_id: str
    split: str
    frames: Tensor
    frame_times_s: Tensor
    subject_mask: Tensor | None = None
    source_kind: str = "synthetic"
    domain: str = SCENE_DOMAIN
    source_id: str | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be nonempty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if self.domain != SCENE_DOMAIN:
            raise ValueError(f"Scene domain must be {SCENE_DOMAIN}")
        if not isinstance(self.frames, Tensor) or self.frames.dtype != torch.float32:
            raise ValueError("frames must be float32 tensors")
        if self.frames.ndim != 4 or self.frames.shape[1] != 3 or min(self.frames.shape) <= 0:
            raise ValueError("frames must be [T,3,H,W]")
        if not torch.isfinite(self.frames).all() or (self.frames < 0).any():
            raise ValueError("frames must contain finite nonnegative radiance; HDR >1 is legal")
        if not isinstance(self.frame_times_s, Tensor) or self.frame_times_s.ndim != 1 or len(self.frame_times_s) != len(self.frames):
            raise ValueError("frame_times_s must match frame count")
        if self.frame_times_s.dtype == torch.bool or torch.is_complex(self.frame_times_s):
            raise ValueError("frame times must be real numeric values, not bool/complex")
        # Image data is float32, but absolute frame timestamps need float64:
        # float32 cannot represent a millisecond shutter near a large epoch,
        # and unsigned integer subtraction can wrap before an ordering check.
        self.frame_times_s = self.frame_times_s.to(dtype=torch.float64, device=self.frames.device)
        if not torch.isfinite(self.frame_times_s).all() or (len(self.frame_times_s)>1 and not (torch.diff(self.frame_times_s)>0).all()):
            raise ValueError("frame times must be finite and strictly increasing")
        if self.subject_mask is not None:
            if self.subject_mask.shape != (1,*self.frames.shape[-2:]) or not torch.isfinite(self.subject_mask).all() or (self.subject_mask < 0).any() or (self.subject_mask > 1).any():
                raise ValueError("subject_mask must be [1,H,W] with finite values in [0,1]")
            self.subject_mask = self.subject_mask.to(dtype=torch.float32, device=self.frames.device)
        self.source_id = self.scene_id if self.source_id is None else self.source_id
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be nonempty")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be an object")


@dataclass
class CaptureResult:
    rgb: Tensor
    noise_variance: Tensor
    saturation_mask: Tensor
    action: CaptureAction
    metadata: dict[str, Any]
