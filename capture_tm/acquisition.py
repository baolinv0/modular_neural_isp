"""Native Bayer acquisition from explicitly linear, relative sensor radiance.

This is an engineering simulator, not a measured Apple/Samsung calibration.
Rows expose top to bottom: the first and last row shutter centers are
``center_s - rolling_shutter_s/2`` and ``center_s + rolling_shutter_s/2``.
``readout_s`` follows the final row's shutter close. It does not collect light;
capture protocols must reserve that time separately from scene shutter support.
"""
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import CaptureAction, CaptureResult, Scene, SensorProfile


_CFA_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}
_PREPARED_KEY = "physical_acquisition"


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _kernel_tensor(value: Any) -> Tensor:
    try:
        kernels = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("psf_kernels must be rectangular numeric kernels") from error
    if kernels.ndim == 2:
        kernels = kernels[None].expand(3, -1, -1)
    elif kernels.ndim == 4 and kernels.shape[1] == 1:
        kernels = kernels[:, 0]
    if kernels.ndim != 3 or kernels.shape[0] != 3:
        raise ValueError("psf_kernels must be [H,W], [3,H,W], or [3,1,H,W]")
    if any(size <= 0 or size % 2 == 0 for size in kernels.shape[-2:]):
        raise ValueError("PSF dimensions must be positive and odd")
    if not torch.isfinite(kernels).all() or (kernels < 0).any() or (kernels.sum((-2, -1)) <= 0).any():
        raise ValueError("PSFs must be finite, nonnegative, and have positive energy")
    return kernels


@dataclass(frozen=True)
class AcquisitionProfile:
    cfa_pattern: str = "RGGB"
    spatial_downsample: int = 1
    readout_s: float = .001
    rolling_shutter_s: float = 0.
    input_stage: str = "post_optics"
    psf_kernels: Any = None

    def __post_init__(self):
        if self.cfa_pattern not in _CFA_PATTERNS:
            raise ValueError("cfa_pattern must be RGGB, BGGR, GRBG, or GBRG")
        factor = self.spatial_downsample
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
            raise ValueError("spatial_downsample must be a positive integer")
        for name in ("readout_s", "rolling_shutter_s"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        if self.input_stage not in {"pre_optics", "post_optics"}:
            raise ValueError("input_stage must be pre_optics or post_optics")
        if self.psf_kernels is not None:
            if self.input_stage == "post_optics":
                raise ValueError("post_optics input cannot apply an additional PSF; declare pre_optics input")
            kernels = _kernel_tensor(self.psf_kernels)
            # An immutable, serializable profile identity prevents accidental
            # double application of optics/area reduction to a prepared Scene.
            immutable = tuple(tuple(tuple(float(x) for x in row) for row in channel) for channel in kernels)
            object.__setattr__(self, "psf_kernels", immutable)

    def to_dict(self) -> dict:
        kernels = None if self.psf_kernels is None else [
            [list(row) for row in channel] for channel in self.psf_kernels
        ]
        return {"cfa_pattern": self.cfa_pattern, "spatial_downsample": self.spatial_downsample,
                "readout_s": self.readout_s, "rolling_shutter_s": self.rolling_shutter_s,
                "input_stage": self.input_stage, "psf_kernels": kernels}

    @classmethod
    def from_dict(cls, value: dict) -> "AcquisitionProfile":
        if not isinstance(value, dict):
            raise ValueError("acquisition must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"invalid acquisition fields: {error}") from error


@dataclass
class RawCaptureResult(CaptureResult):
    """RGB compatibility bridge plus native quantized Bayer measurements.

    ``raw_dn`` contains ADC DN including black level. ``raw_noise_variance``
    is in squared black-subtracted sensor-normalized units, as is RGB variance.
    Both are uncensored analytic proxies; clipping invalidates their estimates.
    """
    raw_dn: Tensor
    raw_noise_variance: Tensor
    raw_saturation_mask: Tensor


def _check_image(value: Tensor, channels: int, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim not in (3, 4) or value.shape[-3] != channels:
        raise ValueError(f"{name} must be [{channels},H,W] or [N,{channels},H,W]")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    if min(value.shape[-2:]) < 2 or any(size % 2 for size in value.shape[-2:]):
        raise ValueError("Bayer geometry must have even height and width of at least two")
    if value.ndim == 4 and value.shape[0] == 0:
        raise ValueError("Bayer batch cannot be empty")


def _cfa_masks(height: int, width: int, pattern: str, like: Tensor) -> Tensor:
    if pattern not in _CFA_PATTERNS:
        raise ValueError("cfa_pattern must be RGGB, BGGR, GRBG, or GBRG")
    masks = like.new_zeros((3, height, width))
    for phase, letter in enumerate(pattern):
        masks["RGB".index(letter), phase // 2::2, phase % 2::2] = 1
    return masks


def mosaic_bayer(rgb: Tensor, cfa_pattern: str = "RGGB") -> Tensor:
    """Select one actual sensor channel at each Bayer site, without clipping."""
    _check_image(rgb, 3, "rgb")
    masks = _cfa_masks(*rgb.shape[-2:], cfa_pattern, rgb)
    return (rgb * masks).sum(-3, keepdim=True)


def _demosaic_geometry(raw: Tensor, pattern: str) -> tuple[Tensor, Tensor, Tensor]:
    masks = _cfa_masks(*raw.shape[-2:], pattern, raw)[None]
    rb = [[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]
    green = [[0., 1., 0.], [1., 4., 1.], [0., 1., 0.]]
    kernels = raw.new_tensor([rb, green, rb])[:, None]
    denominator = F.conv2d(masks, kernels, padding=1, groups=3)
    return masks, kernels, denominator


def _interpolate(raw: Tensor, pattern: str, *, variance: bool = False, mask: bool = False) -> Tensor:
    singleton = raw.ndim == 3
    batched = raw[None] if singleton else raw
    masks, kernels, denominator = _demosaic_geometry(batched, pattern)
    if variance:
        kernels = kernels.square()
        denominator = denominator.square()
    interpolated = F.conv2d(batched * masks, kernels, padding=1, groups=3)
    result = interpolated > 0 if mask else interpolated / denominator
    return result[0] if singleton else result


def demosaic_bayer(raw: Tensor, cfa_pattern: str = "RGGB") -> Tensor:
    """Bilinear camera RGB, preserving CFA samples and constant-color edges.

    At boundaries, weights are normalized over the available in-image samples.
    Negative black-subtracted values are retained. No WB/CCM or tone mapping is
    applied. Independent sample variance requires squared interpolation weights.
    """
    _check_image(raw, 1, "raw")
    return _interpolate(raw, cfa_pattern)


def prepare_scene(scene: Scene, acquisition: AcquisitionProfile) -> Scene:
    """Apply declared optics and pixel-area averages once at native resolution.

    Temporal support and radiometric scale are unchanged. Consequently the
    instantaneous image at t=0 is a common reference after optics/pixel area,
    before any finite shutter, noise, analog gain, or ADC clipping.
    """
    if not isinstance(acquisition, AcquisitionProfile):
        raise ValueError("acquisition must be an AcquisitionProfile")
    marker = scene.provenance.get(_PREPARED_KEY)
    if marker is not None:
        if not isinstance(marker, dict) or marker.get("profile") != acquisition.to_dict():
            raise ValueError("prepared scene acquisition profile differs; prepare the original source scene")
        return scene
    height, width = scene.frames.shape[-2:]
    factor = acquisition.spatial_downsample
    if height % factor or width % factor:
        raise ValueError("source dimensions must be divisible by spatial_downsample")
    native_height, native_width = height // factor, width // factor
    if min(native_height, native_width) < 2 or native_height % 2 or native_width % 2:
        raise ValueError("native Bayer height and width must be even and at least two")
    frames = scene.frames
    applied_psf = acquisition.input_stage == "pre_optics" and acquisition.psf_kernels is not None
    if applied_psf:
        kernels = _kernel_tensor(acquisition.psf_kernels).to(frames)
        kernels = kernels / kernels.sum((-2, -1), keepdim=True)
        py, px = kernels.shape[-2] // 2, kernels.shape[-1] // 2
        # torch.conv2d computes correlation. A declared PSF is the response to
        # a point source, so flip its spatial axes for physical convolution.
        frames = F.conv2d(F.pad(frames, (px, px, py, py), mode="replicate"),
                          kernels.flip((-2, -1))[:, None], groups=3)
    if factor != 1:
        frames = F.avg_pool2d(frames, factor, stride=factor)
    subject_mask = scene.subject_mask
    if subject_mask is not None and factor != 1:
        subject_mask = F.avg_pool2d(subject_mask[None], factor, stride=factor)[0]
    marker = {"version": 1, "profile": acquisition.to_dict(),
              "source_resolution": [height, width], "native_resolution": [native_height, native_width],
              "source_input_stage": acquisition.input_stage, "psf_applied": applied_psf,
              "psf_convention": "point_response_convolution",
              "optics_assumption": "source_already_post_optics" if acquisition.input_stage == "post_optics" else
                                   ("declared_per_channel_PSF" if applied_psf else "ideal_delta_PSF"),
              "psf_boundary": "replicate" if applied_psf else None,
              "pixel_area_reduction": "nonoverlapping_arithmetic_mean",
              "reference_stage": "after_optics_and_pixel_area_before_shutter",
              "reference_time_s": 0.}
    return Scene(scene.scene_id, scene.split, frames, scene.frame_times_s, subject_mask,
                 scene.source_kind, scene.domain, scene.source_id,
                 {**scene.provenance, _PREPARED_KEY: marker})


def _integrate_rows(scene: Scene, exposure_s: float, center_s: float, rolling_s: float) -> tuple[Tensor, int]:
    """Exact per-row trapezoid integration of the piecewise-linear source."""
    if len(scene.frames) == 1:
        return scene.frames[0], 1
    # Work relative to the first timestamp; epoch size must not loosen support.
    times = scene.frame_times_s - scene.frame_times_s[0]
    relative_center = center_s - float(scene.frame_times_s[0])
    centers = torch.linspace(-rolling_s / 2, rolling_s / 2, scene.frames.shape[-2],
                             dtype=torch.float64, device=times.device) + relative_center
    start, end = centers - exposure_s / 2, centers + exposure_s / 2
    tolerance = max(1e-12, float(times[-1]) * torch.finfo(torch.float64).eps * 16)
    if float(start.min()) < -tolerance or float(end.max()) > float(times[-1]) + tolerance:
        raise ValueError(f"rolling shutter interval exceeds scene time support [0,{float(times[-1]):g}]")
    start, end = start.clamp_min(0.), end.clamp_max(float(times[-1]))
    duration = end - start
    if not torch.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("exposure duration cannot be resolved within scene time support")
    left, right = times[:-1, None], times[1:, None]
    span = right - left
    a, b = torch.maximum(start[None], left), torch.minimum(end[None], right)
    ua, ub = (a - left) / span, (b - left) / span
    valid = b > a
    second = torch.where(valid, .5 * (ub.square() - ua.square()) * span, 0.)
    first = torch.where(valid, (ub - ua) * span, 0.) - second
    weights = times.new_zeros((len(times), len(centers)))
    weights[:-1] += first
    weights[1:] += second
    weights /= duration[None]
    radiance = torch.einsum("th,tchw->chw", weights.to(scene.frames), scene.frames)
    temporal_samples = int(((times[:, None] > start) & (times[:, None] < end)).sum(0).max()) + 2
    return radiance, temporal_samples


def capture_raw(scene: Scene, action: CaptureAction, sensor: SensorProfile, *,
                acquisition: AcquisitionProfile | None = None, center_s: float = 0.,
                seed: int = 0, noisy: bool = True) -> RawCaptureResult:
    """Integrate light, sample the CFA, read the sensor, then bilinear demosaic."""
    if action.digital_gain != 1.:
        raise ValueError("capture_raw requires digital_gain=1; virtual gain belongs to tone mapping")
    center_s = float(center_s)
    if not math.isfinite(center_s):
        raise ValueError("center_s must be finite")
    if acquisition is None:
        marker = scene.provenance.get(_PREPARED_KEY)
        acquisition = AcquisitionProfile.from_dict(marker["profile"]) if isinstance(marker, dict) and "profile" in marker else AcquisitionProfile()
    scene = prepare_scene(scene, acquisition)
    marker = scene.provenance[_PREPARED_KEY]
    half_window = (action.exposure_s + acquisition.rolling_shutter_s) / 2
    shutter_interval = [center_s - half_window, center_s + half_window]
    readout_end = shutter_interval[1] + acquisition.readout_s
    if not all(math.isfinite(t) for t in (*shutter_interval, readout_end)):
        raise ValueError("capture timing interval must be finite")
    radiance, temporal_samples = _integrate_rows(scene, action.exposure_s, center_s, acquisition.rolling_shutter_s)
    raw_radiance = mosaic_bayer(radiance, acquisition.cfa_pattern)
    # Only nonnegative photon irradiance is physical; no unit-range or full-well
    # clipping has occurred before integration and CFA channel selection.
    expected_e = raw_radiance.clamp_min(0.) * (action.exposure_s / sensor.reference_exposure_s) * sensor.full_well_e
    if not torch.isfinite(expected_e).all():
        raise ValueError("capture electron expectation overflow")
    generator = torch.Generator(device=radiance.device).manual_seed(int(seed))
    photons = torch.poisson(expected_e, generator=generator) if noisy else expected_e.clone()
    full_well_mask = photons >= sensor.full_well_e
    electrons = photons.clamp_max(sensor.full_well_e)
    if noisy and sensor.read_noise_e:
        electrons = electrons + torch.randn(electrons.shape, dtype=electrons.dtype, device=electrons.device,
                                           generator=generator) * sensor.read_noise_e
    adc_max = float(2 ** sensor.bit_depth - 1)
    adc_range = adc_max - sensor.black_level_dn
    dn = electrons / sensor.full_well_e * action.analog_gain * adc_range
    if noisy and sensor.adc_noise_dn:
        dn = dn + torch.randn(dn.shape, dtype=dn.dtype, device=dn.device, generator=generator) * sensor.adc_noise_dn
    dn = dn + sensor.black_level_dn
    adc_mask, black_underflow = dn >= adc_max, dn < sensor.black_level_dn
    raw_dn = dn.round().clamp(0., adc_max)
    normalized = (raw_dn - sensor.black_level_dn) / adc_range
    raw_variance = ((expected_e + sensor.read_noise_e ** 2) * (action.analog_gain / sensor.full_well_e) ** 2 +
                    (sensor.adc_noise_dn ** 2 + 1 / 12) / adc_range ** 2)
    raw_saturation = full_well_mask | adc_mask
    rgb = demosaic_bayer(normalized, acquisition.cfa_pattern)
    variance = _interpolate(raw_variance, acquisition.cfa_pattern, variance=True)
    saturation = _interpolate(raw_saturation.to(rgb), acquisition.cfa_pattern, mask=True)
    metadata = {
        "scene_id": scene.scene_id, "source_id": scene.source_id, "source_kind": scene.source_kind,
        "split": scene.split, "provenance": dict(scene.provenance),
        "domain": "sensor_linear_rgb", "source_domain": scene.domain, "raw_domain": "bayer_adc_dn",
        "action": action.to_dict(), "acquisition": acquisition.to_dict(), "center_s": center_s,
        "shutter_interval_s": shutter_interval, "readout_end_s": readout_end,
        "rolling_row_center_convention": "top_to_bottom_centered_first_to_last_row_span",
        "first_row_center_s": center_s - acquisition.rolling_shutter_s / 2,
        "last_row_center_s": center_s + acquisition.rolling_shutter_s / 2,
        "source_resolution": marker["source_resolution"], "native_resolution": marker["native_resolution"],
        "source_input_stage": marker["source_input_stage"], "psf_applied": marker["psf_applied"],
        "psf_convention": marker["psf_convention"],
        "optics_assumption": marker["optics_assumption"],
        "reference_stage": marker["reference_stage"], "reference_time_s": 0.,
        "reference_exposure_s": sensor.reference_exposure_s, "temporal_samples": temporal_samples,
        "integrated_radiance_mean": float(radiance.mean()),
        "expected_photons_mean_e": float(expected_e.mean()),
        "full_well_saturation_fraction": float(full_well_mask.float().mean()),
        "adc_saturation_fraction": float(adc_mask.float().mean()),
        "black_underflow_fraction": float(black_underflow.float().mean()),
        "noise_seed": int(seed), "noisy": bool(noisy),
        "raw_noise_variance_domain": "black_subtracted_sensor_normalized_unclipped_variance_proxy",
        "noise_variance_domain": "demosaiced_sensor_normalized_unclipped_variance_proxy",
        "noise_variance_validity": "away from ADC clipping and full-well saturation; independent CFA samples; squared bilinear weights; uniform quantization approximation; hypothetical proxy when noisy=False; demosaic introduces spatial/channel covariance not returned",
        "reference_calibration_assumption": "unit-gain reference white equals full-well; unity ADC gain maps full-well to white DN",
        "sensor_calibration_status": sensor.calibration_status, "sensor_provenance": dict(sensor.provenance),
        "calibration_limits": ["relative sensor radiance, not absolute photon flux or measured phone calibration",
                               "independent CFA shot/read/ADC noise; no crosstalk, row noise, or spectral calibration",
                               "declared PSF and pixel-area model; bilinear demosaic with available-sample boundary normalization"],
        "physical_model": "Bayer_Poisson_fullwell_read_analog_ADC_bilinear_v1",
    }
    return RawCaptureResult(rgb.float(), variance.float(), saturation, action, metadata,
                            raw_dn.float(), raw_variance.float(), raw_saturation)
