"""Electron-domain research sensor model with exact piecewise-linear integration.

Defaults are declared engineering assumptions, not a calibrated phone sensor.
Digital gain remains floating-point HDR; only full-well and ADC clipping are
irreversible. Spatial RGB channels use independent noise, without CFA/row noise.
"""
import math
from typing import Iterable
import torch

from .types import CaptureAction, CaptureResult, Scene, SensorProfile


def _integrate(scene: Scene, exposure_s: float, center_s: float):
    if not math.isfinite(center_s):
        raise ValueError("center_s must be finite")
    if len(scene.frames) == 1:
        return scene.frames[0], 1
    absolute_times = scene.frame_times_s
    # Translate the time origin before constructing a small shutter window.
    # Numerical support tolerance must depend on the span, not on an epoch.
    relative_center = center_s-float(absolute_times[0])
    times = absolute_times-absolute_times[0]
    start, end = relative_center-exposure_s/2, relative_center+exposure_s/2
    tolerance = max(1e-12,float(times[-1])*torch.finfo(torch.float64).eps*16)
    if start < float(times[0])-tolerance or end > float(times[-1])+tolerance:
        raise ValueError(f"shutter interval [{start:g},{end:g}] exceeds scene time support [{float(times[0]):g},{float(times[-1]):g}]")
    start, end = max(start,float(times[0])), min(end,float(times[-1]))
    if end <= start:
        raise ValueError("exposure duration cannot be resolved within scene time support")
    knots = torch.cat([times.new_tensor([start]),times[(times>start)&(times<end)],times.new_tensor([end])])
    upper = torch.searchsorted(times,knots,right=True).clamp(1,len(times)-1)
    lower = upper-1
    weight = (knots-times[lower])/(times[upper]-times[lower])
    images = scene.frames[lower]*(1-weight[:,None,None,None]) + scene.frames[upper]*weight[:,None,None,None]
    integral = ((images[:-1]+images[1:])*.5*torch.diff(knots)[:,None,None,None]).sum(0)
    return (integral/(end-start)).float(), len(knots)


def capture(scene: Scene, action: CaptureAction, sensor: SensorProfile, *, center_s=0., seed=0, noisy=True) -> CaptureResult:
    radiance, temporal_samples = _integrate(scene, action.exposure_s, float(center_s))
    expected_e = radiance * (action.exposure_s/sensor.reference_exposure_s) * sensor.full_well_e
    if not torch.isfinite(expected_e).all():
        raise ValueError("capture electron expectation overflow")
    generator = torch.Generator(device=radiance.device).manual_seed(int(seed))
    photons = torch.poisson(expected_e, generator=generator) if noisy else expected_e.clone()
    full_well_mask = photons >= sensor.full_well_e
    electrons = photons.clamp(max=sensor.full_well_e)
    if noisy and sensor.read_noise_e:
        electrons = electrons + torch.randn(electrons.shape, device=electrons.device, dtype=electrons.dtype, generator=generator)*sensor.read_noise_e
    adc_max = float(2**sensor.bit_depth-1)
    adc_range = adc_max-sensor.black_level_dn
    dn = electrons/sensor.full_well_e*action.analog_gain*adc_range
    if noisy and sensor.adc_noise_dn:
        dn = dn + torch.randn(dn.shape,device=dn.device,dtype=dn.dtype,generator=generator)*sensor.adc_noise_dn
    dn = dn+sensor.black_level_dn
    adc_mask = dn >= adc_max
    black_underflow = dn < sensor.black_level_dn
    quantized = dn.round().clamp(0,adc_max)
    rgb = ((quantized-sensor.black_level_dn)/adc_range)*action.digital_gain
    # Analytic variance before clipping/quantization, plus uniform quantization
    # approximation; it is not an estimate of censored saturated pixel variance.
    variance = ((expected_e+sensor.read_noise_e**2)*(action.analog_gain/sensor.full_well_e)**2 + (sensor.adc_noise_dn**2+1/12)/adc_range**2)*action.digital_gain**2
    metadata = {
        "scene_id": scene.scene_id,"source_id":scene.source_id,"source_kind":scene.source_kind,
        "split":scene.split,"domain":"sensor_linear_rgb","provenance":dict(scene.provenance),
        "action":action.to_dict(),"center_s":float(center_s),
        "shutter_interval_s":[float(center_s)-action.exposure_s/2,float(center_s)+action.exposure_s/2],
        "reference_exposure_s":sensor.reference_exposure_s,"temporal_samples":temporal_samples,
        "integrated_radiance_mean":float(radiance.mean()),
        "expected_photons_mean_e":float(expected_e.mean()),
        "full_well_saturation_fraction":float(full_well_mask.float().mean()),
        "adc_saturation_fraction":float(adc_mask.float().mean()),
        "black_underflow_fraction":float(black_underflow.float().mean()),
        "digital_overrange_fraction":float((rgb>1).float().mean()),
        "noise_seed":int(seed),"noisy":bool(noisy),
        "noise_variance_domain":"sensor_normalized_unclipped_variance_proxy",
        "noise_variance_validity":"away from upper/lower ADC clipping and full-well saturation; uniform quantization error approximation; hypothetical proxy when noisy=False",
        "reference_calibration_assumption":"unit-gain reference white equals full-well; unity ADC gain maps full-well to white DN",
        "sensor_calibration_status":sensor.calibration_status,
        "sensor_provenance":dict(sensor.provenance),
        "physical_model":"independent_RGB_Poisson_read_analog_ADC_floatdigital_v1",
    }
    return CaptureResult(rgb.float(),variance.float(),full_well_mask|adc_mask,action,metadata)


def action_grid(exposures_s: Iterable[float], analog_gains: Iterable[float], *, digital_gain=1.) -> list[CaptureAction]:
    exposures, gains = list(exposures_s), list(analog_gains)
    if not exposures or not gains:
        raise ValueError("action grid must contain exposure times and analog gains")
    return [CaptureAction(t,g,digital_gain) for t in exposures for g in gains]
