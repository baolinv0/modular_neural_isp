"""Causal capture protocols and a fixed, observation-only HDR fusion baseline.

``fixed_fuse`` returns common-reference *sensor* radiance. ``render_capture``
applies the fixed WB/CCM front end once. No simulator variance, latent image,
oracle saturation mask, or future preview enters the inference decisions.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .simulation import capture
from .types import CaptureAction, CaptureResult, Scene, SensorProfile

CAPTURE_WINDOW_S = (-.05, .05)
MAX_SHUTTER_S = 1 / 30
HDR_CENTERS_S = (-1 / 30, 0., 1 / 30)


@dataclass(frozen=True)
class CapturePlan:
    actions: tuple[CaptureAction, ...]
    centers_s: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "centers_s", tuple(float(t) for t in self.centers_s))
        if len(self.actions) not in (1, 3) or len(self.actions) != len(self.centers_s):
            raise ValueError("a capture plan needs one or three matching actions and centers")
        if not all(isinstance(a, CaptureAction) for a in self.actions):
            raise ValueError("plan actions must be CaptureAction values")
        if not all(math.isfinite(t) for t in self.centers_s):
            raise ValueError("plan centers must be finite")
        if any(self.centers_s[i] + self.actions[i].exposure_s / 2 >
               self.centers_s[i + 1] - self.actions[i + 1].exposure_s / 2 + 1e-12
               for i in range(len(self.actions) - 1)):
            raise ValueError("ordered capture shutter intervals must not overlap")


def build_plan_bank(scheme: str, sensor: SensorProfile) -> list[CapturePlan]:
    """Legal alternatives within a shared fixed 100 ms capture window.

    The bank fixes digital gain to one. Equal-EV shutter/analog-gain choices
    retain the photon/motion tradeoff. HDR includes twelve (-2, 0, +2) EV
    brackets and twelve narrow/asymmetric alternatives so a learned policy can
    optimize the bracket shape. Every plan has fixed slot centers, including
    reference time zero for every policy.
    Actual open-shutter durations vary; the allowed scheduling budget does not.
    """
    if scheme not in {"apple", "samsung"}:
        raise ValueError("scheme must be apple or samsung")
    plans = []
    # Sensor reference shutter may vary, while physical slots remain 30 Hz.
    base = min(sensor.reference_exposure_s, 1 / 120)
    factors = (.125, .5, 1., 4.) if scheme == "apple" else (.125, .25, .5, 1.)
    for factor in factors:
        for gain in (1., 2., 4.):
            center = base * factor
            if scheme == "apple":
                plans.append(CapturePlan((CaptureAction(center, gain),), (0.,)))
            else:
                plans.append(CapturePlan(tuple(CaptureAction(center * f, gain) for f in (.25, 1., 4.)),
                                         HDR_CENTERS_S))
    if scheme == "samsung":
        for factor in (.25, 1.):
            for gain in (1., 2.):
                for bracket in ((.25, 1., 2.), (.5, 1., 4.), (.5, 1., 2.)):
                    plans.append(CapturePlan(tuple(CaptureAction(base * factor * f, gain) for f in bracket),
                                             HDR_CENTERS_S))
    return plans


def plan_features(plans: list[CapturePlan], sensor: SensorProfile) -> Tensor:
    if not plans or len({len(p.actions) for p in plans}) != 1:
        raise ValueError("plan bank must be nonempty with a common frame count")
    return torch.tensor([[[math.log2(a.exposure_s / sensor.reference_exposure_s),
                           math.log2(a.analog_gain), math.log2(a.digital_gain)]
                          for a in p.actions] for p in plans], dtype=torch.float32)


def fixed_frontend(rgb: Tensor, sensor: SensorProfile) -> Tensor:
    """Fixed WB/CCM; retain HDR values and clamp only negative output colors."""
    wb = rgb.new_tensor(sensor.wb).reshape(3, 1, 1)
    ccm = rgb.new_tensor(sensor.ccm)
    return torch.einsum("ij,jhw->ihw", ccm, rgb * wb).clamp_min(0)


def observe_previews(scene: Scene, sensor: SensorProfile, seed: int = 0) -> tuple[Tensor, Tensor, dict]:
    """Three noisy, strictly historical observations with effective metadata.

    Preview radiometry is black-subtracted sensor RGB clipped to [0,1]. A
    calibrated fixed reference shutter is used, with no hidden auto-brightening.
    Dynamic scenes must provide real temporal support; the simulator raises if
    a requested preview is outside it. Static singleton scenes need no padding.
    """
    action = CaptureAction(min(sensor.reference_exposure_s, 1 / 120))
    centers = (-4 / 30, -3 / 30, -2 / 30)
    results = [capture(scene, action, sensor, center_s=t, seed=seed + i * 104729)
               for i, t in enumerate(centers)]
    previews = torch.stack([r.rgb.clamp(0, 1) for r in results])
    state = previews.new_tensor([[math.log2(action.exposure_s / sensor.reference_exposure_s), 0., 0.]]).repeat(3, 1)
    return previews, state, {"centers_s": list(centers),
                             "shutter_ends_s": [t + action.exposure_s / 2 for t in centers],
                             "action": action.to_dict(), "domain": "bounded_sensor_linear_rgb",
                             "causal": True, "noise_seed": int(seed)}


def rule_plan_index(previews: Tensor, state: Tensor, plans: list[CapturePlan], sensor: SensorProfile) -> int:
    """Frozen scene-adaptive meter; HDR changes center with a fixed bracket.

    Median scene luminance meters toward 18% while a high quantile limits
    highlights. Visible preview motion breaks equal-EV ties toward short
    shutters; otherwise the rule prefers photon collection over analog gain.
    """
    if previews.ndim != 4 or previews.shape[:2] != (3, 3) or state.shape != (3, 3):
        raise ValueError("expected previews [3,3,H,W] and physical state [3,3]")
    if not torch.isfinite(previews).all() or not torch.isfinite(state).all():
        raise ValueError("previews and metadata must be finite")
    if not plans:
        raise ValueError("plan bank cannot be empty")
    radiance = previews.clamp_min(0) / torch.exp2(state.sum(-1))[:, None, None, None]
    y = radiance[-1].mean(0)
    median = float(torch.quantile(y, .5).clamp_min(1e-4))
    high = float(torch.quantile(y, .99).clamp_min(1e-4))
    desired = .18 / median
    shortest_bracket = .25 if len(plans[0].actions) == 3 else 1.
    highlight_safe = .92 / (high * shortest_bracket)
    desired = min(desired, max(highlight_safe, desired / 4))
    desired_ev = math.log2(desired)
    relative_motion = float((radiance[1:] - radiance[:-1]).abs().mean() /
                            (radiance.mean() + .02))
    moving = relative_motion > .08
    ref = CaptureAction(sensor.reference_exposure_s)
    costs = []
    for plan in plans:
        middle = plan.actions[len(plan.actions) // 2]
        # The frozen HDR control changes its center only. Learned AE may choose
        # any bank plan, including the narrow/asymmetric bracket alternatives.
        if len(plan.actions) == 3 and not (
                math.isclose(plan.actions[0].scale(middle), .25) and
                math.isclose(plan.actions[2].scale(middle), 4.)):
            costs.append(float("inf"))
            continue
        ev_error = abs(math.log2(middle.scale(ref)) - desired_ev)
        shutter_ev = math.log2(middle.exposure_s / sensor.reference_exposure_s)
        costs.append(ev_error + (.025 if moving else -.025) * shutter_ev)
    if not any(math.isfinite(c) for c in costs):
        raise ValueError("rule HDR meter requires at least one fixed (-2,0,+2) bracket")
    return min(range(len(costs)), key=costs.__getitem__)


def _observed_statistics(result: CaptureResult, sensor: SensorProfile) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate uncensored variance from measurements and calibrated constants."""
    action, rgb = result.action, result.rgb
    adc_range = (2 ** sensor.bit_depth - 1) - sensor.black_level_dn
    # A measured electron count substitutes for unknown expected electrons.
    measured_e = (rgb / (action.analog_gain * action.digital_gain)).clamp_min(0) * sensor.full_well_e
    variance = ((measured_e + sensor.read_noise_e ** 2) * (action.analog_gain / sensor.full_well_e) ** 2 +
                (sensor.adc_noise_dn ** 2 + 1 / 12) / adc_range ** 2) * action.digital_gain ** 2
    ceiling = min(1., action.analog_gain) * action.digital_gain
    saturated = rgb >= ceiling * .995
    useful = (~saturated) & (rgb > variance.sqrt() * 2)
    return variance.clamp_min(1e-12), saturated, useful


def _shift(x: Tensor, dy: int, dx: int) -> tuple[Tensor, Tensor]:
    moved = torch.roll(x, (dy, dx), (-2, -1))
    valid = torch.ones((1, *x.shape[-2:]), dtype=torch.bool, device=x.device)
    if dy > 0:
        valid[:, :dy] = False
    elif dy < 0:
        valid[:, dy:] = False
    if dx > 0:
        valid[:, :, :dx] = False
    elif dx < 0:
        valid[:, :, dx:] = False
    return moved, valid


def _translation(image: Tensor, reference: Tensor, usable: Tensor, ref_usable: Tensor,
                 max_translation: int) -> tuple[int, int]:
    """Coarse bounded global translation from robust exposure-normalized RGB."""
    if max_translation == 0:
        return 0, 0
    h, w = image.shape[-2:]
    scale = max(1, math.ceil(max(h, w) / 32))
    size = (max(1, h // scale), max(1, w // scale))
    a = F.interpolate(torch.log1p(image.clamp_min(0)).mean(0)[None, None], size=size, mode="area")[0]
    b = F.interpolate(torch.log1p(reference.clamp_min(0)).mean(0)[None, None], size=size, mode="area")[0]
    am = F.interpolate(usable.all(0).float()[None, None], size=size, mode="area")[0] > .99
    bm = F.interpolate(ref_usable.all(0).float()[None, None], size=size, mode="area")[0] > .99
    radius = max(1, math.ceil(max_translation / scale))
    best, shift = float("inf"), (0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            moved, border = _shift(a, dy, dx)
            moved_mask, _ = _shift(am, dy, dx)
            mask = moved_mask & bm & border
            if int(mask.sum()) < max(4, int(a.numel() * .15)):
                continue
            # Trim a minority of independent moving objects, keeping a global
            # camera-motion model. Prefer zero translation when equally good.
            errors = (moved - b).abs()[mask]
            score = float(errors.clamp_max(torch.quantile(errors, .8)).mean()) + .0001 * (abs(dy) + abs(dx))
            if score < best:
                best, shift = score, (dy * scale, dx * scale)
    return tuple(max(-max_translation, min(max_translation, s)) for s in shift)


def fixed_fuse(captures: list[CaptureResult], sensor: SensorProfile, *, reference_index: int = 1,
               max_translation: int = 4) -> dict:
    """Fixed F0: normalize, align, reject moving pixels, precision fuse.

    Inputs are ordered physical measurements. The middle exposure defines
    geometry. Registration is global integer translation; local motion uses
    conservative reference fallback rather than synthesized details. Occluded
    clipped reference regions cannot be reliably disambiguated by this F0.
    """
    if len(captures) != 3 or reference_index not in range(3):
        raise ValueError("HDR fusion requires three frames and a legal reference index")
    if max_translation < 0 or int(max_translation) != max_translation:
        raise ValueError("max_translation must be a nonnegative integer")
    if any(c.rgb.shape != captures[0].rgb.shape for c in captures):
        raise ValueError("fusion inputs must have the same shape")
    ref_action = CaptureAction(sensor.reference_exposure_s)
    radiances, variances, saturated, useful = [], [], [], []
    for result in captures:
        variance, saturation, information = _observed_statistics(result, sensor)
        scale = result.action.scale(ref_action)
        radiances.append(result.rgb / scale)
        variances.append(variance / scale ** 2)
        saturated.append(saturation)
        useful.append(information)
    reference = radiances[reference_index]
    ref_var = variances[reference_index]
    ref_good = ~saturated[reference_index]
    aligned, aligned_var, good, information = [], [], [], []
    shifts, rejected = [], []
    for i in range(3):
        dy, dx = (0, 0) if i == reference_index else _translation(
            radiances[i], reference, ~saturated[i], ref_good, max_translation)
        im, border = _shift(radiances[i], dy, dx)
        variance, _ = _shift(variances[i], dy, dx)
        bad, _ = _shift(saturated[i], dy, dx)
        info, _ = _shift(useful[i], dy, dx)
        valid = (~bad) & border
        if i == reference_index:
            reject = torch.zeros_like(valid)
        else:
            tolerance = 5 * (variance + ref_var).sqrt() + .015 + .08 * torch.maximum(im.abs(), reference.abs())
            # Shared spatial rejection avoids combining different object colors.
            mismatch = ((im - reference).abs() > tolerance) & ref_good & valid
            reject = mismatch.any(0, keepdim=True).expand_as(valid)
        aligned.append(im)
        aligned_var.append(variance)
        good.append(valid & ~reject)
        information.append(info & valid & ~reject)
        shifts.append([dy, dx])
        rejected.append(reject)
    images, variance = torch.stack(aligned), torch.stack(aligned_var)
    valid = torch.stack(good)
    weights = valid / variance.clamp_min(1e-12)
    precision = weights.sum(0)
    fused = (weights * images).sum(0) / precision.clamp_min(1e-12)
    # All-saturated measurements provide lower bounds only. Preserve the largest
    # bound for display, while marking this information as missing explicitly.
    fallback = torch.maximum(reference, images.max(0).values)
    fused = torch.where(precision > 0, fused, fallback)
    missing = ~torch.stack(information).any(0)
    snr = fused.clamp_min(0) * precision.sqrt()
    reliability = ((snr / (snr + 3)) * (~missing)).mean(0, keepdim=True).clamp(0, 1)
    return {"image": fused.clamp_min(0), "reliability": reliability, "missing": missing,
            "capture_ev": 0., "metadata": {
                "domain": "sensor_linear_relative_radiance", "reference_index": reference_index,
                "reference_center_s": float(captures[reference_index].metadata.get("center_s", 0.)),
                "normalization_count": 1, "alignment_shifts_yx": shifts,
                "fusion": "observed_translation_rejection_inverse_variance_v1",
                "motion_rejection_fraction": float(torch.stack(rejected).float().mean()),
                "missing_all_fraction": float(missing.float().mean()),
                "noise_estimation": "observed_signal_sensor_profile", "oracle_fields_used": False}}


def compose_captures(captures: list[CaptureResult], sensor: SensorProfile) -> dict:
    """Finish actual captured data using its effective actions and timestamps.

    The caller supplies black-subtracted sensor-linear RGB and effective
    physical metadata. No scene object or target is accepted on this path.
    Legacy CaptureResult variance/saturation fields are intentionally unused.
    """
    results = list(captures)
    if len(results) not in (1, 3):
        raise ValueError("composition requires one or three captured frames")
    for result in results:
        if result.rgb.ndim != 3 or result.rgb.shape[0] != 3 or not torch.isfinite(result.rgb).all():
            raise ValueError("actual RGB measurements must be finite [3,H,W] tensors")
        if not isinstance(result.action, CaptureAction):
            raise ValueError("actual captures need an effective physical action")
        if "center_s" not in result.metadata or not math.isfinite(float(result.metadata["center_s"])):
            raise ValueError("actual captures need a finite effective center_s timestamp")
    centers = tuple(float(r.metadata["center_s"]) for r in results)
    CapturePlan(tuple(r.action for r in results), centers)  # Reject overlap/out-of-order readback.
    if len(results) == 3:
        output = fixed_fuse(results, sensor)
        output["image"] = fixed_frontend(output["image"], sensor)
    else:
        result = results[0]
        variance, saturated, information = _observed_statistics(result, sensor)
        snr = result.rgb.clamp_min(0) / variance.sqrt()
        reliability = ((snr / (snr + 3)) * (~saturated)).mean(0, keepdim=True).clamp(0, 1)
        output = {"image": fixed_frontend(result.rgb, sensor), "reliability": reliability,
                  "missing": ~information,
                  "capture_ev": math.log2(result.action.scale(CaptureAction(sensor.reference_exposure_s))),
                  "metadata": {"domain": "linear_rgb_capture_units", "reference_center_s": centers[0],
                               "normalization_count": 0, "missing_all_fraction": float((~information).float().mean()),
                               "noise_estimation": "observed_signal_sensor_profile", "oracle_fields_used": False}}
    output["metadata"].update({"frame_count": len(results), "centers_s": list(centers),
                               "actions": [r.action.to_dict() for r in results],
                               "fixed_frontend": "WB_CCM_nonnegative_no_highlight_clip"})
    return output


def render_capture(scene: Scene, plan: CapturePlan, sensor: SensorProfile, seed: int = 0,
                   noisy: bool = True) -> dict:
    results = [capture(scene, action, sensor, center_s=center, seed=seed + i * 104729, noisy=noisy)
               for i, (action, center) in enumerate(zip(plan.actions, plan.centers_s))]
    return compose_captures(results, sensor)
