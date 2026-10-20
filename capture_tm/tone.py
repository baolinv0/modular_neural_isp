"""HDR-safe engineering interpretations of Apple/Samsung AE--TM mechanisms.

These global curves are not complete patent reproductions. Source statistics
come only from the observed input. Capture bias and user intent are separate;
the resulting virtual gain is integrated once into curve construction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .modular import ModularPhotofinishingBackend, _validate_linear_rgb


def _batch_value(value, x, name):
    value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    if value.numel() == 1:
        value = value.reshape(1).expand(x.shape[0])
    elif value.numel() == x.shape[0]:
        value = value.reshape(x.shape[0])
    else:
        raise ValueError(f"{name} must be scalar or have one value per image")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _hdr_shoulder(y, gain):
    """Fixed .7 linear anchor, C1 shoulder, monotone on [0,infinity).

    No input white-point normalization: a reference lowtone has the same
    mapping across captures after compensation. Unlike a finite-domain DRC
    with D(1)=1, the HDR extension reserves headroom for legal values >1.
    """
    z = y * gain
    shoulder = 1. - .09 / (z - .4).clamp_min(.3)
    return torch.where(z <= .7, z, shoulder)


def _node_histogram(position, weights, bins):
    """Linearly deposit sample mass on neighboring nodes, preserving mass."""
    position = position.clamp(0, bins - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp_max(bins - 1)
    fraction = position - lower
    hist = torch.zeros(position.shape[0], bins, dtype=position.dtype, device=position.device)
    hist.scatter_add_(1, lower, weights * (1 - fraction))
    hist.scatter_add_(1, upper, weights * fraction)
    return hist / hist.sum(-1, keepdim=True)


def _mass_preserving_lpf(hist):
    """Normalize each input bin's truncated boundary support before LPF."""
    kernel = hist.new_tensor([.25, .5, .25]).reshape(1, 1, 3)
    support = F.conv1d(torch.ones_like(hist[:, None]), kernel, padding=1)
    return F.conv1d(hist[:, None] / support, kernel, padding=1)[:, 0]


def _inverse_cdf(cdf, probabilities):
    indices = torch.searchsorted(cdf.contiguous(), probabilities.contiguous()).clamp_max(cdf.shape[-1] - 1)
    lower = (indices - 1).clamp_min(0)
    low_p = cdf.gather(1, lower)
    low_p = torch.where(indices == 0, torch.zeros_like(low_p), low_p)
    high_p = cdf.gather(1, indices)
    fraction = ((probabilities - low_p) / (high_p - low_p).clamp_min(1e-8)).clamp(0, 1)
    return (lower + fraction * (indices - lower)) / (cdf.shape[-1] - 1)


class PatentToneMapper(nn.Module):
    """Map observed sensor-linear HDR into bounded RGB, then frozen backend.

    ``virtual_gain`` is an *additional* positive rendering request, not an
    already-applied pixel gain. Capture correction is capped first to the
    configured EV budget; intent plus additional request uses the remaining
    signed budget. ``applied_render_ev``/``virtual_gain`` are authoritative
    total values. No clean/reference target is accepted by this interface.

    Samsung uses the unscaled source CDF and a gain-scaled target histogram.
    ``Z=min(G*Y,1)`` clips statistics only, never the source pixels. This
    explicit coordinate definition avoids the cutoff ambiguity in the
    patent's printed Eq. (2). A 20% fixed-anchor HDR curve regularizes empty
    bins/atoms and ensures HDR highlights do not collapse. Histogram LPF
    preserves probability mass; it does not claim exact mean preservation.
    A distribution spanning no more than the first log-input bin width is
    unresolved: use the fixed-anchor curve directly, keeping histograms as
    diagnostics. This avoids atomic-CDF/LPF brightness offsets in flat scenes.
    """

    def __init__(self, mode="apple", backend=None, *, curve_bins=256, max_render_ev=4.):
        super().__init__()
        if mode not in ("apple", "samsung"):
            raise ValueError("mode must be 'apple' or 'samsung'")
        if isinstance(curve_bins, bool) or not isinstance(curve_bins, int) or curve_bins < 8:
            raise ValueError("curve_bins must be an integer >=8")
        if not math.isfinite(max_render_ev) or not 0 < max_render_ev <= 32:
            raise ValueError("max_render_ev must be finite in (0,32]")
        self.mode = mode
        self.curve_bins = curve_bins
        self.max_render_ev = float(max_render_ev)
        self.backend = ModularPhotofinishingBackend() if backend is None else backend
        if not isinstance(self.backend, nn.Module):
            raise TypeError("backend must be a torch nn.Module")
        self.backend.requires_grad_(False)
        self.backend.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backend.eval()
        return self

    def forward(self, x, capture_bias_ev=0., render_intent_ev=0., *,
                virtual_gain=None, subject_mask=None):
        _validate_linear_rgb(x, bounded=False)
        bias = _batch_value(capture_bias_ev, x, "capture_bias_ev")
        intent = _batch_value(render_intent_ev, x, "render_intent_ev")
        extra = _batch_value(1. if virtual_gain is None else virtual_gain, x, "virtual_gain")
        if (extra <= 0).any():
            raise ValueError("virtual_gain must be positive")
        request_capture = -bias
        compensation = request_capture.clamp(-self.max_render_ev, self.max_render_ev)
        request_style = intent + torch.log2(extra)
        requested = request_capture + request_style
        applied = (compensation + request_style).clamp(-self.max_render_ev, self.max_render_ev)
        gain = torch.exp2(applied)
        y = x.amax(dim=1, keepdim=True)
        maximum = y.flatten(1).amax(1).clamp_min(1.)
        log_max = torch.log1p(maximum)
        nodes = torch.linspace(0, 1, self.curve_bins, dtype=x.dtype, device=x.device)[None]
        curve_input = torch.expm1(nodes * log_max[:, None])
        curve_input[:, -1] = maximum
        curve = _hdr_shoulder(curve_input, gain[:, None])
        weights = torch.ones_like(y)
        if subject_mask is not None:
            mask = torch.as_tensor(subject_mask, dtype=x.dtype, device=x.device)
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            if mask.shape == (1, 1, x.shape[2], x.shape[3]):
                mask = mask.expand(x.shape[0], -1, -1, -1)
            if mask.shape != y.shape or not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
                raise ValueError("subject_mask must be finite [B,1,H,W] in [0,1]")
            weights = weights + 2 * mask
        diagnostics = {}
        if self.mode == "apple":
            mapped_y = _hdr_shoulder(y, gain[:, None, None, None])
        else:
            base_curve = curve
            observed_range = y.flatten(1).amax(1) - y.flatten(1).amin(1)
            first_bin_width = curve_input[:, 1] - curve_input[:, 0]
            degenerate = observed_range <= first_bin_width
            position = torch.log1p(y.flatten(1)) / log_max[:, None] * (self.curve_bins - 1)
            source = _node_histogram(position, weights.flatten(1), self.curve_bins)
            # G acts on the target distribution only; HR input is untouched.
            target_position = (y.flatten(1) * gain[:, None]).clamp(0, 1) * (self.curve_bins - 1)
            scaled = _node_histogram(target_position, weights.flatten(1), self.curve_bins)
            target = _mass_preserving_lpf(scaled)
            target = target / target.sum(-1, keepdim=True)
            # Keep the true target support: adding uniform epsilon mass would
            # make source rank 1 map to white even for an entirely dark scene.
            target_cdf = target.cumsum(-1)
            target_cdf = target_cdf / target_cdf[:, -1:]
            source_cdf = source.cumsum(-1)
            source_cdf = source_cdf / source_cdf[:, -1:]
            matched = _inverse_cdf(target_cdf, source_cdf)
            curve = .8 * matched + .2 * curve
            curve[:, 0] = 0.
            curve = torch.where(degenerate[:, None], base_curve, curve)
            lower = position.floor().long().clamp(0, self.curve_bins - 1)
            upper = (lower + 1).clamp_max(self.curve_bins - 1)
            fraction = (position - lower).clamp(0, 1)
            mapped = curve.gather(1, lower) * (1 - fraction) + curve.gather(1, upper) * fraction
            mapped_y = mapped.reshape_as(y)
            # Evaluate the analytic fallback directly, rather than interpolating
            # it in log coordinates and introducing another lowtone bias.
            mapped_y = torch.where(degenerate[:, None, None, None],
                                   _hdr_shoulder(y, gain[:, None, None, None]), mapped_y)
            diagnostics = {"source_histogram": source, "scaled_histogram": scaled,
                           "target_histogram": target,
                           "degenerate_distribution_mask": degenerate}
        # Common RGB gain preserves channel ratios; max-RGB controls gamut.
        pre_backend = torch.where(y > 0, x / y.clamp_min(torch.finfo(x.dtype).tiny) * mapped_y,
                                  torch.zeros_like(x)).clamp(0, 1)
        output = self.backend(pre_backend)
        if not isinstance(output, torch.Tensor) or output.shape != x.shape:
            raise RuntimeError("backend must return RGB with the input shape")
        if not torch.isfinite(output).all() or ((output < 0) | (output > 1)).any():
            raise RuntimeError("backend must return finite display RGB in [0,1]")
        return {"output": output, "pre_backend": pre_backend, "tone_curve": curve,
                "tone_curve_input": curve_input, "virtual_gain": gain,
                "capture_compensation_ev": compensation,
                "requested_capture_compensation_ev": request_capture,
                "requested_render_ev": requested, "applied_render_ev": applied,
                **diagnostics}
