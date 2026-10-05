"""Differentiable shared rendering helpers; all images use BCHW layout."""
import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


def luminance(rgb):
    """Rec.709 luminance of linear RGB, with a retained scalar channel."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('RGB must have shape (B,3,H,W)')
    return (rgb * rgb.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)


def apply_luminance(rgb, new_y, eps=1e-6):
    """Preserve chromaticity; map near-black colors through a neutral fallback."""
    y = luminance(rgb)
    if new_y.shape != y.shape:
        raise ValueError('new luminance must have shape (B,1,H,W)')
    scaled = rgb * (new_y / y.clamp_min(eps))
    return torch.where(y > eps, scaled, new_y.expand_as(rgb))


def monotonic_curves(logits):
    """(...,N-1) logits -> (...,N) increasing knots with exact 0/1 ends."""
    if logits.shape[-1] < 1:
        raise ValueError('curves require at least one increment')
    increments = F.softplus(logits) + 1e-6
    partial = increments.cumsum(-1)
    knots = partial[..., :-1] / partial[..., -1:]
    return torch.cat((torch.zeros_like(partial[..., :1]), knots,
                      torch.ones_like(partial[..., :1])), -1)


def evaluate_curves(values, curves):
    """BCHW values and BKN curves -> BKC HW; normalized input is clamped to [0,1]."""
    if values.ndim != 4 or curves.ndim != 3 or values.shape[0] != curves.shape[0] or curves.shape[-1] < 2:
        raise ValueError('expected BCHW values and (B,K,N>=2) curves')
    b, c, h, w = values.shape
    k, n = curves.shape[1:]
    position = values.clamp(0, 1).reshape(b, 1, -1).expand(b, k, -1) * (n - 1)
    lower = position.floor().long().clamp(max=n - 2)
    fraction = position - lower.to(position.dtype)
    y0, y1 = curves.gather(2, lower), curves.gather(2, lower + 1)
    return (y0 + fraction * (y1 - y0)).reshape(b, k, c, h, w)


def _box_filter(x, radius):
    if radius == 0:
        return x
    return F.avg_pool2d(F.pad(x, (radius,) * 4, mode='replicate'), 2 * radius + 1, stride=1)


def guided_upsample(low, guide, radius=3, eps=.01):
    """Fast scalar-guide filtering: BCHW low + B1HW guide -> full-resolution BCHW.

    Fit local linear coefficients at low resolution, then interpolate coefficients
    and apply them to the original guide. Replicated borders support singleton axes.
    """
    if low.ndim != 4 or guide.ndim != 4 or guide.shape[1] != 1 or low.shape[0] != guide.shape[0]:
        raise ValueError('expected BCHW low and B1HW guide with matching batch')
    if type(radius) is not int or radius < 0 or eps <= 0:
        raise ValueError('radius must be nonnegative integer; eps must be positive')
    small_guide = F.interpolate(guide, size=low.shape[-2:], mode='bilinear', align_corners=False)
    mean_i, mean_p = _box_filter(small_guide, radius), _box_filter(low, radius)
    # Center the response before covariance to avoid cancellation for constant maps.
    centered_p = low - low[..., :1, :1]
    covariance = (_box_filter(small_guide * centered_p, radius)
                  - mean_i * _box_filter(centered_p, radius))
    variance = (_box_filter(small_guide.square(), radius) - mean_i.square()).clamp_min(0)
    a = covariance / (variance + eps)
    b = mean_p - a * mean_i
    a, b = _box_filter(a, radius), _box_filter(b, radius)
    a = F.interpolate(a, size=guide.shape[-2:], mode='bilinear', align_corners=False)
    b = F.interpolate(b, size=guide.shape[-2:], mode='bilinear', align_corners=False)
    return a * guide + b


def tone_curve(x, a, b, c):
    """Stable upstream parametric tone curve, preserving its numerical formula."""
    from photofinishing.photofinishing_model import PhotofinishingModule
    # Use the stable arithmetic directly so MPS never detaches training gradients.
    eps = 1e-8
    x = x.clamp(eps, 1. - eps)
    xa = PhotofinishingModule.safe_pow(x, a)
    return xa / (xa + PhotofinishingModule.safe_pow(c * (1. - x).clamp(eps, 1.), b) + eps)


def total_variation(x):
    """Sum of mean absolute differences on available spatial axes, scalar."""
    out = x.sum() * 0
    if x.shape[-2] > 1:
        out = out + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    if x.shape[-1] > 1:
        out = out + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return out


class ConditionEncoder(nn.Module):
    """Shared RGB/semantic features; confidence gates every semantic channel."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(nn.Conv2d(6 + config.semantic_channels, config.width, 3, padding=1),
                                 nn.SiLU(), nn.Conv2d(config.width, config.width, 3, padding=1), nn.SiLU())

    @staticmethod
    def resize_analysis(value, size):
        """Reduce rendered images, never exchange nonlinear TM with downsampling."""
        if value.shape[-2:] == tuple(size):
            return value
        if all(source >= target for source, target in zip(value.shape[-2:], size)):
            return F.interpolate(value, size=size, mode='area')
        return F.interpolate(value, size=size, mode='bilinear', align_corners=False, antialias=True)

    def semantic_inputs(self, gain, semantics=None, confidence=None):
        """Validate separate semantic resolution; gate probabilities before reducing.

        Confidence is either shared B1HW or per-semantic-channel BCHW. Missing
        confidence means trusted labels; missing semantics means absent labels.
        S0 and S1 intentionally ignore supplied inference conditioning.
        """
        if self.config.semantic_mode != 'explicit':
            return None
        b = gain.shape[0]
        for name, value, channels in [('semantics', semantics, (self.config.semantic_channels,)),
                                       ('confidence', confidence, (1, self.config.semantic_channels))]:
            if value is None:
                continue
            if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] != b or
                value.shape[1] not in channels or min(value.shape[-2:]) < 1):
                raise ValueError(f'{name} must have matching batch, valid channels and nonempty spatial dimensions')
            if not value.is_floating_point() or not value.isfinite().all() or (value < 0).any() or (value > 1).any():
                raise ValueError(f'{name} must be finite floating probabilities in [0,1]')
        if semantics is None:
            return None
        if confidence is not None and confidence.shape[-2:] != semantics.shape[-2:]:
            raise ValueError('confidence and semantics must have matching spatial resolution')
        masks = semantics.to(gain)
        return masks if confidence is None else masks * confidence.to(gain)

    def forward(self, gain, base, semantics=None, confidence=None):
        if gain.ndim != 4 or gain.shape[1] != 3 or base.shape != gain.shape:
            raise ValueError('gain/base must have identical (B,3,H,W) shapes')
        if not gain.is_floating_point() or not base.is_floating_point() or not gain.isfinite().all() or not base.isfinite().all():
            raise ValueError('gain/base must be finite floating RGB')
        size = (self.config.analysis_size,) * 2
        masks = self.semantic_inputs(gain, semantics, confidence)
        # Resize each input first. In particular no full-resolution 9-channel
        # concatenation or full-size allocation of missing semantic channels.
        gain_small = self.resize_analysis(gain, size)
        base_small = self.resize_analysis(base.to(gain), size)
        masks_small = (gain.new_zeros(gain.shape[0], self.config.semantic_channels, *size)
                       if masks is None else self.resize_analysis(masks, size))
        return self.net(torch.cat((gain_small, base_small, masks_small), 1))
