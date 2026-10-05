"""Interchangeable local tone mapping inside the unchanged upstream pipeline."""
from pathlib import Path
from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from photofinishing.photofinishing_model import PhotofinishingModule, LuTNet
from utils.constants import CBCR_LUT_SIZE, EPS
from .config import ModelConfig
from .operators import build_operator


class _HistogramSqrt(torch.autograd.Function):
    """Exact sqrt forward; zero gradient for bins underflowed to exactly zero.

    Gaussian histogram bins can underflow in float32 for saturated colors.
    Ordinary sqrt then multiplies its infinite derivative by a zero Gaussian
    derivative. The continuous Gaussian-to-root chain tends to zero there.
    """
    @staticmethod
    def forward(ctx, histogram):
        root = histogram.sqrt()
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, gradient):
        (root,) = ctx.saved_tensors
        denominator = 2 * root.clamp_min(torch.finfo(root.dtype).tiny)
        return torch.where(root > 0, gradient / denominator, torch.zeros_like(gradient))


class _StableLuTNet(LuTNet):
    @staticmethod
    def _differentiable_cbcr_histogram(cbcr, bins=CBCR_LUT_SIZE, min_val=-.5, max_val=.5, sigma=.075):
        # Match upstream arithmetic and defaults exactly through normalization.
        b = cbcr.shape[0]
        cb = cbcr[:, 0].reshape(b, -1, 1, 1)
        cr = cbcr[:, 1].reshape(b, -1, 1, 1)
        edges = torch.linspace(min_val, max_val, bins, device=cbcr.device)
        cb_centers = edges.view(1, 1, bins, 1)
        cr_centers = edges.view(1, 1, 1, bins)
        cb_diff = (cb - cb_centers) ** 2
        cr_diff = (cr - cr_centers) ** 2
        weight = torch.exp(-(cb_diff + cr_diff) / (2 * sigma ** 2))
        histogram = weight.sum(dim=1, keepdim=True)
        histogram = histogram / (histogram.sum(dim=(2, 3), keepdim=True) + EPS)
        return _HistogramSqrt.apply(histogram)


class _StablePhotofinishingModule(PhotofinishingModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        original_lut = self._lut_net
        # Preserve initialization and RNG progression of the original backbone.
        with torch.random.fork_rng(devices=[]):
            self._lut_net = _StableLuTNet(act_func=self._act, lut_size=self._lut_size)
        self._lut_net.load_state_dict(original_lut.state_dict(), strict=True)


class TMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError('config must be a ModelConfig')
        config.validate()
        self.config = config
        self.backbone = _StablePhotofinishingModule(device=torch.device('cpu'), use_3d_lut=config.use_3d_lut)
        # Upstream stores these as ordinary tensors. Nonpersistent buffers allow
        # .to(device/dtype), retaining strict compatibility with original weights.
        for name in ('_rgb_to_ycbcr_matrix', '_ycbcr_to_rgb_matrix'):
            value = getattr(self.backbone, name)
            delattr(self.backbone, name)
            self.backbone.register_buffer(name, value, persistent=False)
        self.operator = None if config.algorithm == 'baseline' else build_operator(config)
        self.semantic_head = (nn.Conv2d(config.width, config.semantic_channels, 1)
                              if config.semantic_mode == 'train_only' else None)
        # Original local network is bypassed by candidates; do not optimize unused weights.
        if self.operator is not None:
            self.backbone._ltm_net.requires_grad_(False)
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.backbone._device = self.backbone._rgb_to_ycbcr_matrix.device
        return result

    def load_upstream(self, checkpoint, map_location='cpu'):
        """Load an original *bare* state dict strictly, rejecting partial weights."""
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TypeError('upstream checkpoint must be a path or a state dict')
        self.backbone.load_state_dict(checkpoint, strict=True)
        return self

    def checkpoint(self):
        """Model-only payload; training tooling may add optimizer/scheduler/RNG."""
        return {'config': self.config.to_dict(), 'state_dict': self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, map_location='cpu'):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=True)
        model = cls(ModelConfig.from_dict(checkpoint['config']))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model

    @staticmethod
    def _validate_image(image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[0], *image.shape[-2:]) < 1:
            raise ValueError('input must have nonempty (B,3,H,W) shape')
        if not image.is_floating_point() or not image.isfinite().all():
            raise ValueError('input must be finite floating point linear sRGB')
        if (image < 0).any() or (image > 1).any():
            raise ValueError('normalized input linear sRGB must lie in [0,1]')

    def forward(self, image, semantics=None, confidence=None):
        self._validate_image(image)
        upstream = self.backbone
        if self.operator is None:
            # Original reflection padding requires >=20 pixels in each axis.
            # Preserve exact upstream behavior at valid sizes; extend only tiny inputs.
            h, w = image.shape[-2:]
            padded = F.pad(image, (0, max(0, 20 - w), 0, max(0, 20 - h)), mode='replicate')
            result = upstream(padded, training_mode=True)
            maps = {key: value[..., :h, :w] if torch.is_tensor(value) and value.ndim == 4
                    and value.shape[-2:] == padded.shape[-2:] else value
                    for key, value in result.items() if key != 'output' and value is not None}
            return {'output': result['output'][..., :h, :w],
                    'linear': result['lsrgb_ltm'][..., :h, :w], 'maps': maps}

        gain_factor = upstream._gain_net(image)
        gain = upstream._apply_gain(image, gain_factor)
        gtm_params = upstream._gtm_net(gain)
        base = upstream._apply_gtm(gain, gtm_params)
        if self.config.semantic_mode != 'explicit':
            semantics, confidence = None, None
        rendered = self.operator(gain, base, semantics, confidence)
        linear = rendered['image']
        if linear.shape != image.shape:
            raise ValueError('operator must preserve the RGB input shape')
        chroma_input = linear
        if upstream._3d_lut is not None:
            chroma_input = upstream._apply_3d_lut_on_rgb(chroma_input, upstream._3d_lut())
        ycbcr = upstream.rgb_to_ycbcr(chroma_input.clamp(0., 1.))
        chroma_lut = upstream._lut_net(ycbcr)
        cbcr = upstream._apply_2d_lut_on_cbcr(ycbcr[:, 1:], chroma_lut)
        pre_gamma = upstream.ycbcr_to_rgb(torch.cat((ycbcr[:, :1], cbcr), 1))
        gamma = upstream._gamma_net(pre_gamma)
        # Chroma interpolation can produce negative RGB. Positive power on a
        # negative base has NaN derivatives; keep candidate rendering finite.
        output = upstream._apply_gamma(pre_gamma.clamp_min(1e-8), gamma).clamp(0., 1.)
        maps = dict(rendered['maps'])
        maps.update(gain_factor=gain_factor, gtm_params=gtm_params, gain_rgb=gain,
                    base_rgb=base, gamma_factor=gamma)
        result = {'output': output, 'linear': linear, 'maps': maps}
        if self.training and self.semantic_head is not None:
            logits = self.semantic_head(rendered['features'])
            result['semantic_logits'] = F.interpolate(logits, size=image.shape[-2:],
                                                       mode='bilinear', align_corners=False)
        return result
