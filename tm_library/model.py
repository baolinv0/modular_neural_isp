"""Interchangeable local tone mapping inside the unchanged upstream pipeline."""
from pathlib import Path
from dataclasses import replace
from numbers import Real
import math
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
    """Version-two composable tone mapper for normalized linear sRGB.

    Stages are ordinary synchronous Python boundaries, not a hardware executor:
    ``context = prepare_global(image)`` contains original image, gain-adjusted
    RGB and fully rendered GTM. ``predict_controls(context, ..., stage='base')``
    predicts local controls; ``render(context, controls)`` returns pre-chroma RGB.
    A correction is predicted only after that RGB anchor exists, by passing it to
    both correction stages. ``finish(rendered['image'], rendered['controls'])``
    applies downstream color/gamma, using cached original-reference controls in
    fixed_reference mode. Returned render controls carry their context for finish.
    No stage substitutes TM(downsample(image)) for downsample(TM(image)).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError('config must be a ModelConfig')
        config.validate()
        self.config = config
        self.backbone = _StablePhotofinishingModule(device=torch.device('cpu'), use_3d_lut=config.use_3d_lut)
        for name in ('_rgb_to_ycbcr_matrix', '_ycbcr_to_rgb_matrix'):
            value = getattr(self.backbone, name)
            delattr(self.backbone, name)
            self.backbone.register_buffer(name, value, persistent=False)
        self.operator = (None if config.effective_base in ('baseline', 'gtm') else
                         build_operator(replace(config, algorithm=config.effective_base, base_algorithm=None)))
        self.correction_operator = (None if config.correction_algorithm == 'none' else
            build_operator(replace(config, algorithm=config.correction_algorithm, base_algorithm=None)))
        self.semantic_head = (nn.Conv2d(config.width, config.semantic_channels, 1)
                              if config.semantic_mode == 'train_only' else None)
        if config.effective_base != 'baseline':
            self.backbone._ltm_net.requires_grad_(False)
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.backbone._device = self.backbone._rgb_to_ycbcr_matrix.device
        return result

    def load_upstream(self, checkpoint, map_location='cpu'):
        """Load original bare tensor weights strictly, never a library payload."""
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TypeError('upstream checkpoint must be a path or a state dict')
        self.backbone.load_state_dict(checkpoint, strict=True)
        return self

    def checkpoint(self):
        return {'format': 'tm_library_v2', 'architecture_version': 2,
                'config': self.config.to_dict(), 'state_dict': self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, map_location='cpu'):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=True)
        if (not isinstance(checkpoint, Mapping) or checkpoint.get('format') != 'tm_library_v2' or
            checkpoint.get('architecture_version') != 2 or
            not isinstance(checkpoint.get('config'), Mapping) or
            checkpoint['config'].get('architecture_version') != 2):
            raise ValueError('Unsupported v1/unknown library checkpoint: v2 changes A/B/E architecture; '
                             'retrain or use an explicit migration. Original bare weights use load_upstream().')
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

    @staticmethod
    def _crop(value, shape):
        return value[..., :shape[0], :shape[1]]

    def prepare_global(self, image):
        """Return original RGB, gain RGB, GTM RGB and their upstream controls.

        Original LTM needs reflection-padding-safe dimensions. Its baseline
        compatibility path extends tiny inputs by replication before global TM.
        Full-size original inputs use unchanged upstream arithmetic.
        """
        self._validate_image(image)
        h, w = image.shape[-2:]
        work = (F.pad(image, (0, max(0, 20 - w), 0, max(0, 20 - h)), mode='replicate')
                if self.config.effective_base == 'baseline' else image)
        gain_factor = self.backbone._gain_net(work)
        gain = self.backbone._apply_gain(work, gain_factor)
        gtm_params = self.backbone._gtm_net(gain)
        base = self.backbone._apply_gtm(gain, gtm_params)
        context = {'image': image, 'gain': self._crop(gain, (h, w)),
                   'base': self._crop(base, (h, w)), 'gain_factor': gain_factor,
                   'gtm_params': gtm_params, '_work_gain': gain, '_work_base': base}
        if self.config.post_mode == 'fixed_reference':
            # Reference always means ORIGINAL LTM. Backbone is frozen by config,
            # and reference prediction does not connect the candidate's graph.
            with torch.no_grad():
                reference = self._original_reference(context)
                context['post_controls'] = self._predict_post(reference)
        return context

    def _original_reference(self, context):
        gain, base = context['_work_gain'], context['_work_base']
        h, w = gain.shape[-2:]
        if min(h, w) < 20:
            # For a candidate with tiny images compute the original reference
            # from the padded INPUT, not from padded nonlinear gain/GTM outputs.
            image = context['image']
            image = F.pad(image, (0, max(0, 20 - w), 0, max(0, 20 - h)), mode='replicate')
            gain = self.backbone._apply_gain(image, self.backbone._gain_net(image))
            base = self.backbone._apply_gtm(gain, self.backbone._gtm_net(gain))
        params = self.backbone._ltm_net(gain, base, training_mode=True)
        linear = self.backbone._apply_ltm(gain, base, params)
        return linear

    def predict_controls(self, context, semantics=None, confidence=None, stage='base', anchor=None):
        """Predict flat low-resolution operator controls with retained features.

        Correction conditioning uses its selected anchor as gain AND base.
        Semantics/confidence may have a separate shared low spatial resolution.
        """
        if stage not in ('base', 'correction'):
            raise ValueError('stage must be base or correction')
        if self.config.semantic_mode != 'explicit':
            semantics, confidence = None, None
        if stage == 'base':
            if self.operator is not None:
                controls = self.operator.predict(context['gain'], context['base'], semantics, confidence)
            elif self.config.effective_base == 'baseline':
                controls = {'ltm_params': self.backbone._ltm_net(context['_work_gain'], context['_work_base'], training_mode=True),
                            'features': None}
            else:
                controls = {'features': None}
        else:
            if self.correction_operator is None:
                raise ValueError('no correction operator configured')
            if anchor is None or anchor.shape != context['image'].shape:
                raise ValueError('correction prediction requires selected pre-chroma RGB anchor')
            controls = self.correction_operator.predict(anchor, anchor, semantics, confidence)
            if self.config.correction_semantic_channel is not None:
                masks = self.correction_operator.encoder.semantic_inputs(anchor, semantics, confidence)
                controls['_semantic_gate'] = (anchor.new_zeros(anchor.shape[0], 1, *anchor.shape[-2:]) if masks is None else
                    F.interpolate(masks[:, self.config.correction_semantic_channel:self.config.correction_semantic_channel + 1],
                                  size=anchor.shape[-2:], mode='bilinear', align_corners=False))
        return {**controls, '_context': context, '_stage': stage}

    def _correction_gate(self, anchor, controls, correction_mask, correction_strength):
        strength = self.config.correction_strength if correction_strength is None else correction_strength
        if (isinstance(strength, bool) or not isinstance(strength, Real) or
            not math.isfinite(strength) or not 0 <= strength <= 1):
            raise ValueError('correction_strength must be finite numeric in [0,1]')
        gate = anchor.new_full((anchor.shape[0], 1, *anchor.shape[-2:]), float(strength))
        if correction_mask is not None:
            if (not torch.is_tensor(correction_mask) or correction_mask.shape != gate.shape or
                not correction_mask.is_floating_point() or not correction_mask.isfinite().all() or
                (correction_mask < 0).any() or (correction_mask > 1).any()):
                raise ValueError('correction_mask must be finite probabilities with shape (B,1,H,W)')
            gate = gate * correction_mask.to(anchor)
        if '_semantic_gate' in controls:
            gate = gate * controls['_semantic_gate']
        return gate

    def render(self, context, controls, stage='base', anchor=None, correction_mask=None,
               correction_strength=None, return_maps=True):
        """Render pre-chroma RGB; zero correction controls preserve anchor exactly.

        A uses a bounded additive candidate-minus-anchor delta. C multiplies the
        scalar EV by the user/semantic gate BEFORE exp2, preserving chromaticity.
        Diagnostic tensors are returned only when return_maps is true.
        """
        if stage not in ('base', 'correction'):
            raise ValueError('stage must be base or correction')
        if stage == 'base':
            if self.operator is not None:
                result = self.operator.render(context['gain'], context['base'], controls, return_maps=return_maps)
            elif self.config.effective_base == 'baseline':
                linear = self.backbone._apply_ltm(context['_work_gain'], context['_work_base'], controls['ltm_params'])
                context['_baseline_full'] = linear
                result = {'image': self._crop(linear, context['image'].shape[-2:]), 'features': None,
                          'maps': {'ltm_params': controls['ltm_params']} if return_maps else {}}
            else:
                result = {'image': context['base'], 'features': None, 'maps': {}}
        else:
            if self.correction_operator is None or anchor is None or anchor.shape != context['image'].shape:
                raise ValueError('correction rendering requires configured operator and pre-chroma RGB anchor')
            gate = self._correction_gate(anchor, controls, correction_mask, correction_strength)
            if self.config.correction_algorithm == 'gain_residual':
                result = self.correction_operator.render(anchor, anchor, {**controls, 'ev_scale': gate}, return_maps=return_maps)
            else:
                result = self.correction_operator.render(anchor, anchor, controls, return_maps=return_maps)
                delta = (result['image'] - anchor).clamp(-self.config.correction_max_delta, self.config.correction_max_delta)
                result['image'] = anchor + gate * delta
                if return_maps:
                    result['maps'].update(correction_delta=delta)
            # Explicit selection protects exact zero fallback even for -0 and
            # avoids cancellation of tiny anchors in additive corrections.
            result['image'] = torch.where(gate == 0, anchor, result['image'])
            if return_maps:
                result['maps']['correction_gate'] = gate
        if result['image'].shape != context['image'].shape:
            raise ValueError('operator must preserve RGB input shape')
        result['controls'] = {**controls, '_context': context}
        return result

    def _chroma_input(self, linear):
        if self.backbone._3d_lut is not None:
            linear = self.backbone._apply_3d_lut_on_rgb(linear, self.backbone._3d_lut())
        return self.backbone.rgb_to_ycbcr(linear.clamp(0., 1.))

    def _apply_chroma(self, ycbcr, lut):
        cbcr = self.backbone._apply_2d_lut_on_cbcr(ycbcr[:, 1:], lut)
        return self.backbone.ycbcr_to_rgb(torch.cat((ycbcr[:, :1], cbcr), 1))

    def _predict_post(self, reference):
        ycbcr = self._chroma_input(reference)
        lut = self.backbone._lut_net(ycbcr)
        pre_gamma = self._apply_chroma(ycbcr, lut)
        return {'cbcr_lut': lut, 'gamma_factor': self.backbone._gamma_net(pre_gamma)}

    def finish(self, linear, controls=None, return_maps=True):
        """Apply adaptive post controls, or cached frozen original-LTM controls.

        Fixed reference requires controls returned by render/predict so that its
        frame context is explicit; controls never depend on candidate pixels.
        """
        context = controls.get('_context') if controls is not None else None
        post_linear = linear
        if context is not None and '_baseline_full' in context:
            baseline_full = context['_baseline_full']
            h, w = linear.shape[-2:]
            if baseline_full.shape[-2:] != (h, w):
                # Preserve the existing tiny-baseline upstream padded boundary.
                # Corrections touch the actual image ROI, not the padding.
                top = torch.cat((linear, baseline_full[..., :h, w:]), -1)
                post_linear = torch.cat((top, baseline_full[..., h:, :]), -2)
        if self.config.post_mode == 'fixed_reference':
            if context is None or 'post_controls' not in context:
                raise ValueError('fixed_reference finish requires controls carrying prepared frame context')
            post = context['post_controls']
            ycbcr = self._chroma_input(post_linear)
            pre_gamma = self._apply_chroma(ycbcr, post['cbcr_lut'])
            gamma = post['gamma_factor']
            lut = post['cbcr_lut']
        else:
            ycbcr = self._chroma_input(post_linear)
            lut = self.backbone._lut_net(ycbcr)
            pre_gamma = self._apply_chroma(ycbcr, lut)
            gamma = self.backbone._gamma_net(pre_gamma)
        # Original baseline arithmetic remains unchanged. Other candidates can
        # generate negative reconstructed RGB, which needs a finite power base.
        power_input = pre_gamma if self.config.effective_base == 'baseline' else pre_gamma.clamp_min(1e-8)
        output = self.backbone._apply_gamma(power_input, gamma).clamp(0., 1.)
        return {'output': self._crop(output, linear.shape[-2:]), 'linear': linear,
                'maps': {'cbcr_lut': lut, 'gamma_factor': gamma} if return_maps else {}}

    def forward(self, image, semantics=None, confidence=None, *, return_maps=True,
                correction_mask=None, correction_strength=None):
        context = self.prepare_global(image)
        controls = self.predict_controls(context, semantics, confidence)
        rendered = self.render(context, controls, return_maps=return_maps)
        base_maps = rendered['maps']
        if self.correction_operator is not None:
            controls = self.predict_controls(context, semantics, confidence, stage='correction', anchor=rendered['image'])
            rendered = self.render(context, controls, stage='correction', anchor=rendered['image'],
                                   correction_mask=correction_mask, correction_strength=correction_strength,
                                   return_maps=return_maps)
        elif correction_mask is not None or correction_strength is not None:
            raise ValueError('correction controls require a configured correction operator')
        result = self.finish(rendered['image'], rendered['controls'], return_maps=return_maps)
        if return_maps:
            result['maps'].update(base_maps)
            if self.correction_operator is not None:
                result['maps'].update({f'correction_{key}': value for key, value in rendered['maps'].items()})
            result['maps'].update(gain_factor=context['gain_factor'], gtm_params=context['gtm_params'],
                                  gain_rgb=context['gain'], base_rgb=context['base'])
        if self.training and self.semantic_head is not None:
            logits = self.semantic_head(rendered['features'])
            result['semantic_logits'] = F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
        return result
