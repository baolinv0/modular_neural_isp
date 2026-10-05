"""Differentiable, exposure-aware adaptation of Modular Neural ISP tone mapping.

The shipped networks remain image-adaptive frozen coefficient predictors. Small
zero-initialized adapters learn gain, shoulder, GTM and spatial LTM jointly with
capture policy. ``prepare`` is independent of every trainable parameter, so its
outputs can be cached for finite-candidate training without changing gradients.
This is parameter-efficient TM training, not full-backbone fine-tuning.
"""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .modular import ModularPhotofinishingBackend, _validate_linear_rgb


class ConditionalToneMapper(nn.Module):
    """Single-capture Apple or exposure-normalized HDR Samsung renderer.

    ``capture_ev`` is log2 effective exposure relative to the sensor reference.
    Only Apple uses it to undo capture brightness. Samsung input is already in
    reference-radiance units, so capture_ev never rescales it. ``render_ev`` is
    independent scene appearance intent. Confidence is in [0, 1].
    """

    histogram_bins = 24

    def __init__(self, scheme: str, weights: str | Path | None = None,
                 trainable: bool = True):
        super().__init__()
        if scheme not in {"apple", "samsung"}:
            raise ValueError("scheme must be apple or samsung")
        self.scheme = scheme
        backend = ModularPhotofinishingBackend(weights)
        self.model = backend.model
        self.checkpoint_sha256 = backend.checkpoint_sha256
        self.renderer_identity = f"conditional-{scheme}-adapters-v1:{backend.renderer_identity}"
        # 8 physical/statistical conditions and source/virtual-target histograms.
        self.conditioner = nn.Sequential(
            nn.Linear(8 + 2 * self.histogram_bins, 32), nn.SiLU(), nn.Linear(32, 10))
        # Spatial confidence and canonical RGB modulate the five LTM coefficients.
        self.local_adapter = nn.Sequential(
            nn.Conv2d(4, 12, 3, padding=1), nn.SiLU(), nn.Conv2d(12, 5, 1))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        nn.init.zeros_(self.local_adapter[-1].weight)
        nn.init.zeros_(self.local_adapter[-1].bias)
        self.register_buffer("histogram_centers", torch.linspace(0., 1., self.histogram_bins))
        self.model.requires_grad_(False)
        self.conditioner.requires_grad_(trainable)
        self.local_adapter.requires_grad_(trainable)
        self.train(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        # Predictor inference behavior is shared by all factorial groups.
        self.model.eval()
        return self

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.model._rgb_to_ycbcr_matrix = fn(self.model._rgb_to_ycbcr_matrix)
        self.model._ycbcr_to_rgb_matrix = fn(self.model._ycbcr_to_rgb_matrix)
        self.model._device = next(self.model.parameters()).device
        return self

    @staticmethod
    def _batch_ev(value, x: torch.Tensor, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        if value.ndim == 0:
            value = value.expand(x.shape[0])
        if value.shape != (x.shape[0],) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite scalar or [B]")
        if (value.abs() > 16).any():
            raise ValueError(f"{name} must lie in [-16, 16] EV")
        return value

    @staticmethod
    def _shoulder(rgb: torch.Tensor, knee=.7) -> torch.Tensor:
        """Hue-ratio-preserving C1 shoulder, linear below knee, asymptote 1."""
        peak = rgb.amax(dim=1, keepdim=True)
        knee = torch.as_tensor(knee, dtype=rgb.dtype, device=rgb.device)
        width = 1. - knee
        mapped = torch.where(peak <= knee, peak,
                             1. - width.square() / (peak + 1. - 2. * knee).clamp_min(.01))
        return rgb * (mapped / peak.clamp_min(1e-8))

    @staticmethod
    def _local_tone(model, x, x_gtm, coefficients):
        # Original LTM multiplies x*g and then apply_tm clips at one. Replace
        # that destructive local clip with the bounded monotone odds gain.
        w = torch.sigmoid(coefficients[:, 0:1])
        a, b, c, gain = [coefficients[:, i:i + 1] for i in range(1, 5)]
        gained = x * gain / (1. + x * (gain - 1.)).clamp_min(1e-8)
        local = model.apply_tm(gained, a, b, c)
        return (1. - w) * x_gtm + w * local

    def prepare(self, x: torch.Tensor, capture_ev=0., reliability=None) -> dict[str, torch.Tensor]:
        """Compute frozen image-adaptive coefficients, keeping input gradients.

        All values are tensors with leading batch dimension. Cache only when
        input/capture metadata are fixed; changing radiance requires re-prepare.
        No trainable adapter is executed here, and no no_grad/detach is imposed.
        """
        _validate_linear_rgb(x, bounded=False)
        if min(x.shape[-2:]) < 12:
            raise ValueError("Modular tone mapping requires H and W >= 12")
        if x.device != next(self.model.parameters()).device:
            raise ValueError("input and tone mapper must be on the same device")
        capture_ev = self._batch_ev(capture_ev, x, "capture_ev")
        if self.scheme == "samsung":
            capture_ev = torch.zeros_like(capture_ev)
        radiance = x * torch.exp2(-capture_ev[:, None, None, None])
        if not torch.isfinite(radiance).all():
            raise ValueError("exposure-normalized radiance overflowed")
        if reliability is None:
            reliability = torch.ones_like(x[:, :1])
        if (not isinstance(reliability, torch.Tensor) or
                reliability.shape != (x.shape[0], 1, *x.shape[-2:]) or
                reliability.device != x.device or not torch.isfinite(reliability).all() or
                (reliability < 0).any() or (reliability > 1).any()):
            raise ValueError("reliability must be finite [B,1,H,W] in [0,1] on input device")
        reliability = reliability.to(x.dtype)
        base_view = self._shoulder(radiance)
        gain = self.model._gain_net(base_view)
        base_gain_view = self._shoulder(radiance * gain)
        gtm = self.model._gtm_net(base_gain_view)
        base_gtm = self.model._apply_gtm(base_gain_view, gtm)
        ltm = self.model._ltm_net(base_gain_view, base_gtm, post_process_ltm=False)
        base_ltm = self._local_tone(self.model, base_gain_view, base_gtm, ltm)
        if self.model._3d_lut is not None:
            base_ltm = self.model._apply_3d_lut_on_rgb(base_ltm, self.model._3d_lut())
        base_ycbcr = self.model.rgb_to_ycbcr(base_ltm)
        chroma_lut = self.model._lut_net(base_ycbcr)
        base_cbcr = self.model._apply_2d_lut_on_cbcr(base_ycbcr[:, 1:], chroma_lut)
        base_color = self.model.ycbcr_to_rgb(torch.cat([base_ycbcr[:, :1], base_cbcr], dim=1))
        gamma = self.model._gamma_net(base_color)
        return dict(radiance=radiance, capture_ev=capture_ev, reliability=reliability,
                    gain=gain, gtm=gtm, ltm=ltm, chroma_lut=chroma_lut,
                    gamma=gamma, base_view=base_view)

    def _soft_histogram(self, y: torch.Tensor) -> torch.Tensor:
        # Soft PDF remains differentiable in image values and virtual gain.
        sample = F.adaptive_avg_pool2d(y, (16, 16)).flatten(1).unsqueeze(-1)
        distance = (sample - self.histogram_centers[None, None]) * (self.histogram_bins - 1)
        assignment = torch.softmax(-.5 * distance.square(), dim=-1)
        return assignment.mean(dim=1)

    def render_prepared(self, prepared: dict[str, torch.Tensor], render_ev=0.) -> torch.Tensor:
        """Render fixed radiance/coefficient tensors using trainable TM adapters."""
        x = prepared["radiance"]
        render_ev = self._batch_ev(render_ev, x, "render_ev")
        virtual_gain = torch.exp2(render_ev[:, None, None, None])
        y = x.amax(dim=1, keepdim=True)
        # Samsung's virtual gain changes the target PDF used to build the curve;
        # source and target histograms share the same display-intensity axis.
        source_pdf = self._soft_histogram(y.clamp(0, 1))
        target_pdf = self._soft_histogram((y * virtual_gain).clamp(0, 1))
        confidence = prepared["reliability"]
        stats = torch.stack([
            prepared["capture_ev"] / 8., render_ev / 8.,
            confidence.mean(dim=(1, 2, 3)), confidence.var(dim=(1, 2, 3), unbiased=False),
            torch.log2(y.mean(dim=(1, 2, 3)).clamp_min(1e-6)) / 12.,
            torch.log2(y.square().mean(dim=(1, 2, 3)).clamp_min(1e-6)) / 24.,
            (y > 1.).to(y.dtype).mean(dim=(1, 2, 3)),
            (y < .01).to(y.dtype).mean(dim=(1, 2, 3))], dim=1)
        context = torch.cat([stats, source_pdf, target_pdf], dim=1)
        correction = torch.tanh(self.conditioner(context))
        # One coordinated gain: checkpoint gain + render intent + learned EV.
        gain = prepared["gain"] * virtual_gain * torch.exp2(correction[:, :1, None, None])
        if self.scheme == "samsung":
            # Fixed histogram contribution makes virtual-gain curve construction
            # active even with zero-initialized learned adapters.
            shift = ((target_pdf - source_pdf) * self.histogram_centers).sum(dim=1)
            knee_offset = .06 * torch.tanh(4. * shift)
        else:
            knee_offset = torch.zeros_like(render_ev)
        knee = .7 + knee_offset + .08 * correction[:, 1]
        toned_input = self._shoulder(x * gain, knee[:, None, None, None])
        gtm_params = prepared["gtm"] * torch.exp(.35 * correction[:, 2:5])
        global_tone = self.model._apply_gtm(toned_input, gtm_params)
        spatial = torch.tanh(self.local_adapter(torch.cat([prepared["base_view"], confidence], dim=1)))
        # Confidence spatially gates local adaptation, reducing adjustment in
        # regions where HDR fusion supplies little reliable information.
        delta = .25 * correction[:, 5:, None, None] + .15 * spatial * confidence
        ltm_params = prepared["ltm"] * torch.exp(delta)
        local_tone = self._local_tone(self.model, toned_input, global_tone, ltm_params)
        if self.model._3d_lut is not None:
            local_tone = self.model._apply_3d_lut_on_rgb(local_tone, self.model._3d_lut())
        ycbcr = self.model.rgb_to_ycbcr(local_tone)
        cbcr = self.model._apply_2d_lut_on_cbcr(ycbcr[:, 1:], prepared["chroma_lut"])
        color = self.model.ycbcr_to_rgb(torch.cat([ycbcr[:, :1], cbcr], dim=1))
        # The checkpoint's fractional gamma has an infinite derivative at zero.
        # A numerical floor keeps joint training finite while preserving black.
        output = color.clamp_min(1e-7).pow(prepared["gamma"])
        output = torch.where(color > 0, output, torch.zeros_like(output))
        return output.clamp(0, 1)

    def forward(self, x: torch.Tensor, capture_ev=0., render_ev=0., reliability=None) -> torch.Tensor:
        return self.render_prepared(self.prepare(x, capture_ev, reliability), render_ev)
