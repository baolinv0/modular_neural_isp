"""Differentiable synthetic-exposure Laplacian fusion (research candidate D).

Exposure synthesis multiplies nonnegative, *untruncated* pre-TM gain RGB by
2**EV and then applies the channelwise Reinhard curve x/(1+x). This produces
valid RGB exposures without erasing differences above one before synthesis.
It is a deliberate rendering curve, not a camera capture or a claim of extra
scene information / independent-noise reduction. Channelwise compression can
change saturated colors; subsequent shared chroma processing is unchanged.

Each RGB exposure has a Laplacian pyramid; each predicted softmax weight has
a Gaussian pyramid. The coarsest blended image is expanded and successive
blended detail bands are added. Convex weights at every scale do not imply a
convex final RGB image: reconstruction can overshoot. We preserve that output
for clipping diagnostics; the common pipeline owns its final clamp.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tm_library.common import ConditionEncoder
from tm_library.config import ModelConfig


def _gaussian_reduce(image: Tensor) -> Tensor:
    """Replicate-padded binomial Gaussian filtering followed by decimation."""
    channels = image.shape[1]
    kernel = image.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    horizontal = kernel.reshape(1, 1, 1, 5).expand(channels, 1, 1, 5)
    vertical = kernel.reshape(1, 1, 5, 1).expand(channels, 1, 5, 1)
    blurred = F.conv2d(F.pad(image, (2, 2, 0, 0), mode="replicate"), horizontal, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (0, 0, 2, 2), mode="replicate"), vertical, groups=channels)
    return blurred[..., ::2, ::2]


def gaussian_pyramid(image: Tensor, levels: int) -> list[Tensor]:
    """Return at most ``levels`` scales, stopping when either axis is one.

    Shapes use ceil division on decimation, so odd dimensions never require
    cropping or mismatched reconstruction. Singleton inputs remain valid.
    """
    if levels < 1:
        raise ValueError("pyramid levels must be positive")
    pyramid = [image]
    for _ in range(levels - 1):
        if min(pyramid[-1].shape[-2:]) <= 1:
            break
        pyramid.append(_gaussian_reduce(pyramid[-1]))
    return pyramid


def _expand(image: Tensor, size: tuple[int, int]) -> Tensor:
    # The same expansion defines both detail bands and reconstruction, giving
    # exact telescoping reconstruction even for odd shapes.
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def laplacian_pyramid(image: Tensor, levels: int) -> list[Tensor]:
    """Fine-to-coarse Laplacian bands followed by the residual Gaussian image."""
    gaussian = gaussian_pyramid(image, levels)
    return [fine - _expand(coarse, fine.shape[-2:]) for fine, coarse in zip(gaussian, gaussian[1:])] + [gaussian[-1]]


def reconstruct_pyramid(pyramid: Sequence[Tensor]) -> Tensor:
    """Reconstruct an image from fine-to-coarse bands without output clipping."""
    if not pyramid:
        raise ValueError("cannot reconstruct an empty pyramid")
    image = pyramid[-1]
    for band in reversed(pyramid[:-1]):
        image = _expand(image, band.shape[-2:]) + band
    return image


def multiscale_fusion(exposures: Tensor, weights: Tensor, levels: int, *, return_maps: bool = True) -> tuple[Tensor, list[Tensor]]:
    """Fuse BK3HW exposures using BKHW weights, normalized at every scale.

    The input weights are expected to be nonnegative with positive sums, as
    provided by softmax. The normalization also permits exact one-hot weights.
    With diagnostics disabled, normalized scale weights are consumed one at a
    time and no diagnostic list retains them. The pyramids remain intrinsic to
    this renderer's multiscale mechanism.
    """
    batch, count, channels, height, width = exposures.shape
    if weights.shape != (batch, count, height, width):
        raise ValueError("exposure weights must have shape (B,K,H,W)")
    rgb_pyramid = laplacian_pyramid(exposures.reshape(batch * count, channels, height, width), levels)
    raw_weights = gaussian_pyramid(weights, levels)
    scale_weights = []
    blended = []
    for band, scale in zip(rgb_pyramid, raw_weights):
        weight = scale / scale.sum(dim=1, keepdim=True).clamp_min(torch.finfo(scale.dtype).tiny)
        band = band.reshape(batch, count, channels, *band.shape[-2:])
        blended.append((band * weight.unsqueeze(2)).sum(dim=1))
        if return_maps:
            scale_weights.append(weight)
    return reconstruct_pyramid(blended), scale_weights


class ExposureFusionOperator(nn.Module):
    """Feature/semantic-conditioned, genuinely multiscale exposure fusion.

    Semantics and confidence use the common encoder's gating contract. A
    low-resolution convolution predicts weights, bilinearly resized before
    softmax; no semantic labels are assigned fixed desired brightness.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ConditionEncoder(config)
        self.weight_head = nn.Conv2d(config.width, len(config.exposures), kernel_size=1)
        self.register_buffer("exposure_ev", torch.tensor(config.exposures, dtype=torch.float32))

    def predict(self, gain: Tensor, base: Tensor, semantics: Tensor | None = None, confidence: Tensor | None = None) -> dict:
        """Predict only analysis-resolution feature and exposure-weight logits."""
        features = self.encoder(gain, base, semantics, confidence)
        return {"features": features, "weight_logits": self.weight_head(features),
                "exposure_ev": self.exposure_ev}

    def render(self, gain: Tensor, base: Tensor, controls: dict, return_maps: bool = True) -> dict:
        """Synthesize full-resolution exposures and reconstruct their pyramids."""
        logits = F.interpolate(controls["weight_logits"], size=gain.shape[-2:], mode="bilinear", align_corners=False)
        weights = logits.softmax(dim=1)
        multipliers = torch.exp2(controls["exposure_ev"].to(gain))[None, :, None, None, None]
        radiance = gain.clamp_min(0).unsqueeze(1) * multipliers
        exposures = radiance / (1 + radiance)
        image, scale_weights = multiscale_fusion(exposures, weights, self.config.pyramid_levels,
                                                return_maps=return_maps)
        maps = {}
        if return_maps:
            maps = {"exposures": exposures, "weights": weights}
            maps.update({f"weights_scale_{index}": scale for index, scale in enumerate(scale_weights)})
        return {"image": image, "features": controls["features"], "maps": maps}

    def forward(self, gain: Tensor, base: Tensor, semantics: Tensor | None = None,
                confidence: Tensor | None = None, *, return_maps: bool = True) -> dict:
        controls = self.predict(gain, base, semantics, confidence)
        return self.render(gain, base, controls, return_maps=return_maps)
