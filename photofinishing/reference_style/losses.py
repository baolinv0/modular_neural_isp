"""Non-pixel-aligned luma and chroma reference losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import ReferenceStyleLossWeights, TrainingStage
from .statistics import (
    channel_moments,
    global_contrast,
    luminance_occupancy,
    normalized_gradient_map,
    rgb_to_luma,
    rgb_to_ycbcr,
    soft_histogram,
)


@dataclass
class LossResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


class UnalignedReferenceStyleLoss(nn.Module):
    """Reference loss that never computes source-reference pixel error."""

    def __init__(self, weights: ReferenceStyleLossWeights | None = None, bins: int = 32):
        super().__init__()
        self.weights = weights or ReferenceStyleLossWeights()
        self.bins = bins

    def forward(self, output: torch.Tensor, reference: torch.Tensor, anchor: torch.Tensor,
                stage: TrainingStage) -> LossResult:
        if output.ndim != 4 or reference.ndim != 4 or anchor.shape != output.shape:
            raise ValueError("invalid output/reference/anchor shapes")
        if output.shape[0] != reference.shape[0]:
            raise ValueError("batch size mismatch")
        if not all(torch.isfinite(t).all() for t in (output, reference, anchor)):
            raise ValueError("non-finite loss input")
        if stage == TrainingStage.LUMA:
            return self._luma_loss(output, reference, anchor)
        if stage == TrainingStage.CHROMA:
            return self._chroma_loss(output, reference, anchor)
        raise ValueError(f"unsupported stage: {stage}")

    def _luma_loss(self, output: torch.Tensor, reference: torch.Tensor, anchor: torch.Tensor) -> LossResult:
        out_y = rgb_to_luma(output.clamp(0, 1))
        ref_y = rgb_to_luma(reference.clamp(0, 1))
        anchor_y = rgb_to_luma(anchor.clamp(0, 1))
        hist = F.l1_loss(soft_histogram(out_y, self.bins), soft_histogram(ref_y, self.bins))
        moments = F.l1_loss(channel_moments(out_y), channel_moments(ref_y))
        occupancy = F.l1_loss(luminance_occupancy(out_y), luminance_occupancy(ref_y))
        contrast = F.l1_loss(global_contrast(out_y), global_contrast(ref_y))
        content = F.l1_loss(normalized_gradient_map(out_y), normalized_gradient_map(anchor_y))
        terms = {
            "histogram": hist,
            "moments": moments,
            "occupancy": occupancy,
            "contrast": contrast,
            "content_anchor": content,
        }
        total = (
            self.weights.histogram * hist
            + self.weights.moments * moments
            + self.weights.occupancy * occupancy
            + self.weights.contrast * contrast
            + self.weights.content_anchor * content
        )
        return LossResult(total=total, terms=terms)

    def _chroma_loss(self, output: torch.Tensor, reference: torch.Tensor, anchor: torch.Tensor) -> LossResult:
        out_ycbcr = rgb_to_ycbcr(output.clamp(0, 1))
        ref_ycbcr = rgb_to_ycbcr(reference.clamp(0, 1))
        anchor_ycbcr = rgb_to_ycbcr(anchor.clamp(0, 1))
        out_cbcr = out_ycbcr[:, 1:]
        ref_cbcr = ref_ycbcr[:, 1:]
        cb_hist = F.l1_loss(
            soft_histogram(out_cbcr[:, 0:1] + 0.5, self.bins),
            soft_histogram(ref_cbcr[:, 0:1] + 0.5, self.bins),
        )
        cr_hist = F.l1_loss(
            soft_histogram(out_cbcr[:, 1:2] + 0.5, self.bins),
            soft_histogram(ref_cbcr[:, 1:2] + 0.5, self.bins),
        )
        moments = F.l1_loss(channel_moments(out_cbcr), channel_moments(ref_cbcr))
        luma_preservation = F.l1_loss(out_ycbcr[:, :1], anchor_ycbcr[:, :1])
        neutral_mask = anchor_ycbcr[:, 1:].square().sum(dim=1, keepdim=True).sqrt() < 0.04
        if neutral_mask.any():
            neutral = out_cbcr.masked_select(neutral_mask.expand_as(out_cbcr)).abs().mean()
        else:
            neutral = output.new_tensor(0.0)
        terms = {
            "cb_histogram": cb_hist,
            "cr_histogram": cr_hist,
            "moments": moments,
            "luma_preservation": luma_preservation,
            "neutral_preservation": neutral,
        }
        total = (
            self.weights.histogram * 0.5 * (cb_hist + cr_hist)
            + self.weights.moments * moments
            + self.weights.luma_preservation * luma_preservation
            + self.weights.neutral_preservation * neutral
        )
        return LossResult(total=total, terms=terms)
