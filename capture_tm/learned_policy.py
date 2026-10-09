"""Causal temporal exposure selection and differentiable discrete quality risk.

An AdaptiveAE-inspired observation interface is shared by the single-capture
and three-capture HDR algorithms. This is a small new CNN, not an AdaptiveAE
checkpoint or a reproduction of its reinforcement-learning implementation.
Only past sensor previews, their effective settings, and legal planned settings
enter the scorer. The training objective evaluates each capture independently;
inference selects one legal candidate with ``scores.argmax(-1)``.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_mask(mask: Tensor, shape: tuple[int, int], device: torch.device) -> None:
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError("feasible_mask must be boolean with shape [B,K]")
    if mask.device != device:
        raise ValueError("feasible_mask must share the input device")
    if not mask.any(-1).all():
        raise ValueError("every observation needs at least one feasible candidate")


def preview_clipping_guard(previews: Tensor, state: Tensor, features: Tensor,
                           rule_indices: Tensor, *, tolerance: float = .01):
    """Causal *estimated* all-frame clipping risk relative to the rule plan.

    Latest sensor-linear preview and actual EV project each candidate slot.
    Count a channel/pixel only when ALL slots exceed 98% ADC headroom. A
    candidate may exceed the reference risk by at most ``tolerance``. This is
    a static-scene, linear-response proxy, not a calibrated safety guarantee:
    clipped previews, motion, noise, full-well changes and unseen highlights
    limit its accuracy. No candidate measurement or target is an input.
    """
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError('tolerance must be finite in [0,1]')
    if (previews.ndim != 5 or previews.shape[2] != 3 or state.shape != (*previews.shape[:2], 3)
            or features.ndim != 4 or features.shape[0] != len(previews)
            or features.shape[2] not in (1, 3) or features.shape[3] != 3):
        raise ValueError('expected previews [B,T,3,H,W], state [B,T,3], features [B,K,N,3]')
    if (rule_indices.shape != (len(previews),) or rule_indices.dtype != torch.long
            or (rule_indices < 0).any() or (rule_indices >= features.shape[1]).any()):
        raise ValueError('rule indices must identify one reference per observation')
    if any(t.device != previews.device for t in (state, features, rule_indices)):
        raise ValueError('guard inputs must share a device')
    if (not all(torch.isfinite(t).all() for t in (previews, state, features))
            or (previews < 0).any() or (previews > 1).any()):
        raise ValueError('guard observations must be finite and bounded')
    ev_delta = features.sum(-1) - state[:, -1].sum(-1)[:, None, None]
    projected = previews[:, -1, None, None] * torch.exp2(ev_delta[..., None, None, None].clamp(-32, 32))
    risk = (projected >= .98).all(2).float().mean((2, 3, 4))
    reference = risk.gather(1, rule_indices[:, None])
    mask = risk <= reference + tolerance
    mask.scatter_(1, rule_indices[:, None], True)
    return mask, risk


class TemporalExposurePolicy(nn.Module):
    """Score ordered single-frame actions or three-frame physical capture plans.

    Metadata and candidates use ``(log2(t/reference_t), log2(analog_gain),
    log2(digital_gain))``. Previews must be bounded sensor observations, ordered
    oldest to latest; metadata records settings actually used for those images.
    Candidate slots retain capture order, including the middle reference slot
    for HDR. Reordering the *candidate bank* simply reorders output scores.

    A shared CNN sees every preview. A second CNN sees adjacent raw differences
    and differences after approximate radiometric exposure normalization. Soft
    histograms, effective settings, and explicit temporal difference statistics
    provide brightness, exposure, and motion context without future data.
    """

    def __init__(self, width: int = 16, history_length: int = 3):
        super().__init__()
        for name, value in (("width", width), ("history_length", history_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.width = width
        self.history_length = history_length
        self.histogram_bins = 16

        def encoder(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d(2), nn.Flatten(),
                nn.Linear(8 * width, 2 * width), nn.SiLU(),
            )

        self.image_encoder = encoder(3)
        self.motion_encoder = encoder(6)
        frame_features = 2 * width + 3 * self.histogram_bins + 3 + 1
        motion_features = 2 * width + 12 + 1
        self.context_encoder = nn.Sequential(
            nn.Linear(history_length * frame_features + (history_length - 1) * motion_features,
                      4 * width), nn.SiLU(),
            nn.Linear(4 * width, 4 * width), nn.SiLU(),
        )
        # Three fixed temporal slots, each carrying three log settings plus a
        # presence bit. Single-capture candidates occupy the reference slot.
        self.plan_encoder = nn.Sequential(nn.Linear(12, 4 * width), nn.SiLU(),
                                          nn.Linear(4 * width, 4 * width), nn.SiLU())
        self.scorer = nn.Sequential(nn.Linear(12 * width, 2 * width), nn.SiLU(),
                                    nn.Linear(2 * width, 1))

    def _histograms(self, previews: Tensor) -> Tensor:
        pixels = previews.flatten(-2) * (self.histogram_bins - 1)
        lower = pixels.floor().long().clamp(0, self.histogram_bins - 1)
        upper = (lower + 1).clamp_max(self.histogram_bins - 1)
        fraction = pixels - lower.to(pixels.dtype)
        histogram = previews.new_zeros(*pixels.shape[:-1], self.histogram_bins)
        histogram.scatter_add_(-1, lower, 1. - fraction)
        histogram.scatter_add_(-1, upper, fraction)
        return histogram.flatten(2) / pixels.shape[-1]

    def forward(self, previews: Tensor, capture_state: Tensor,
                candidate_features: Tensor, feasible_mask: Tensor) -> Tensor:
        """Return scores [B,K], masking illegal actions with negative infinity.

        Inputs are [B,T,3,H,W], [B,T,3], [B,K,N,3], [B,K]. N must be 1 or 3.
        Histories shorter than ``history_length`` are left-padded with absent
        observations. No future observation is inferred or padded as present.
        """
        if not isinstance(previews, Tensor) or previews.ndim != 5:
            raise ValueError("previews must have shape [B,T,3,H,W]")
        batch, history, channels, height, width = previews.shape
        if batch < 1 or not 1 <= history <= self.history_length or channels != 3 or min(height, width) < 1:
            raise ValueError("preview batch, history, channels, or image dimensions are invalid")
        if not isinstance(capture_state, Tensor) or tuple(capture_state.shape) != (batch, history, 3):
            raise ValueError("capture_state must align with observed previews [B,T,3]")
        if (not isinstance(candidate_features, Tensor) or candidate_features.ndim != 4
                or candidate_features.shape[0] != batch or candidate_features.shape[1] < 1
                or candidate_features.shape[2] not in (1, 3) or candidate_features.shape[3] != 3):
            raise ValueError("candidate_features must have shape [B,K,N,3], N=1 or 3")
        candidates, plan_length = candidate_features.shape[1:3]
        _validate_mask(feasible_mask, (batch, candidates), previews.device)
        for name, value in (("previews", previews), ("capture_state", capture_state),
                            ("candidate_features", candidate_features)):
            if (not value.is_floating_point() or value.dtype != self.image_encoder[0].weight.dtype
                    or value.device != previews.device or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating point on the model dtype/input device")
        if (previews < 0).any() or (previews > 1).any():
            raise ValueError("previews must be bounded sensor observations in [0,1]")

        encoded = self.image_encoder(previews.reshape(batch * history, 3, height, width))
        frame_context = torch.cat((encoded.reshape(batch, history, -1),
                                   self._histograms(previews), capture_state,
                                   previews.new_ones(batch, history, 1)), dim=-1)
        frame_context = F.pad(frame_context, (0, 0, self.history_length - history, 0))
        if history > 1:
            # log1p limits very dark preview amplification; clipping in the
            # observed image remains uncertainty, not reconstructed information.
            inverse_exposure = torch.exp2(-capture_state.sum(-1).clamp(-16., 16.))
            radiometric = torch.log1p(previews * inverse_exposure[..., None, None, None])
            differences = torch.cat((previews[:, 1:] - previews[:, :-1],
                                     radiometric[:, 1:] - radiometric[:, :-1]), dim=2)
            motion = self.motion_encoder(differences.reshape(batch * (history - 1), 6, height, width))
            statistics = torch.cat((differences.abs().mean((-2, -1)),
                                    differences.abs().amax((-2, -1))), dim=-1)
            motion_context = torch.cat((motion.reshape(batch, history - 1, -1), statistics,
                                        previews.new_ones(batch, history - 1, 1)), dim=-1)
        else:
            motion_context = previews.new_zeros(batch, 0, 2 * self.width + 13)
        motion_context = F.pad(motion_context, (0, 0, self.history_length - history, 0))
        context = self.context_encoder(torch.cat((frame_context.flatten(1), motion_context.flatten(1)), -1))

        plan = torch.cat((candidate_features,
                          previews.new_ones(batch, candidates, plan_length, 1)), dim=-1)
        if plan_length == 1:
            plan = F.pad(plan, (0, 0, 1, 1))
        plan = self.plan_encoder(plan.flatten(2))
        context = context[:, None].expand(-1, candidates, -1)
        scores = self.scorer(torch.cat((context, plan, context * plan), dim=-1)).squeeze(-1)
        if not torch.isfinite(scores[feasible_mask]).all():
            raise ValueError("policy produced nonfinite feasible scores")
        return scores.masked_fill(~feasible_mask, -torch.inf)


def _validate_quality(scores: Tensor, costs: Tensor, feasible_mask: Tensor) -> None:
    if (scores.ndim != 2 or min(scores.shape) < 1 or costs.shape != scores.shape
            or not scores.is_floating_point() or not costs.is_floating_point()):
        raise ValueError("scores and costs must be floating point with matching [B,K] shapes")
    _validate_mask(feasible_mask, tuple(scores.shape), scores.device)
    if costs.device != scores.device:
        raise ValueError("costs and scores must share a device")
    if not torch.isfinite(scores[feasible_mask]).all() or not torch.isfinite(costs[feasible_mask]).all():
        raise ValueError("feasible scores and costs must be finite")


def expected_quality_loss(scores: Tensor, costs: Tensor, feasible_mask: Tensor,
                          temperature: float = 1.) -> Tensor:
    """Mean exact categorical quality risk, differentiable in AE and TM.

    Each ``costs[b,k]`` is measured on the separately captured and rendered
    candidate k. Gradients remain attached to those costs, so TM is optimized
    alongside AE probabilities. No mixed RAW or mixed rendered capture is used.
    Invalid entries may contain NaN/inf and are removed before arithmetic.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    _validate_quality(scores, costs, feasible_mask)
    legal_scores = scores.masked_fill(~feasible_mask, -torch.inf)
    centered_scores = legal_scores - legal_scores.amax(-1, keepdim=True)
    probabilities = (centered_scores / temperature).softmax(-1)
    safe_costs = costs.masked_fill(~feasible_mask, 0.)
    return (probabilities * safe_costs).sum(-1).mean()


def warmstart_loss(scores: Tensor, costs: Tensor, feasible_mask: Tensor) -> Tensor:
    """Supervise the best physical candidate without weakening small cost gaps.

    Tied minima receive equal mass; other legal candidates receive zero. Costs
    are labels and deliberately detached in this pretraining loss. Joint
    training subsequently uses ``expected_quality_loss`` to update AE and TM.
    """
    _validate_quality(scores, costs, feasible_mask)
    teacher_costs = costs.detach().masked_fill(~feasible_mask, torch.inf)
    best = feasible_mask & (teacher_costs == teacher_costs.amin(-1, keepdim=True))
    teacher = best.to(scores.dtype) / best.sum(-1, keepdim=True)
    log_probabilities = scores.masked_fill(~feasible_mask, -torch.inf).log_softmax(-1)
    log_probabilities = log_probabilities.masked_fill(~feasible_mask, 0.)
    return -(teacher * log_probabilities).sum(-1).mean()
