"""C's supervised physical-action ranker, inspired by AdaptiveAE observations.

The image encoder is a small randomly initialized CNN trained with this ranker.
It is not AlexNet, an official AdaptiveAE checkpoint, or an RL/PPO reproduction.
Only already captured, bounded sensor observations and their *effective* action
metadata enter this policy; candidate images and clean scene data never enter it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def action_features(actions: Sequence, reference_exposure_s: float, *, device=None,
                    dtype=torch.float32) -> Tensor:
    """Encode capture history or candidates with the same raw log2 convention.

    Columns: log2(t/reference_t), log2(analog_gain), log2(digital_gain).
    These are brightness coordinates, not photon-quality estimates. The caller
    must pass CaptureResult.action for observed history, never a pending request.
    """
    if not math.isfinite(reference_exposure_s) or reference_exposure_s <= 0:
        raise ValueError("reference_exposure_s must be positive and finite")
    if not dtype.is_floating_point:
        raise ValueError("action features require a floating point dtype")
    rows = []
    for action in actions:
        values = (float(action.exposure_s), float(action.analog_gain), float(action.digital_gain))
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("capture action values must be positive and finite")
        rows.append((math.log2(values[0]) - math.log2(reference_exposure_s),
                     math.log2(values[1]), math.log2(values[2])))
    if not rows:
        raise ValueError("at least one capture action is required")
    result = torch.tensor(rows, device=device, dtype=dtype)
    if not torch.isfinite(result).all():
        raise ValueError("capture log features overflow the requested dtype")
    return result


def _validate_mask(mask: Tensor, shape: tuple[int, int]) -> None:
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError("feasible_mask must be boolean with shape [B,K]")
    if not mask.any(dim=-1).all():
        raise ValueError("every batch row must contain at least one feasible action")


class ExposurePolicy(nn.Module):
    """Shared candidate scorer for variable-size physical capture action grids.

    ``num_actions`` records the default training grid size; it does not allocate
    one parameter per action. At inference any K>0 is supported, with scores
    invariant to candidate ordering. ``use_auxiliary=False`` is the matched
    image-history ablation: all observed-image histograms remain available, but
    external capture state and stage conditions are replaced by fixed zeros.
    Both variants instantiate and execute the same encoders/scorer, with the
    same parameter count, legality checks and image information.
    """
    def __init__(self, num_actions: int, *, width: int = 24, histogram_bins: int = 32,
                 history_length: int = 3, use_auxiliary: bool = True):
        super().__init__()
        for name, value in (("num_actions", num_actions), ("width", width),
                            ("histogram_bins", histogram_bins), ("history_length", history_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if histogram_bins < 2:
            raise ValueError("histogram_bins must be at least two")
        self.num_actions = num_actions
        self.width = width
        self.histogram_bins = histogram_bins
        self.history_length = history_length
        self.use_auxiliary = bool(use_auxiliary)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2 * width, 2 * width), nn.SiLU(),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(history_length * (3 * histogram_bins + 3), 4 * width), nn.SiLU(),
            nn.Linear(4 * width, 2 * width), nn.SiLU(),
        )
        self.stage_encoder = nn.Sequential(nn.Linear(2, width), nn.SiLU())
        context_width = 5 * width
        self.candidate_encoder = nn.Sequential(nn.Linear(3, width), nn.SiLU())
        self.scorer = nn.Sequential(
            nn.Linear(context_width + width, 2 * width), nn.SiLU(), nn.Linear(2 * width, 1),
        )
        self.register_buffer("histogram_centers", torch.linspace(0., 1., histogram_bins))

    def _histograms(self, previews: Tensor) -> Tensor:
        # Linear bin interpolation avoids a hard, nondifferentiable histogram.
        # scatter_add builds [B,T,C,bins] without [B,T,C,H,W,bins] memory growth.
        pixels = previews.flatten(-2) * (self.histogram_bins - 1)
        lower = pixels.floor().long().clamp(0, self.histogram_bins - 1)
        upper = (lower + 1).clamp_max(self.histogram_bins - 1)
        fraction = pixels - lower.to(pixels.dtype)
        histogram = previews.new_zeros(*pixels.shape[:-1], self.histogram_bins)
        histogram.scatter_add_(-1, lower, 1. - fraction)
        histogram.scatter_add_(-1, upper, fraction)
        return histogram / pixels.shape[-1]

    def forward(self, previews: Tensor, capture_state: Tensor, candidate_features: Tensor,
                feasible_mask: Tensor, *, stage=None) -> Tensor:
        """Return bounded legal scores and -inf for every illegal candidate.

        Images [B,T,3,H,W] must be sensor-normalized observations in [0,1], ordered
        oldest to latest. Metadata [B,T,3] must correspond to those same captures.
        ``stage`` is [B,2] current/total steps, with 0<=current<=total and total>0.
        Short histories are zero-padded on the oldest side; future frames are
        never padded or synthesized. The latest observed image drives the CNN.
        """
        if not isinstance(previews, Tensor) or previews.ndim != 5:
            raise ValueError("previews must have shape [B,T,3,H,W]")
        batch, history, channels, height, width = previews.shape
        if batch < 1 or not 1 <= history <= self.history_length or channels != 3 or min(height, width) < 1:
            raise ValueError("preview batch/history/channel/spatial shape is invalid")
        if tuple(capture_state.shape) != (batch, history, 3):
            raise ValueError("capture_state must align with preview history [B,T,3]")
        if candidate_features.ndim != 3 or candidate_features.shape[0] != batch or candidate_features.shape[2] != 3 or candidate_features.shape[1] < 1:
            raise ValueError("candidate_features must have shape [B,K,3], K>0")
        candidates = candidate_features.shape[1]
        _validate_mask(feasible_mask, (batch, candidates))
        for name, value in (("previews", previews), ("capture_state", capture_state),
                            ("candidate_features", candidate_features)):
            if value.dtype != torch.float32 or value.device != previews.device or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite float32 on the preview device")
        if feasible_mask.device != previews.device:
            raise ValueError("feasible_mask must be on the preview device")
        if (previews < 0).any() or (previews > 1).any():
            raise ValueError("previews must be bounded sensor observations in [0,1]")
        if stage is None:
            stage = previews.new_tensor([float(history - 1), float(self.history_length)]).expand(batch, 2)
        elif not isinstance(stage, Tensor) or tuple(stage.shape) != (batch, 2):
            raise ValueError("stage must have shape [B,2]")
        if not stage.is_floating_point() or stage.device != previews.device or not torch.isfinite(stage).all():
            raise ValueError("stage must be finite floating point metadata on the preview device")
        if (stage[:, 0] < 0).any() or (stage[:, 1] <= 0).any() or (stage[:, 0] > stage[:, 1]).any():
            raise ValueError("stage requires 0 <= current <= total, total > 0")
        context = [self.image_encoder(previews[:, -1])]
        histogram = self._histograms(previews).flatten(2)
        observed_state = capture_state if self.use_auxiliary else torch.zeros_like(capture_state)
        history_features = torch.cat((histogram, observed_state), dim=-1)
        if history < self.history_length:
            history_features = F.pad(history_features, (0, 0, self.history_length - history, 0))
        context.append(self.history_encoder(history_features.flatten(1)))
        normalized_stage = torch.stack((stage[:, 0] / stage[:, 1], torch.log2(stage[:, 1]) / 8.), dim=-1).to(previews.dtype)
        observed_stage = normalized_stage if self.use_auxiliary else torch.zeros_like(normalized_stage)
        context.append(self.stage_encoder(observed_stage))
        context_features = torch.cat(context, dim=-1).unsqueeze(1).expand(-1, candidates, -1)
        candidate_embedding = self.candidate_encoder(candidate_features)
        logits = self.scorer(torch.cat((context_features, candidate_embedding), dim=-1)).squeeze(-1)
        scores = 20. * torch.tanh(logits / 20.)
        if not torch.isfinite(scores[feasible_mask]).all():
            raise ValueError("policy produced nonfinite feasible scores")
        return scores.masked_fill(~feasible_mask, -torch.inf)


def ranking_loss(scores: Tensor, target_costs: Tensor, feasible_mask: Tensor) -> Tensor:
    """Fit a legal-only soft oracle distribution plus centered score utilities.

    Lower costs are better. A per-observation centered/scaled teacher avoids a
    renderer's arbitrary cost scale dominating training. Invalid scores/costs
    may contain NaN/inf and are removed *before* any arithmetic. This is offline
    supervised ranking; no policy-gradient/RL reproduction is implied.
    """
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1 or target_costs.shape != scores.shape:
        raise ValueError("scores and target_costs must have matching [B,K] shapes")
    _validate_mask(feasible_mask, tuple(scores.shape))
    if target_costs.device != scores.device or feasible_mask.device != scores.device:
        raise ValueError("ranking tensors must share a device")
    if not scores.is_floating_point() or not target_costs.is_floating_point():
        raise ValueError("ranking scores and costs must be floating point")
    if not torch.isfinite(scores[feasible_mask]).all() or not torch.isfinite(target_costs[feasible_mask]).all():
        raise ValueError("feasible scores and teacher costs must be finite")
    # Normalize before squaring or casting to score precision. Finite teacher
    # costs can be very large/small, and their physical unit must not erase rank.
    costs = target_costs.detach().double().masked_fill(~feasible_mask, 0.)
    max_abs = costs.abs().amax(-1, keepdim=True)
    costs = costs / torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    legal_count = feasible_mask.sum(-1, keepdim=True).to(scores.dtype)
    mean_cost = costs.sum(-1, keepdim=True) / legal_count
    centered_cost = (costs - mean_cost).masked_fill(~feasible_mask, 0.)
    cost_scale = (centered_cost.square().sum(-1, keepdim=True) / legal_count).sqrt()
    cost_scale = torch.where(cost_scale > 0, cost_scale, torch.ones_like(cost_scale))
    utilities = (-centered_cost / cost_scale).to(scores.dtype)
    teacher = utilities.masked_fill(~feasible_mask, -torch.inf).softmax(-1)
    safe_scores = scores.masked_fill(~feasible_mask, 0.)
    log_student = safe_scores.masked_fill(~feasible_mask, -torch.inf).log_softmax(-1).masked_fill(~feasible_mask, 0.)
    distribution_loss = -(teacher * log_student).sum(-1)
    centered_scores = safe_scores - safe_scores.sum(-1, keepdim=True) / legal_count
    residual = (centered_scores - utilities).masked_fill(~feasible_mask, 0.)
    fitting_loss = residual.square().sum(-1) / legal_count.squeeze(-1)
    return (distribution_loss + .1 * fitting_loss).mean()
