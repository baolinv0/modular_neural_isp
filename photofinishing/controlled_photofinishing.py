"""Frozen-baseline Gain/GTM brightness-control wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .brightness_ops import srgb_oetf
    from .control_adapters import build_control_adapter, count_trainable_parameters
except ImportError:
    from brightness_ops import srgb_oetf
    from control_adapters import build_control_adapter, count_trainable_parameters


@dataclass(frozen=True)
class HeadParts:
    feature_extractor: nn.Sequential
    linear: nn.Linear
    activation: nn.Module
    input_size: int
    input_channels: int


def _find_sequential(module: nn.Module, candidate_paths: tuple[str, ...]) -> nn.Sequential:
    for path in candidate_paths:
        current = module
        found = True
        for attr in path.split("."):
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found and isinstance(current, nn.Sequential):
            return current
    raise AttributeError(f"Could not find supported Sequential head in {type(module).__name__}")


def _split_parameter_head(module: nn.Module, candidate_paths: tuple[str, ...]) -> HeadParts:
    sequence = _find_sequential(module, candidate_paths)
    if len(sequence) < 2 or not isinstance(sequence[-2], nn.Linear):
        raise TypeError("Expected final Linear followed by an activation")
    first_conv = next((item for item in sequence if isinstance(item, nn.Conv2d)), None)
    if first_conv is None:
        raise TypeError("Expected a Conv2d in parameter predictor")
    input_size = int(getattr(module, "_input_size", 0))
    if input_size <= 0:
        raise ValueError("Parameter predictor must expose a positive _input_size")
    return HeadParts(
        feature_extractor=sequence[:-2],
        linear=sequence[-2],
        activation=sequence[-1],
        input_size=input_size,
        input_channels=first_conv.in_channels,
    )


def _match_channels(image: torch.Tensor, channels: int) -> torch.Tensor:
    if image.shape[1] == channels:
        return image
    if channels == 1 and image.shape[1] == 3:
        weights = image.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)
    raise ValueError(f"Cannot map {image.shape[1]} input channels to {channels}")


def freeze_baseline(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


class ControlledLuminancePhotofinishing(nn.Module):
    """Adds one of four control methods to frozen Gain and GTM parameter heads."""

    def __init__(self, baseline: nn.Module, *, method: str,
                 target_params_per_head: int = 2048) -> None:
        super().__init__()
        self.baseline = baseline
        freeze_baseline(self.baseline)
        self.method = method

        self.gain_head = _split_parameter_head(
            self.baseline._gain_net, ("_gain_net._net", "_net"))
        self.gtm_head = _split_parameter_head(
            self.baseline._gtm_net, ("_gtm_net", "_net"))
        self.gain_control = build_control_adapter(
            method, self.gain_head.linear.in_features,
            self.gain_head.linear.out_features,
            target_params=target_params_per_head)
        self.gtm_control = build_control_adapter(
            method, self.gtm_head.linear.in_features,
            self.gtm_head.linear.out_features,
            target_params=target_params_per_head)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def control_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.gain_control.parameters()
        yield from self.gtm_control.parameters()

    def trainable_parameter_count(self) -> int:
        return (count_trainable_parameters(self.gain_control)
                + count_trainable_parameters(self.gtm_control))

    @staticmethod
    def _features(image: torch.Tensor, head: HeadParts) -> torch.Tensor:
        x = _match_channels(image, head.input_channels)
        x = F.interpolate(x, size=(head.input_size, head.input_size),
                          mode="bilinear", align_corners=True)
        return head.feature_extractor(x)

    def _predict_gain(self, image: torch.Tensor, alpha: torch.Tensor):
        feature = self._features(image, self.gain_head)
        raw = self.gain_head.linear(feature)
        raw = raw + self.gain_control(feature, alpha, self.gain_head.linear)
        scale = self.gain_head.activation(raw)
        gain_net = self.baseline._gain_net
        gain = gain_net._gain_min + (gain_net._gain_max - gain_net._gain_min) * scale
        return gain.view(-1, 1, 1, 1), raw

    def _predict_gtm(self, image: torch.Tensor, alpha: torch.Tensor):
        feature = self._features(image, self.gtm_head)
        raw = self.gtm_head.linear(feature)
        raw = raw + self.gtm_control(feature, alpha, self.gtm_head.linear)
        return self.gtm_head.activation(raw), raw

    def _forward_impl(self, image: torch.Tensor, alpha: torch.Tensor,
                      training_mode: bool):
        gain, gain_raw = self._predict_gain(image, alpha)
        after_gain = self.baseline._apply_gain(image, gain)
        gtm_params, gtm_raw = self._predict_gtm(after_gain, alpha)
        after_gtm = self.baseline._apply_gtm(after_gain, gtm_params)
        ltm_params = self.baseline._ltm_net(
            after_gain, after_gtm, training_mode=training_mode)
        linear_output = self.baseline._apply_ltm(
            after_gain, after_gtm, ltm_params).clamp(0.0, 1.0)
        output = srgb_oetf(linear_output)
        return {
            "output": output,
            "linear_output": linear_output,
            "gain": gain,
            "gain_raw": gain_raw,
            "gtm_params": gtm_params,
            "gtm_raw": gtm_raw,
            "ltm_params": ltm_params,
            "lsrgb_gain": after_gain,
            "lsrgb_gtm": after_gtm,
        }

    def forward(self, image: torch.Tensor, alpha: torch.Tensor,
                *, training_mode: bool = True):
        return self._forward_impl(image, alpha, training_mode)

    @torch.no_grad()
    def forward_baseline_brightness(self, image: torch.Tensor,
                                    *, training_mode: bool = True):
        alpha = image.new_zeros((image.shape[0], 1))
        return self._forward_impl(image, alpha, training_mode)


def load_frozen_baseline_checkpoint(baseline: nn.Module, checkpoint_path: str,
                                    *, map_location="cpu", strict: bool = True):
    state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    baseline.load_state_dict(state, strict=strict)
    freeze_baseline(baseline)
    return baseline


def find_target_params_per_head(baseline: nn.Module, method: str,
                                desired_total_params: int, *,
                                min_candidate: int = 8,
                                max_candidate: int | None = None,
                                step: int = 8) -> tuple[int, int]:
    """Find the per-head request whose realized trainable count is closest."""
    if desired_total_params <= 0:
        raise ValueError("desired_total_params must be positive")
    gain_head = _split_parameter_head(
        baseline._gain_net, ("_gain_net._net", "_net"))
    gtm_head = _split_parameter_head(
        baseline._gtm_net, ("_gtm_net", "_net"))
    if max_candidate is None:
        max_candidate = max(desired_total_params * 2, 256)
    best_candidate = min_candidate
    best_count = -1
    best_error = float("inf")
    for candidate in range(min_candidate, max_candidate + 1, step):
        gain_adapter = build_control_adapter(
            method, gain_head.linear.in_features,
            gain_head.linear.out_features, target_params=candidate)
        gtm_adapter = build_control_adapter(
            method, gtm_head.linear.in_features,
            gtm_head.linear.out_features, target_params=candidate)
        count = (count_trainable_parameters(gain_adapter)
                 + count_trainable_parameters(gtm_adapter))
        error = abs(count - desired_total_params)
        if error < best_error:
            best_candidate, best_count, best_error = candidate, count, error
    return best_candidate, best_count
