import torch
from torch import nn

from photofinishing.controlled_photofinishing import (
    ControlledLuminancePhotofinishing,
    find_target_params_per_head,
)


class _Predictor(nn.Module):
    def __init__(self, feature_dim, output_dim, activation):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, feature_dim, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, output_dim),
            activation,
        )


class _Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self._input_size = 8
        self._gain_min = 0.5
        self._gain_max = 2.0
        self._gain_net = nn.Module()
        self._gain_net._net = _Predictor(16, 1, nn.Sigmoid()).seq


class _GTM(nn.Module):
    def __init__(self):
        super().__init__()
        self._input_size = 8
        self._gtm_net = _Predictor(32, 3, nn.Softplus()).seq


class _LTM(nn.Module):
    def forward(self, x, x_gtm, training_mode=False):
        return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])


class RealisticFakeBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self._gain_net = _Gain()
        self._gtm_net = _GTM()
        self._ltm_net = _LTM()

    @staticmethod
    def _apply_gain(x, gain):
        return x * gain

    @staticmethod
    def _apply_gtm(x, params):
        return x

    @staticmethod
    def _apply_ltm(x, x_gtm, params):
        return x_gtm


def test_all_variants_match_common_parameter_budget_within_ten_percent():
    budget = 2048
    counts = []
    for method in ("param_residual", "parallel_adapter", "film", "dual_lora"):
        baseline = RealisticFakeBaseline()
        request, _ = find_target_params_per_head(baseline, method, budget)
        model = ControlledLuminancePhotofinishing(
            baseline, method=method, target_params_per_head=request)
        counts.append(model.trainable_parameter_count())
    mean_count = sum(counts) / len(counts)
    assert max(abs(count - mean_count) / mean_count for count in counts) <= 0.10
