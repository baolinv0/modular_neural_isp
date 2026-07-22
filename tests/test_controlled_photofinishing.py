import torch
from torch import nn

from photofinishing.controlled_photofinishing import ControlledLuminancePhotofinishing


class TinyGainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._input_size = 8
        self._gain_min = 0.5
        self._gain_max = 2.0
        self._gain_net = nn.Module()
        self._gain_net._net = nn.Sequential(
            nn.Conv2d(3, 4, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self._gain_net._net(x)
        gain = self._gain_min + (self._gain_max - self._gain_min) * scale
        return gain.view(-1, 1, 1, 1)


class TinyGTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._input_size = 8
        self._gtm_net = nn.Sequential(
            nn.Conv2d(3, 5, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(5, 3),
            nn.Softplus(),
        )

    def forward(self, x):
        return self._gtm_net(x)


class TinyLTM(nn.Module):
    def forward(self, x, x_gtm, training_mode=False):
        return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self._gain_net = TinyGainNet()
        self._gtm_net = TinyGTMNet()
        self._ltm_net = TinyLTM()

    @staticmethod
    def _apply_gain(x, gain):
        return x * gain

    @staticmethod
    def _apply_gtm(x, params):
        gain = params[:, 0:1, None, None] / (1.0 + params[:, 0:1, None, None])
        return torch.clamp(x * (0.5 + gain), 0.0, 1.0)

    @staticmethod
    def _apply_ltm(x_gain, x_gtm, params):
        return x_gtm


def test_alpha_zero_matches_frozen_baseline_brightness_path():
    torch.manual_seed(0)
    baseline = TinyBaseline()
    model = ControlledLuminancePhotofinishing(
        baseline, method="dual_lora", target_params_per_head=256)
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        expected = model.forward_baseline_brightness(x)["output"]
        actual = model(x, torch.zeros(2, 1))["output"]
    assert torch.equal(actual, expected)
    assert all(not parameter.requires_grad for parameter in baseline.parameters())
    assert any(parameter.requires_grad for parameter in model.control_parameters())


def test_gradients_flow_through_frozen_brightness_path_to_adapters():
    torch.manual_seed(1)
    baseline = TinyBaseline()
    model = ControlledLuminancePhotofinishing(
        baseline, method="param_residual", target_params_per_head=256)
    x = torch.rand(2, 3, 8, 8)
    output = model(x, torch.ones(2, 1))["output"]
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in model.control_parameters())
    assert all(parameter.grad is None for parameter in baseline.parameters())
