import torch
import torch.nn as nn

from photofinishing.unpaired_chroma_heads import FrozenLUTAffineResidual


class TinyBaseLUT(nn.Module):
    def __init__(self, lut_size: int = 4, scale: float = 1.0):
        super().__init__()
        self._lut_size = lut_size
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.tensor(0.0))
        coords = torch.linspace(-0.5, 0.5, lut_size)
        cb, cr = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("identity", torch.stack([cb, cr], dim=0).unsqueeze(0))

    def get_cbcr_lut_size(self) -> int:
        return self._lut_size

    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        return self.scale * self.identity.expand(ycbcr.shape[0], -1, -1, -1) + 0.0 * self.weight


def test_affine_residual_is_identity_at_initialization_even_outside_nominal_gamut():
    base = TinyBaseLUT(scale=2.0)
    head = FrozenLUTAffineResidual(base)
    image = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        expected = base(image)
        actual = head(image)
    assert expected.abs().max().item() > 0.5
    assert torch.equal(actual, expected)


def test_affine_residual_has_exactly_six_trainable_scalars_and_freezes_base():
    head = FrozenLUTAffineResidual(TinyBaseLUT())
    trainable = {name: parameter for name, parameter in head.named_parameters() if parameter.requires_grad}
    assert set(trainable) == {"matrix_raw", "bias_raw"}
    assert sum(parameter.numel() for parameter in trainable.values()) == 6
    assert all(not parameter.requires_grad for parameter in head.base_lut_net.parameters())
    assert head.get_num_of_params() == 6


def test_affine_residual_keeps_frozen_base_in_eval_mode():
    head = FrozenLUTAffineResidual(TinyBaseLUT())
    head.train()
    assert head.training
    assert not head.base_lut_net.training


def test_affine_residual_gradients_reach_only_six_parameters():
    head = FrozenLUTAffineResidual(TinyBaseLUT())
    image = torch.rand(1, 3, 8, 8)
    loss = head(image).square().mean()
    loss.backward()
    assert head.matrix_raw.grad is not None
    assert head.bias_raw.grad is not None
    assert all(parameter.grad is None for parameter in head.base_lut_net.parameters())


def test_affine_parameters_are_bounded_without_clamping_base_lut():
    head = FrozenLUTAffineResidual(TinyBaseLUT(), matrix_limit=0.15, bias_limit=0.05)
    with torch.no_grad():
        head.matrix_raw.fill_(100.0)
        head.bias_raw.fill_(100.0)
    matrix_delta = head.effective_matrix() - torch.eye(2)
    bias = head.effective_bias()
    assert matrix_delta.abs().max().item() <= 0.150001
    assert bias.abs().max().item() <= 0.050001
