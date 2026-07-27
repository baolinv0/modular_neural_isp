import torch
import torch.nn as nn

from photofinishing.unpaired_chroma_heads import FrozenLUTAffineResidual


class TinyBaseLUT(nn.Module):
    def __init__(self, lut_size: int = 4):
        super().__init__()
        self._lut_size = lut_size
        self.weight = nn.Parameter(torch.tensor(0.0))
        coords = torch.linspace(-0.5, 0.5, lut_size)
        cb, cr = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("identity", torch.stack([cb, cr], dim=0).unsqueeze(0))

    def get_cbcr_lut_size(self) -> int:
        return self._lut_size

    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        return self.identity.expand(ycbcr.shape[0], -1, -1, -1) + 0.0 * self.weight


def test_affine_residual_is_identity_at_initialization():
    base = TinyBaseLUT()
    head = FrozenLUTAffineResidual(base)
    image = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        expected = base(image)
        actual = head(image)
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


def test_affine_residual_output_is_bounded_to_cbcr_domain():
    head = FrozenLUTAffineResidual(TinyBaseLUT(), matrix_limit=2.0, bias_limit=2.0)
    with torch.no_grad():
        head.matrix_raw.fill_(10.0)
        head.bias_raw.fill_(10.0)
    output = head(torch.rand(1, 3, 8, 8))
    assert output.min().item() >= -0.5
    assert output.max().item() <= 0.5
