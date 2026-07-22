import torch

from photofinishing.control_adapters import build_control_adapter


METHODS = ["param_residual", "parallel_adapter", "film", "dual_lora"]


def test_all_adapters_are_exactly_zero_at_alpha_zero():
    feature = torch.randn(4, 16)
    base_head = torch.nn.Linear(16, 3, bias=True)
    alpha = torch.zeros(4, 1)
    for method in METHODS:
        adapter = build_control_adapter(method, 16, 3, target_params=512)
        delta = adapter(feature, alpha, base_head)
        assert delta.shape == (4, 3)
        assert torch.equal(delta, torch.zeros_like(delta))


def test_positive_and_negative_branches_are_independent():
    feature = torch.randn(2, 16)
    base_head = torch.nn.Linear(16, 3)
    for method in METHODS:
        adapter = build_control_adapter(method, 16, 3, target_params=512)
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if "pos" in name:
                    parameter.fill_(0.1)
                elif "neg" in name:
                    parameter.fill_(0.2)
        pos = adapter(feature, torch.ones(2, 1), base_head)
        neg = adapter(feature, -torch.ones(2, 1), base_head)
        assert not torch.allclose(pos, neg)
