import torch

from photofinishing.control_losses import BrightnessControlLoss, pairwise_monotonic_loss


def test_pairwise_monotonic_loss_penalizes_reversal_only():
    low = torch.full((2, 3, 4, 4), 0.4)
    high = torch.full((2, 3, 4, 4), 0.6)
    assert pairwise_monotonic_loss(low, high, margin=0.0).item() == 0.0
    assert pairwise_monotonic_loss(high, low, margin=0.0).item() > 0.0


def test_brightness_control_loss_is_zero_for_exact_targets_and_anchor():
    loss_fn = BrightnessControlLoss(
        log_luma_weight=1.0,
        gradient_weight=1.0,
        monotonic_weight=1.0,
        anchor_weight=1.0,
    )
    low = torch.full((1, 3, 4, 4), 0.2)
    high = torch.full((1, 3, 4, 4), 0.5)
    baseline = torch.full((1, 3, 4, 4), 0.3)
    loss, details = loss_fn(low, high, low, high, baseline, baseline)
    assert loss.item() == 0.0
    assert details["total"] == 0.0
