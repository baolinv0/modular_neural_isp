import torch
from capture_tm.joint_objective import fixed_target, image_metrics


def test_target_keeps_hdr_highlights_distinct_and_fixed_black():
    x = torch.tensor([0., 1., 4., 16.]).reshape(1, 1, 1, 4).expand(1, 3, 2, 4)
    y = fixed_target(x)
    assert torch.equal(y[..., 0], torch.zeros_like(y[..., 0]))
    assert (y[..., 1:] > y[..., :-1]).all()
    assert y.max() < 1
    assert fixed_target(x, render_ev=1.)[..., 1].mean() > y[..., 1].mean()


def test_quality_cost_penalizes_missing_information_and_has_image_gradient():
    target = torch.full((2, 3, 8, 8), .5)
    image = torch.full_like(target, .4, requires_grad=True)
    missing = torch.zeros_like(target, dtype=torch.bool)
    missing[1] = True
    metrics = image_metrics(image, target, missing)
    assert torch.allclose(metrics['cost'][1] - metrics['cost'][0], torch.tensor(.02))
    metrics['cost'].sum().backward()
    assert image.grad.abs().sum() > 0


def test_detail_metric_detects_blur_with_same_mean_brightness():
    target = torch.zeros(1, 3, 8, 8)
    target[..., ::2] = 1
    blurred = torch.full_like(target, .5)
    missing = torch.zeros_like(target, dtype=torch.bool)
    sharp = image_metrics(target, target, missing)
    blur = image_metrics(blurred, target, missing)
    assert sharp['cost'].item() == 0
    assert blur['detail_mae'].item() > .4
