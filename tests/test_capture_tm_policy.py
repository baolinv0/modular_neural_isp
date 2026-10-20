"""Policy tests catch illegal-action selection, metadata leaks and dead training gradients."""
import importlib
from types import SimpleNamespace

import pytest
import torch


def module():
    try:
        return importlib.import_module("capture_tm.policy")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Required policy implementation is absent: {exc}")


def inputs(batch=2, history=3, candidates=4):
    generator = torch.Generator().manual_seed(5)
    return (
        torch.rand(batch, history, 3, 16, 16, generator=generator),
        torch.randn(batch, history, 3, generator=generator),
        torch.randn(batch, candidates, 3, generator=generator),
        torch.ones(batch, candidates, dtype=torch.bool),
    )


def test_action_features_use_one_physical_log2_convention():
    actions = [SimpleNamespace(exposure_s=1 / 60, analog_gain=4, digital_gain=.5)]
    features = module().action_features(actions, 1 / 120)
    torch.testing.assert_close(features, torch.tensor([[1., 2., -1.]]))


@pytest.mark.parametrize("field,value", [("exposure_s", 0), ("analog_gain", float("nan")), ("digital_gain", -1)])
def test_action_features_reject_nonphysical_metadata(field, value):
    action = dict(exposure_s=1 / 120, analog_gain=1., digital_gain=1.)
    action[field] = value
    with pytest.raises(ValueError):
        module().action_features([SimpleNamespace(**action)], 1 / 120)


def test_shared_scorer_handles_dynamic_candidate_count_and_permutation():
    policy = module().ExposurePolicy(4, width=8).eval()
    preview, state, candidates, legal = inputs()
    first = policy(preview, state, candidates, legal)
    permutation = torch.tensor([2, 0, 3, 1])
    reordered = policy(preview, state, candidates[:, permutation], legal[:, permutation])
    torch.testing.assert_close(reordered, first[:, permutation])
    subset = policy(preview, state, candidates[:, :2], legal[:, :2])
    torch.testing.assert_close(subset, first[:, :2])


def test_invalid_actions_are_masked_and_cannot_win():
    policy = module().ExposurePolicy(4, width=8)
    preview, state, candidates, legal = inputs()
    legal[:, 1:] = False
    scores = policy(preview, state, candidates, legal)
    assert torch.isfinite(scores[:, 0]).all()
    assert torch.isneginf(scores[:, 1:]).all()
    assert (scores.argmax(-1) == 0).all()


def test_any_all_invalid_batch_row_fails_explicitly():
    policy = module().ExposurePolicy(4, width=8)
    values = list(inputs())
    values[3][1] = False
    with pytest.raises(ValueError, match="feasible"):
        policy(*values)


@pytest.mark.parametrize("corrupt", ["state_history", "candidate_width", "mask_shape", "mask_type", "preview_range", "nan_state", "too_long"])
def test_forward_validates_aligned_causal_observation_shapes(corrupt):
    policy = module().ExposurePolicy(4, width=8)
    values = list(inputs())
    if corrupt == "state_history":
        values[1] = values[1][:, :2]
    elif corrupt == "candidate_width":
        values[2] = values[2][..., :2]
    elif corrupt == "mask_shape":
        values[3] = values[3][:, :2]
    elif corrupt == "mask_type":
        values[3] = values[3].float()
    elif corrupt == "preview_range":
        values[0][0, 0, 0, 0, 0] = 1.1
    elif corrupt == "nan_state":
        values[1][0, 0, 0] = float("nan")
    elif corrupt == "too_long":
        values = list(inputs(history=4))
    with pytest.raises(ValueError):
        policy(*values)


@pytest.mark.parametrize("stage", [torch.ones(2, 3), torch.tensor([[4., 3.], [1., 3.]]), torch.tensor([[0., 0.], [1., 3.]]), torch.tensor([[0., float("nan")], [1., 3.]])])
def test_stage_validation_rejects_malformed_budget(stage):
    policy = module().ExposurePolicy(4, width=8)
    with pytest.raises(ValueError):
        policy(*inputs(), stage=stage)


def test_image_only_ablation_does_not_consume_auxiliary_metadata():
    policy = module().ExposurePolicy(4, width=8, use_auxiliary=False).eval()
    preview, state, candidates, legal = inputs()
    first = policy(preview, state, candidates, legal, stage=torch.tensor([[0., 3.], [0., 3.]]))
    second = policy(preview, state + 10, candidates, legal, stage=torch.tensor([[2., 3.], [2., 3.]]))
    torch.testing.assert_close(first, second)


def test_image_only_history_ablation_has_matched_parameter_capacity():
    full = module().ExposurePolicy(4, width=8)
    image_history = module().ExposurePolicy(4, width=8, use_auxiliary=False)
    assert sum(value.numel() for value in full.parameters()) == sum(value.numel() for value in image_history.parameters())
    assert set(full.state_dict()) == set(image_history.state_dict())


def test_image_only_history_still_uses_earlier_observed_images():
    torch.manual_seed(7)
    policy = module().ExposurePolicy(4, width=8, use_auxiliary=False).eval()
    preview, state, candidates, legal = inputs()
    first_history = preview.clone()
    second_history = preview.clone()
    first_history[:, :-1] = 0.
    second_history[:, :-1] = 1.
    first = policy(first_history, state, candidates, legal)
    second = policy(second_history, state, candidates, legal)
    # The latest image and physical candidate grid are identical; only past
    # visible images differ. A latest-image-only baseline would return equality.
    assert not torch.allclose(first, second, atol=1e-7, rtol=0.)


def test_full_policy_and_ranking_have_image_and_metadata_gradients():
    policy = module().ExposurePolicy(4, width=8)
    preview, state, candidates, legal = inputs()
    preview.requires_grad_()
    state.requires_grad_()
    candidates.requires_grad_()
    scores = policy(preview, state, candidates, legal)
    loss = module().ranking_loss(scores, torch.tensor([[0., 2., 3., 1.], [3., 0., 2., 1.]]), legal)
    loss.backward()
    for value in (preview, state, candidates):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_ranking_loss_ignores_nan_invalid_costs_and_invalid_gradients():
    scores = torch.tensor([[0., 0., float("-inf")]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    loss = module().ranking_loss(scores, torch.tensor([[0., 2., float("nan")]]), legal)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0
    assert scores.grad[0, 2] == 0


def test_ranking_loss_rejects_nonfinite_feasible_teacher_cost():
    with pytest.raises(ValueError, match="finite"):
        module().ranking_loss(torch.zeros(1, 2), torch.tensor([[0., float("inf")]]), torch.ones(1, 2, dtype=torch.bool))


def test_single_candidate_loss_is_finite_and_differentiable():
    scores = torch.tensor([[4.]], requires_grad=True)
    loss = module().ranking_loss(scores, torch.tensor([[2.]]), torch.tensor([[True]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("costs", [torch.tensor([[0., 1e30]]), torch.tensor([[0., 1e-300]], dtype=torch.float64)])
def test_cost_scale_does_not_erase_finite_oracle_order(costs):
    scores = torch.zeros(1, 2, requires_grad=True)
    loss = module().ranking_loss(scores, costs, torch.ones(1, 2, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad[0, 0] < -.1
    assert scores.grad[0, 1] > .1
