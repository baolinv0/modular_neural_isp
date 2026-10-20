"""Protect causal temporal AE, physical plan order, and joint quality gradients."""
import importlib

import pytest
import torch
from torch import nn


def implementation():
    try:
        return importlib.import_module("capture_tm.learned_policy")
    except ModuleNotFoundError as error:
        pytest.fail(f"Temporal AE and joint expectation are not implemented: {error}")


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations(batch=2, plan_length=1):
    generator = torch.Generator().manual_seed(17)
    return (
        torch.rand(batch, 3, 3, 16, 16, generator=generator),
        torch.zeros(batch, 3, 3),
        torch.randn(batch, 4, plan_length, 3, generator=generator),
        torch.tensor([[True, False, True, True]]).expand(batch, -1).clone(),
    )


@pytest.mark.parametrize("plan_length", [1, 3])
def test_candidate_reordering_preserves_scores_and_legal_choice(plan_length):
    """An action-index classifier or unmasked output would break this contract."""
    policy = implementation().TemporalExposurePolicy(width=8).eval()
    previews, state, candidates, mask = observations(plan_length=plan_length)
    scores = policy(previews, state, candidates, mask)
    order = torch.tensor([2, 0, 3, 1])
    reordered = policy(previews, state, candidates[:, order], mask[:, order])
    torch.testing.assert_close(reordered, scores[:, order])
    assert torch.isneginf(scores[:, 1]).all()
    assert mask.gather(1, scores.argmax(-1, keepdim=True)).all()


def test_hdr_plan_order_is_observable():
    """Averaging action embeddings would erase which exposure is reference-time."""
    torch.manual_seed(19)
    policy = implementation().TemporalExposurePolicy(width=8).eval()
    previews, state, _, _ = observations(batch=1)
    candidates = torch.tensor([[[[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]],
                                [[2., 0., 0.], [0., 0., 0.], [-2., 0., 0.]]]])
    scores = policy(previews, state, candidates, torch.ones(1, 2, dtype=torch.bool))
    assert (scores[0, 0] - scores[0, 1]).abs() > 1e-6


def test_equal_histogram_histories_with_same_latest_frame_reveal_motion():
    """Latest-only CNNs and per-frame histograms cannot distinguish this pair."""
    torch.manual_seed(23)
    policy = implementation().TemporalExposurePolicy(width=8).eval()
    frame = torch.zeros(1, 3, 16, 16)
    frame[:, :, 5:9, 9:13] = .8
    static = frame[:, None].expand(-1, 3, -1, -1, -1).clone()
    moving = static.clone()
    moving[:, 0] = torch.roll(frame, shifts=-6, dims=-1)
    moving[:, 1] = torch.roll(frame, shifts=-3, dims=-1)
    previews = torch.cat((static, moving))
    candidates = torch.tensor([[[[-2., 2., 0.]], [[0., 0., 0.]]]]).expand(2, -1, -1, -1)
    scores = policy(previews, torch.zeros(2, 3, 3), candidates, torch.ones(2, 2, dtype=torch.bool))
    preference = scores[:, 0] - scores[:, 1]
    assert (preference[0] - preference[1]).abs() > 1e-6


def test_observed_effective_metadata_affects_relative_action_preference():
    torch.manual_seed(11)
    policy = implementation().TemporalExposurePolicy(width=8).eval()
    previews, state, candidates, mask = observations(batch=1)
    before = policy(previews, state, candidates, mask)
    state[..., 0] = 2.
    after = policy(previews, state, candidates, mask)
    assert ((before[:, 0] - before[:, 2]) - (after[:, 0] - after[:, 2])).abs() > 1e-6


def test_expectation_is_physical_candidate_average_and_masks_nan_cost_before_math():
    """Detaching costs, averaging images, or multiplying 0*NaN would fail."""
    scores = torch.tensor([[0., 0., float("nan")], [0., 0., -float("inf")]], requires_grad=True)
    costs = torch.tensor([[1., 3., float("nan")], [2., 6., float("inf")]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss = implementation().expected_quality_loss(scores, costs, mask)
    torch.testing.assert_close(loss, torch.tensor(3.))
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[-.25, .25, 0.], [-.5, .5, 0.]]))
    torch.testing.assert_close(costs.grad, torch.tensor([[.25, .25, 0.], [.25, .25, 0.]]))


def test_joint_expectation_updates_real_ae_and_tm_parameters():
    torch.manual_seed(31)
    policy = implementation().TemporalExposurePolicy(width=8)
    tm = nn.Linear(1, 1)
    previews, state, candidates, mask = observations()
    scores = policy(previews, state, candidates, mask)
    candidate_renders = tm(candidates[:, :, 0, :1]).squeeze(-1)
    costs = (candidate_renders - .2).square()
    implementation().expected_quality_loss(scores, costs, mask).backward()
    for model in (policy, tm):
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(value).all() for value in gradients)
        assert sum(value.abs().sum() for value in gradients) > 0


def test_warmstart_separates_small_cost_gaps_and_does_not_train_tm():
    """A diffuse utility target can fail to teach the physically best action."""
    scores = torch.zeros(1, 2, requires_grad=True)
    costs = torch.tensor([[1., 1.000001]], requires_grad=True)
    implementation().warmstart_loss(scores, costs, torch.ones(1, 2, dtype=torch.bool)).backward()
    assert scores.grad[0, 0] < -.4
    assert scores.grad[0, 1] > .4
    assert costs.grad is None


def test_warmstart_does_not_invent_preference_between_exact_ties():
    scores = torch.zeros(1, 3, requires_grad=True)
    implementation().warmstart_loss(scores, torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool)).backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores), atol=1e-7, rtol=0.)


def test_policy_fits_variable_best_action_instead_of_constant_collapse():
    torch.manual_seed(7)
    module = implementation()
    policy = module.TemporalExposurePolicy(width=8)
    brightness = torch.tensor([.04, .07, .10, .13, .70, .76, .83, .90])
    previews = brightness[:, None, None, None, None].expand(-1, 3, 3, 12, 12)
    state = torch.zeros(8, 3, 3)
    candidates = torch.tensor([[[[-1., 0., 0.]], [[1., 0., 0.]]]]).expand(8, -1, -1, -1)
    mask = torch.ones(8, 2, dtype=torch.bool)
    costs = torch.tensor([[1.00001, 1.]] * 4 + [[1., 1.00001]] * 4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.004)
    for _ in range(80):
        optimizer.zero_grad()
        loss = module.warmstart_loss(policy(previews, state, candidates, mask), costs, mask)
        loss.backward()
        optimizer.step()
    prediction = policy(previews, state, candidates, mask).argmax(-1)
    assert prediction.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("problem", ["all_illegal", "wrong_plan_length", "unbounded_preview", "bad_history"])
def test_policy_rejects_unusable_observations_and_plans(problem):
    module = implementation()
    policy = module.TemporalExposurePolicy(width=8)
    previews, state, candidates, mask = observations()
    if problem == "all_illegal":
        mask[0] = False
    elif problem == "wrong_plan_length":
        candidates = candidates.expand(-1, -1, 2, -1)
    elif problem == "unbounded_preview":
        previews[0, 0, 0, 0, 0] = 1.1
    else:
        state = state[:, :2]
    with pytest.raises(ValueError):
        policy(previews, state, candidates, mask)


@pytest.mark.parametrize("temperature", [0., -1., float("nan")])
def test_expected_quality_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        implementation().expected_quality_loss(torch.zeros(1, 2), torch.ones(1, 2),
                                               torch.ones(1, 2, dtype=torch.bool), temperature)


def test_feasible_nonfinite_quality_fails_explicitly():
    with pytest.raises(ValueError, match="finite"):
        implementation().expected_quality_loss(torch.zeros(1, 2), torch.tensor([[1., float("nan")]]),
                                               torch.ones(1, 2, dtype=torch.bool))
