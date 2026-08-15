"""Regression + ranking loss sanity checks (task spec §40)."""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.training.losses import LossWeights, ranking_loss, regression_loss, total_loss


def test_regression_loss_zero_for_perfect_prediction():
    # sigmoid(score) == target exactly
    target = torch.tensor([0.3, 0.7])
    score = torch.logit(target)
    loss = regression_loss(score, target)
    assert loss.item() < 1e-4


def test_regression_loss_positive_for_wrong_prediction():
    target = torch.tensor([0.9])
    score = torch.logit(torch.tensor([0.1]))
    loss = regression_loss(score, target)
    assert loss.item() > 0.1


def test_ranking_loss_zero_pairs_returns_zero():
    score = torch.tensor([0.1, 0.2, 0.3])
    target = torch.tensor([0.1, 0.5, 0.9])
    empty = torch.empty(0, dtype=torch.long)
    loss = ranking_loss(score, target, empty, empty)
    assert loss.item() == 0.0


def test_ranking_loss_small_when_correctly_ordered():
    # score already respects target order with a wide margin
    score = torch.tensor([-5.0, 5.0])
    target = torch.tensor([0.1, 0.9])
    i = torch.tensor([1])  # higher target
    j = torch.tensor([0])  # lower target
    loss = ranking_loss(score, target, i, j)
    assert loss.item() < 0.01


def test_ranking_loss_large_when_misordered():
    score = torch.tensor([5.0, -5.0])  # backwards: low-target item has high score
    target = torch.tensor([0.1, 0.9])
    i = torch.tensor([1])
    j = torch.tensor([0])
    loss = ranking_loss(score, target, i, j)
    assert loss.item() > 1.0


def test_total_loss_combines_both_terms():
    score = torch.tensor([-5.0, 5.0], requires_grad=True)
    target = torch.tensor([0.1, 0.9])
    i = torch.tensor([1])
    j = torch.tensor([0])
    weights = LossWeights(lambda_rank=0.25)
    loss, parts = total_loss(score, target, i, j, weights)
    assert parts["loss"] == pytest.approx(parts["reg"] + 0.25 * parts["rank"], abs=1e-6)
    loss.backward()
    assert score.grad is not None


def test_logit_space_regression_loss_zero_for_perfect_prediction():
    target = torch.tensor([0.3, 0.7])
    score = torch.logit(target)  # raw score == logit(target)
    loss = regression_loss(score, target, space="logit")
    assert loss.item() < 1e-4


def test_logit_space_clips_extreme_targets():
    # y=0 would be logit(-inf); clipping must keep the loss finite
    target = torch.tensor([0.0, 1.0])
    score = torch.tensor([0.0, 0.0])
    loss = regression_loss(score, target, space="logit")
    assert torch.isfinite(loss)


def test_unknown_regression_space_raises():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        regression_loss(torch.tensor([0.0]), torch.tensor([0.5]), space="nonsense")
