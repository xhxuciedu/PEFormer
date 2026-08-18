"""Regression + ranking loss sanity checks (task spec §40)."""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.training.losses import (
    LossWeights,
    ranking_loss,
    regression_loss,
    simplex_loss,
    total_loss,
)


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


def test_simplex_loss_minimized_at_true_proportions():
    from pe_rankformer.training.losses import simplex_loss

    edited = torch.tensor([0.3])
    indel = torch.tensor([0.2])
    # unedited = 0.5 -> perfect logits are log of the true proportions
    perfect = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    wrong = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    assert simplex_loss(perfect, edited, indel) < simplex_loss(wrong, edited, indel)


def test_simplex_loss_handles_edited_plus_indel_above_one():
    from pe_rankformer.training.losses import simplex_loss

    # clipped/renormalized rather than producing a negative 'unedited' class
    loss = simplex_loss(torch.zeros(1, 3), torch.tensor([0.8]), torch.tensor([0.5]))
    assert torch.isfinite(loss)


def test_simplex_loss_marginalises_unobserved_indel():
    """OptiPrime's official mix leaves indel unmeasured for 42.5% of rows; those must
    still contribute a well-defined gradient rather than being imputed to indel=0."""
    import math

    torch.manual_seed(0)
    logits = torch.randn(5, 3, requires_grad=True)
    edited = torch.tensor([0.3, 0.5, 0.2, 0.4, 0.1])
    indel = torch.tensor([0.1, math.nan, 0.05, math.nan, 0.2])

    loss = simplex_loss(logits, edited, indel)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_simplex_loss_all_indel_unobserved_still_trains():
    torch.manual_seed(0)
    logits = torch.randn(4, 3, requires_grad=True)
    edited = torch.tensor([0.3, 0.5, 0.2, 0.4])
    loss = simplex_loss(logits, edited, torch.full((4,), float("nan")))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum() > 0).item()


def test_imputing_zero_indel_differs_from_marginalising():
    """Guards the distinction the masking exists to preserve."""
    torch.manual_seed(0)
    logits = torch.randn(3, 3)
    edited = torch.tensor([0.3, 0.5, 0.2])
    marginalised = simplex_loss(logits, edited, torch.full((3,), float("nan")))
    imputed = simplex_loss(logits, edited, torch.zeros(3))
    assert not torch.isclose(marginalised, imputed)
