"""Regression + within-target ranking losses (task spec §19-21)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def regression_loss(score: torch.Tensor, target: torch.Tensor, huber_beta: float = 0.1) -> torch.Tensor:
    """Smooth L1 (Huber) loss between predicted efficiency (sigmoid of score) and the
    measured efficiency, both in [0,1]."""
    pred = torch.sigmoid(score)
    return F.smooth_l1_loss(pred, target, beta=huber_beta)


def ranking_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    pairs_i: torch.Tensor,
    pairs_j: torch.Tensor,
    min_diff: float = 0.02,
) -> torch.Tensor:
    """RankNet pairwise loss over a set of (i, j) index pairs within `score`/`target`,
    oriented so that target[i] > target[j] for every pair. Pairs with a true-efficiency
    gap below `min_diff` are excluded upstream by the sampler, not here.

    L_ij = log(1 + exp(-(s_i - s_j)))
    """
    if pairs_i.numel() == 0:
        return score.new_zeros(())
    si = score[pairs_i]
    sj = score[pairs_j]
    return F.softplus(-(si - sj)).mean()


@dataclass
class LossWeights:
    lambda_rank: float = 0.25
    huber_beta: float = 0.1
    min_pair_diff: float = 0.02


def total_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    pairs_i: torch.Tensor,
    pairs_j: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_reg = regression_loss(score, target, huber_beta=weights.huber_beta)
    l_rank = ranking_loss(score, target, pairs_i, pairs_j, min_diff=weights.min_pair_diff)
    loss = l_reg + weights.lambda_rank * l_rank
    return loss, {"loss": loss.item(), "reg": l_reg.item(), "rank": l_rank.item()}
