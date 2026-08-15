"""Regression + within-target ranking losses (task spec §19-21)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def regression_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    huber_beta: float = 0.1,
    space: str = "raw",
    logit_clip: float = 0.005,
) -> torch.Tensor:
    """Huber regression loss, in either raw efficiency space or clipped-logit space.

    `space='raw'`   : Huber(sigmoid(score), y)      -- y in [0,1].
    `space='logit'` : Huber(score, logit(clip(y)))  -- compares pre-sigmoid scores
                      directly against the logit-transformed target (task spec §19,
                      which asks for both to be compared on validation).

    The corpus is heavily zero-inflated (27% of rows below 0.01 efficiency), so raw-space
    Huber spends most of its gradient budget on an easy near-zero mass. Logit space
    spreads that mass out; whether that helps or just amplifies measurement noise at the
    low end is an empirical question, answered in reports/pilot_results.md.
    """
    if space == "raw":
        pred = torch.sigmoid(score)
        return F.smooth_l1_loss(pred, target, beta=huber_beta)
    if space == "logit":
        y = target.clamp(logit_clip, 1.0 - logit_clip)
        target_logit = torch.log(y / (1 - y))
        return F.smooth_l1_loss(score, target_logit, beta=1.0)
    raise ValueError(f"unknown regression space: {space!r}")


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
    regression_space: str = "raw"  # "raw" or "logit" (task spec §19 comparison)


def total_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    pairs_i: torch.Tensor,
    pairs_j: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_reg = regression_loss(
        score, target, huber_beta=weights.huber_beta, space=weights.regression_space
    )
    l_rank = ranking_loss(score, target, pairs_i, pairs_j, min_diff=weights.min_pair_diff)
    loss = l_reg + weights.lambda_rank * l_rank
    return loss, {"loss": loss.item(), "reg": l_reg.item(), "rank": l_rank.item()}
