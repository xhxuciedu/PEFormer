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


def simplex_loss(logits: torch.Tensor, edited: torch.Tensor, indel: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy against the observed 3-way outcome proportions.

    Each assayed locus ends in exactly one of {unedited, correctly edited, indel}; the
    measured values are the proportions of reads in each class, i.e. a point on the
    2-simplex. Soft cross-entropy `-Σ_k y_k log p_k` is the natural likelihood for
    proportion data and supervises on the indel channel that a scalar efficiency head
    discards entirely (indel is nonzero for 60% of the corpus and only weakly correlated
    with editing, r=0.25, so it carries genuinely independent signal).

    Rows whose indel is unobserved (NaN) are still usable: for those we marginalise the
    indel class out and score the remaining binary unedited-vs-edited decision, which is
    the correct likelihood under "indel unknown" rather than the wrong one implied by
    imputing indel=0. OptiPrime's official training mix leaves indel unmeasured for the
    Kim rows and the Schwank diverse libraries (42.5% of the corpus), so this matters.
    """
    edited = edited.clamp(0, 1)
    observed = ~torch.isnan(indel)
    indel_f = torch.nan_to_num(indel, nan=0.0).clamp(0, 1)
    unedited = (1.0 - edited - indel_f).clamp(min=0.0)

    log_p = torch.log_softmax(logits, dim=-1)

    # 3-way soft cross-entropy where indel was measured.
    y = torch.stack([unedited, edited, indel_f], dim=-1)
    y = y / y.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    loss_full = -(y * log_p).sum(dim=-1)

    # Where it was not, renormalise over {unedited, edited} only.
    log_p2 = torch.log_softmax(logits[..., :2], dim=-1)
    un2 = (1.0 - edited).clamp(min=0.0)
    y2 = torch.stack([un2, edited], dim=-1)
    y2 = y2 / y2.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    loss_marg = -(y2 * log_p2).sum(dim=-1)

    return torch.where(observed, loss_full, loss_marg).mean()


@dataclass
class LossWeights:
    lambda_rank: float = 0.25
    huber_beta: float = 0.1
    min_pair_diff: float = 0.02
    regression_space: str = "raw"  # "raw" or "logit" (task spec §19 comparison)
    outcome_head: str = "scalar"  # "scalar" or "simplex"


def total_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    pairs_i: torch.Tensor,
    pairs_j: torch.Tensor,
    weights: LossWeights,
    rank_score: torch.Tensor | None = None,
    target_indel: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """`score` is the model's raw head output: (B,) for the scalar head, (B,3) logits for
    the simplex head. `rank_score` is the monotone scalar used for ranking (supplied by
    the model, since it differs between head types)."""
    if weights.outcome_head == "simplex":
        assert target_indel is not None, "simplex head requires target_indel"
        l_reg = simplex_loss(score, target, target_indel)
    else:
        l_reg = regression_loss(
            score, target, huber_beta=weights.huber_beta, space=weights.regression_space
        )
    rs = score if rank_score is None else rank_score
    l_rank = ranking_loss(rs, target, pairs_i, pairs_j, min_diff=weights.min_pair_diff)
    loss = l_reg + weights.lambda_rank * l_rank
    return loss, {"loss": loss.item(), "reg": l_reg.item(), "rank": l_rank.item()}
