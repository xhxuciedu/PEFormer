"""CORAL-style ordinal outcome head (round 4).

The head estimates the target's normalised rank via K-1 cumulative indicators. The
properties worth pinning down are that the score is genuinely *monotone* in the
predicted quantile (otherwise it is not metric-matched at all), that the thresholds
cannot be silently misconfigured, and that adding the head leaves every existing
checkpoint loadable.
"""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.training.losses import LossWeights, ordinal_loss, total_loss

THR = (0.01, 0.05, 0.1, 0.3)


def _cfg(**kw) -> PERankFormerConfig:
    base = dict(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=1,
        context_fields=("cell_type",), context_vocab_sizes={"cell_type": 3}, context_embed_dim=8,
    )
    base.update(kw)
    return PERankFormerConfig(**base)


def _batch(n: int = 6) -> dict[str, torch.Tensor]:
    cfg = _cfg()
    return {
        "edit_ids": torch.randint(1, 5, (n, cfg.edit_seq_len)),
        "peg_nuc_ids": torch.randint(1, 5, (n, cfg.peg_seq_len)),
        "peg_seg_ids": torch.randint(0, 3, (n, cfg.peg_seq_len)),
        "ctx_cell_type": torch.randint(0, 3, (n,)),
    }


def test_head_emits_one_logit_per_threshold():
    model = PERankFormer(_cfg(outcome_head="ordinal", ordinal_thresholds=THR))
    out = model(_batch())
    assert out.shape == (6, len(THR))


def test_score_is_monotone_in_the_logits():
    """The whole justification for this head is that its score orders rows by predicted
    quantile; raising any threshold logit must never lower the score."""
    model = PERankFormer(_cfg(outcome_head="ordinal", ordinal_thresholds=THR))
    out = torch.randn(5, len(THR))
    base = model.efficiency_from_output(out)
    for k in range(len(THR)):
        bumped = out.clone()
        bumped[:, k] += 1.0
        assert (model.efficiency_from_output(bumped) >= base - 1e-6).all()


def test_score_in_unit_interval_and_matches_ranking_score():
    model = PERankFormer(_cfg(outcome_head="ordinal", ordinal_thresholds=THR))
    out = torch.randn(32, len(THR)) * 5
    eff = model.efficiency_from_output(out)
    assert (eff >= 0).all() and (eff <= 1).all()
    # Ranking and reported efficiency must induce the same order, or the loss optimises
    # a different quantity than the one evaluated.
    assert torch.allclose(eff, model.ranking_score(out))


def test_loss_targets_are_the_cumulative_indicators():
    target = torch.tensor([0.0, 0.07, 0.5])
    thr = torch.tensor(THR)
    # A perfect predictor of 1[y > t_k] must drive BCE toward zero.
    y = (target.unsqueeze(-1) > thr.unsqueeze(0)).float()
    confident = (y * 2 - 1) * 20.0
    assert ordinal_loss(confident, target, thr).item() < 1e-6
    # ...and the exactly-wrong predictor must be heavily penalised.
    assert ordinal_loss(-confident, target, thr).item() > 10.0


def test_row_at_a_threshold_counts_as_below_it():
    target = torch.tensor([0.05])
    thr = torch.tensor([0.05])
    y_implied = (target.unsqueeze(-1) > thr.unsqueeze(0)).float()
    assert y_implied.item() == 0.0


def test_total_loss_routes_to_ordinal_and_backprops():
    model = PERankFormer(_cfg(outcome_head="ordinal", ordinal_thresholds=THR))
    out = model(_batch(4))
    target = torch.tensor([0.0, 0.02, 0.2, 0.6])
    w = LossWeights(outcome_head="ordinal", ordinal_thresholds=torch.tensor(THR))
    loss, parts = total_loss(
        out, target, torch.tensor([0, 1]), torch.tensor([2, 3]), w,
        rank_score=model.ranking_score(out),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.head[-1].weight.grad).all()
    assert parts["reg"] > 0


def test_thresholds_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        _cfg(outcome_head="ordinal", ordinal_thresholds=(0.1, 0.1, 0.2))
    with pytest.raises(ValueError, match="strictly increasing"):
        _cfg(outcome_head="ordinal", ordinal_thresholds=(0.3, 0.1))


def test_ordinal_requires_thresholds_and_vice_versa():
    with pytest.raises(ValueError, match="requires >=2"):
        _cfg(outcome_head="ordinal")
    with pytest.raises(ValueError, match="not 'ordinal'"):
        _cfg(outcome_head="simplex", ordinal_thresholds=THR)


def test_unknown_head_rejected():
    with pytest.raises(ValueError, match="unknown outcome_head"):
        _cfg(outcome_head="nonsense")


def test_existing_heads_unaffected():
    """Backward compatibility: every prior checkpoint was saved without these keys."""
    assert PERankFormerConfig().outcome_head == "scalar"
    assert PERankFormerConfig().ordinal_thresholds == ()
    simplex = PERankFormer(_cfg(outcome_head="simplex"))
    assert simplex(_batch()).shape == (6, 3)
