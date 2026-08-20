"""Conditional quantile head with pinball loss (round-5 spec §13)."""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.training.losses import pinball_loss

Q = (0.1, 0.25, 0.5, 0.75, 0.9)


def _cfg(**kw):
    base = dict(d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2,
                n_cross_blocks=1, context_fields=("cell_type",),
                context_vocab_sizes={"cell_type": 3}, context_embed_dim=8,
                outcome_head="quantile", quantile_levels=Q)
    base.update(kw)
    return PERankFormerConfig(**base)


def test_pinball_minimised_at_the_true_quantile():
    """The defining property: the loss for level q must be minimised by the q-th
    quantile of the target, not by its mean."""
    torch.manual_seed(0)
    y = torch.rand(20000) ** 3  # heavily right-skewed, like efficiency
    q = torch.tensor([0.1, 0.5, 0.9])
    truth = torch.quantile(y, q)
    for i, level in enumerate(q):
        cand = torch.stack([truth.clone() for _ in range(len(y))])
        best = pinball_loss(cand, y, q)
        for delta in (-0.05, 0.05):
            worse = cand.clone()
            worse[:, i] += delta
            assert pinball_loss(worse, y, q) > best


def test_asymmetry_direction():
    """Under-predicting a high quantile must cost more than over-predicting it."""
    y = torch.tensor([1.0])
    q = torch.tensor([0.9])
    under = pinball_loss(torch.tensor([[0.5]]), y, q)
    over = pinball_loss(torch.tensor([[1.5]]), y, q)
    assert under > over


def test_head_emits_one_output_per_level_and_ranks_by_median():
    m = PERankFormer(_cfg())
    batch = {
        "edit_ids": torch.randint(1, 5, (4, m.config.edit_seq_len)),
        "peg_nuc_ids": torch.randint(1, 5, (4, m.config.peg_seq_len)),
        "peg_seg_ids": torch.randint(0, 3, (4, m.config.peg_seq_len)),
        "ctx_cell_type": torch.randint(0, 3, (4,)),
    }
    out = m(batch)
    assert out.shape == (4, len(Q))
    # median is index 2 of 5; the score must be exactly that column, untransformed,
    # since this head alone predicts in efficiency units.
    assert torch.allclose(m.efficiency_from_output(out), out[:, 2])
    assert torch.allclose(m.ranking_score(out), out[:, 2])


def test_levels_validated():
    with pytest.raises(ValueError, match="strictly increasing"):
        _cfg(quantile_levels=(0.5, 0.2))
    with pytest.raises(ValueError, match="strictly inside"):
        _cfg(quantile_levels=(0.0, 0.5, 1.0))
    with pytest.raises(ValueError, match="not 'quantile'"):
        PERankFormerConfig(outcome_head="simplex", quantile_levels=Q)


def test_other_heads_unchanged():
    assert PERankFormerConfig().quantile_levels == ()
    assert PERankFormerConfig().outcome_head == "scalar"
