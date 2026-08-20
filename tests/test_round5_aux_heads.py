"""Round-5 auxiliary heads: dual-head (§6), multi-resolution (§7), context-ordinal (§8).

The property that makes these *multitask* experiments rather than disguised ensembling
is that auxiliary heads shape the shared trunk but never touch the prediction. That is
the main thing tested here, because a leak would silently turn a negative result into a
positive one.
"""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.training.losses import LossWeights, total_loss

THR = (0.01, 0.05, 0.1, 0.3)
AUX7 = (0.02, 0.2)


def _cfg(**kw) -> PERankFormerConfig:
    base = dict(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=1,
        context_fields=("cell_type",), context_vocab_sizes={"cell_type": 3}, context_embed_dim=8,
        outcome_head="ordinal", ordinal_thresholds=THR,
    )
    base.update(kw)
    return PERankFormerConfig(**base)


def _batch(n=4):
    c = _cfg()
    return {
        "edit_ids": torch.randint(1, 5, (n, c.edit_seq_len)),
        "peg_nuc_ids": torch.randint(1, 5, (n, c.peg_seq_len)),
        "peg_seg_ids": torch.randint(0, 3, (n, c.peg_seq_len)),
        "ctx_cell_type": torch.randint(0, 3, (n,)),
    }


def test_dual_head_widens_output_but_not_score():
    """The decisive property: adding an auxiliary head must not change what the model
    predicts, only what it learns from."""
    plain, dual = PERankFormer(_cfg()), PERankFormer(_cfg(aux_simplex_weight=0.25))
    assert plain(_batch()).shape == (4, 4)
    assert dual(_batch()).shape == (4, 4 + 3)

    out = torch.randn(6, 4 + 3)
    # Score depends only on the first len(THR) logits; scrambling the rest changes nothing.
    scrambled = out.clone()
    scrambled[:, len(THR):] = torch.randn(6, 3) * 50
    assert torch.allclose(dual.efficiency_from_output(out), dual.efficiency_from_output(scrambled))
    assert torch.allclose(dual.ranking_score(out), dual.ranking_score(scrambled))


def test_multires_appends_one_segment_per_resolution():
    m = PERankFormer(_cfg(ordinal_thresholds_aux=(AUX7, (0.01, 0.02, 0.4)), aux_ordinal_weight=0.3))
    assert m(_batch()).shape == (4, 4 + 2 + 3)
    kinds = [s["kind"] for s in m.head_segments]
    assert kinds == ["ordinal", "ordinal", "ordinal"]
    assert [s["end"] - s["start"] for s in m.head_segments] == [4, 2, 3]


def test_context_ordinal_segment_matches_primary_width():
    m = PERankFormer(_cfg(aux_context_ordinal=True, aux_context_weight=0.25))
    assert m(_batch()).shape == (4, 8)
    assert m.head_segments[-1]["kind"] == "ordinal_ctx"


def test_segments_are_contiguous_and_cover_the_output():
    m = PERankFormer(_cfg(aux_simplex_weight=0.1, ordinal_thresholds_aux=(AUX7,),
                          aux_ordinal_weight=0.2, aux_context_ordinal=True, aux_context_weight=0.1))
    segs = m.head_segments
    assert segs[0]["start"] == 0
    for a, b in zip(segs, segs[1:]):
        assert a["end"] == b["start"], "segments must tile the output without gaps or overlap"
    assert segs[-1]["end"] == m(_batch()).shape[-1]


def test_aux_losses_contribute_and_backprop():
    m = PERankFormer(_cfg(aux_simplex_weight=0.5, aux_context_ordinal=True, aux_context_weight=0.5))
    out = m(_batch())
    tgt = torch.tensor([0.0, 0.05, 0.2, 0.6])
    w = LossWeights(outcome_head="ordinal", ordinal_thresholds=torch.tensor(THR),
                    head_segments=m.head_segments)
    loss, parts = total_loss(out, tgt, torch.tensor([0, 1]), torch.tensor([2, 3]), w,
                             rank_score=m.ranking_score(out),
                             target_indel=torch.tensor([0.01, 0.02, 0.03, 0.04]),
                             target_ctx_q=torch.tensor([0.1, 0.4, 0.6, 0.9]))
    assert "aux_simplex" in parts and "aux_ordinal_ctx" in parts
    loss.backward()
    assert torch.isfinite(m.head[-1].weight.grad).all()


def test_aux_loss_actually_increases_total():
    """A weight of zero and a positive weight must differ, or the experiment is a no-op."""
    m = PERankFormer(_cfg(aux_simplex_weight=0.5))
    out = m(_batch())
    tgt = torch.tensor([0.0, 0.05, 0.2, 0.6])
    indel = torch.tensor([0.01, 0.02, 0.03, 0.04])
    args = (tgt, torch.tensor([0, 1]), torch.tensor([2, 3]))
    kw = dict(rank_score=m.ranking_score(out), target_indel=indel)
    with_aux, _ = total_loss(out, *args, LossWeights(
        outcome_head="ordinal", ordinal_thresholds=torch.tensor(THR),
        head_segments=m.head_segments), **kw)
    without, _ = total_loss(out[..., :4], *args, LossWeights(
        outcome_head="ordinal", ordinal_thresholds=torch.tensor(THR)), **kw)
    assert with_aux.item() > without.item()


def test_context_ordinal_thresholds_are_the_uniform_grid():
    """Context-normalised targets are quantiles in [0,1], so global-distribution
    thresholds would be wrong; the grid must be uniform and interior."""
    m = PERankFormer(_cfg(aux_context_ordinal=True, aux_context_weight=1.0))
    seg = m.head_segments[-1]
    k = seg["end"] - seg["start"]
    grid = torch.linspace(0.0, 1.0, k + 2)[1:-1]
    assert grid.min() > 0 and grid.max() < 1 and len(grid) == k


def test_misconfiguration_is_rejected():
    with pytest.raises(ValueError, match="require outcome_head='ordinal'"):
        PERankFormerConfig(outcome_head="simplex", aux_simplex_weight=0.2)
    with pytest.raises(ValueError, match="aux_ordinal_weight is 0"):
        _cfg(ordinal_thresholds_aux=(AUX7,))
    with pytest.raises(ValueError, match="aux_context_weight is 0"):
        _cfg(aux_context_ordinal=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        _cfg(ordinal_thresholds_aux=((0.5, 0.1),), aux_ordinal_weight=0.1)


def test_defaults_leave_every_prior_model_unchanged():
    c = PERankFormerConfig()
    assert c.aux_simplex_weight == 0.0 and c.ordinal_thresholds_aux == ()
    assert c.aux_context_ordinal is False
    plain = PERankFormer(_cfg())
    assert len(plain.head_segments) == 1
    assert plain(_batch()).shape == (4, len(THR))
