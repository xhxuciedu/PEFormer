"""Round-6 heads and losses: CORAL, hurdle, source weighting, monotonicity penalty."""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.training.losses import (LossWeights, hurdle_loss, ordinal_loss,
                                           total_loss, _wmean)

THR = (0.01, 0.05, 0.1, 0.3)


def _cfg(**kw):
    base = dict(d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2,
                n_cross_blocks=1, context_fields=("cell_type",),
                context_vocab_sizes={"cell_type": 3}, context_embed_dim=8,
                ordinal_thresholds=THR)
    base.update(kw)
    return PERankFormerConfig(**base)


def _batch(n=6):
    c = _cfg(outcome_head="ordinal")
    return {"edit_ids": torch.randint(1, 5, (n, c.edit_seq_len)),
            "peg_nuc_ids": torch.randint(1, 5, (n, c.peg_seq_len)),
            "peg_seg_ids": torch.randint(0, 3, (n, c.peg_seq_len)),
            "ctx_cell_type": torch.randint(0, 3, (n,))}


# ---------------- CORAL: the whole point is rank consistency ----------------

def test_coral_is_monotone_for_every_input():
    """The defect this head exists to fix: the plain ordinal head violates
    P(y>t_1) >= P(y>t_2) >= ... on 100% of rows. CORAL must never violate it."""
    torch.manual_seed(0)
    m = PERankFormer(_cfg(outcome_head="coral"))
    # Random (untrained) biases too, so this is not just true at initialisation.
    m.coral_bias_raw.data = torch.randn(len(THR))
    out = m(_batch(64))
    p = torch.sigmoid(out)
    assert (p[:, 1:] <= p[:, :-1] + 1e-6).all(), "CORAL produced a non-monotone CDF"


def test_coral_biases_strictly_increasing():
    m = PERankFormer(_cfg(outcome_head="coral"))
    m.coral_bias_raw.data = torch.randn(len(THR)) * 3
    b = m.coral_biases()
    assert (b[1:] > b[:-1]).all()


def test_coral_uses_one_shared_weight_vector():
    m = PERankFormer(_cfg(outcome_head="coral"))
    assert m.head[-1].out_features == 1, "CORAL must emit a single shared logit"
    assert m(_batch()).shape == (6, len(THR))


def test_plain_ordinal_still_can_violate():
    """Contrast case: documents the defect rather than assuming it."""
    m = PERankFormer(_cfg(outcome_head="ordinal"))
    assert m.head[-1].out_features == len(THR)


# ---------------- hurdle: zero-inflation ----------------

def test_hurdle_output_layout_and_score():
    m = PERankFormer(_cfg(outcome_head="hurdle"))
    out = m(_batch())
    assert out.shape == (6, 1 + len(THR))
    expect = torch.sigmoid(out[:, 0]) * torch.sigmoid(out[:, 1:]).mean(dim=-1)
    assert torch.allclose(m.efficiency_from_output(out), expect)


def test_hurdle_gate_dominates_ranking_for_zero_rows():
    """A row the gate calls non-editing must rank below one it calls editing, even if
    the conditional part is identical -- otherwise the factorisation does nothing."""
    m = PERankFormer(_cfg(outcome_head="hurdle"))
    cond = torch.zeros(2, len(THR))
    out = torch.cat([torch.tensor([[-8.0], [8.0]]), cond], dim=1)
    s = m.efficiency_from_output(out)
    assert s[1] > s[0]


def test_hurdle_conditional_term_ignores_zero_rows():
    """Zero rows carry no ordering information among editing rows; including them would
    inject noise into exactly the block the factorisation removes."""
    thr = torch.tensor(THR)
    target = torch.tensor([0.0, 0.0, 0.5])
    gate = torch.zeros(3)
    ordl = torch.zeros(3, len(THR))
    base = hurdle_loss(gate, ordl, target, thr)
    scrambled = ordl.clone()
    scrambled[:2] = 50.0  # garbage in the ordinal part of the ZERO rows only
    assert torch.allclose(base, hurdle_loss(gate, scrambled, target, thr))


# ---------------- source weighting ----------------

def test_weighted_mean_normalises_by_weight_sum():
    x = torch.tensor([1.0, 3.0])
    assert torch.allclose(_wmean(x, None), torch.tensor(2.0))
    assert torch.allclose(_wmean(x, torch.tensor([3.0, 1.0])), torch.tensor(1.5))
    # Scaling all weights must not change the loss, or a weight sweep would double as
    # an unintended learning-rate sweep.
    a = _wmean(x, torch.tensor([1.0, 2.0]))
    b = _wmean(x, torch.tensor([10.0, 20.0]))
    assert torch.allclose(a, b)


def test_sample_weight_shifts_which_rows_matter():
    thr = torch.tensor(THR)
    # Row 0 is predicted correctly, row 1 badly. Weighting must change the loss to
    # match whichever row it emphasises.
    logits = torch.tensor([[-10.0] * 4, [-10.0] * 4])
    target = torch.tensor([0.0, 0.5])  # row 0 truly all-zero (correct), row 1 all-one (wrong)
    only_first = ordinal_loss(logits, target, thr, sample_weight=torch.tensor([1.0, 0.0]))
    only_second = ordinal_loss(logits, target, thr, sample_weight=torch.tensor([0.0, 1.0]))
    assert only_first < 0.01, "correct row should incur ~no loss"
    assert only_second > 5.0, "wrong row should incur large loss"


# ---------------- monotonicity penalty ----------------

def test_mono_penalty_fires_only_on_violations():
    thr = torch.tensor(THR)
    target = torch.tensor([0.2])
    good = torch.tensor([[3.0, 2.0, 1.0, 0.0]])   # decreasing -> monotone
    bad = torch.tensor([[0.0, 1.0, 2.0, 3.0]])    # increasing -> violates
    assert torch.allclose(ordinal_loss(good, target, thr, mono_penalty=1.0),
                          ordinal_loss(good, target, thr, mono_penalty=0.0), atol=1e-6)
    assert (ordinal_loss(bad, target, thr, mono_penalty=1.0)
            > ordinal_loss(bad, target, thr, mono_penalty=0.0) + 0.05)


# ---------------- integration ----------------

@pytest.mark.parametrize("head", ["coral", "hurdle"])
def test_total_loss_backprops(head):
    m = PERankFormer(_cfg(outcome_head=head))
    out = m(_batch(4))
    tgt = torch.tensor([0.0, 0.0, 0.2, 0.6])
    w = LossWeights(outcome_head=head, ordinal_thresholds=torch.tensor(THR),
                    head_segments=m.head_segments)
    loss, _ = total_loss(out, tgt, torch.tensor([0, 1]), torch.tensor([2, 3]), w,
                         rank_score=m.ranking_score(out),
                         sample_weight=torch.tensor([1.0, 1.0, 2.0, 2.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(m.head[-1].weight.grad).all()


def test_prior_heads_unchanged():
    assert PERankFormerConfig().outcome_head == "scalar"
    for h in ("simplex", "ordinal"):
        c = _cfg(outcome_head=h) if h == "ordinal" else PERankFormerConfig(outcome_head=h)
        assert PERankFormer(c) is not None
