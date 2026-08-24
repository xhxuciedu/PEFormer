"""Per-source output head (round-8 B1) -- batch-effect model for three assays."""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig

THR = (0.01, 0.05, 0.2)
FIELDS = ("cell_type", "source_study")
SIZES = {"cell_type": 3, "source_study": 4}


def _cfg(**kw):
    base = dict(d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2,
                n_cross_blocks=1, context_fields=FIELDS, context_vocab_sizes=SIZES,
                context_embed_dim=8, outcome_head="ordinal", ordinal_thresholds=THR)
    base.update(kw)
    return PERankFormerConfig(**base)


def _batch(n=8, src=None):
    c = _cfg()
    return {"edit_ids": torch.randint(1, 5, (n, c.edit_seq_len)),
            "peg_nuc_ids": torch.randint(1, 5, (n, c.peg_seq_len)),
            "peg_seg_ids": torch.randint(0, 3, (n, c.peg_seq_len)),
            "ctx_cell_type": torch.randint(0, 3, (n,)),
            "ctx_source_study": (torch.full((n,), src) if src is not None
                                 else torch.randint(0, 4, (n,)))}


def test_source_changes_the_prediction():
    """The mechanism: identical design, different assay, different readout."""
    torch.manual_seed(0)
    m = PERankFormer(_cfg(source_conditional_head=4)).eval()
    for h in m.source_heads:                     # make heads genuinely different
        torch.nn.init.normal_(h.weight, std=0.5)
        torch.nn.init.normal_(h.bias, std=0.5)
    b = _batch(8, src=0)
    with torch.no_grad():
        o0 = m(b)
        b1 = dict(b); b1["ctx_source_study"] = torch.full((8,), 1)
        o1 = m(b1)
    assert not torch.allclose(o0, o1), "source had no effect on the readout"


def test_tied_control_uses_only_head_zero():
    """The control isolates the READOUT, not source entirely.

    `source_study` is also a FiLM context field, so it reaches the trunk in the tied
    control exactly as it does in the baseline -- that is intended, and is what makes
    this a matched control rather than an ablation of source information. What must be
    inert is the per-source *readout*: with heads tied, perturbing any head other than
    head 0 must have no effect at all.
    """
    torch.manual_seed(0)
    m = PERankFormer(_cfg(source_conditional_head=4, tie_source_heads=True)).eval()
    b = _batch(8)
    b["ctx_source_study"] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    with torch.no_grad():
        before = m(b)
        for h in list(m.source_heads)[1:]:
            h.weight.add_(torch.randn_like(h.weight) * 5)
            h.bias.add_(torch.randn_like(h.bias) * 5)
        after = m(b)
    assert torch.allclose(before, after), "tied control is reading from a head other than 0"


def test_control_has_identical_parameter_count():
    live = PERankFormer(_cfg(source_conditional_head=4))
    ctrl = PERankFormer(_cfg(source_conditional_head=4, tie_source_heads=True))
    assert live.num_parameters() == ctrl.num_parameters()


def test_rows_are_routed_to_their_own_head():
    """A mixed batch must give each row the same answer it would get alone -- otherwise
    the masked scatter is mis-routing rows."""
    torch.manual_seed(0)
    m = PERankFormer(_cfg(source_conditional_head=4)).eval()
    for h in m.source_heads:
        torch.nn.init.normal_(h.weight, std=0.5)
    b = _batch(8)
    b["ctx_source_study"] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    with torch.no_grad():
        mixed = m(b)
        for i in range(8):
            single = {k: v[i:i + 1] for k, v in b.items()}
            assert torch.allclose(mixed[i], m(single)[0], atol=1e-5), f"row {i} mis-routed"


def test_backward_reaches_every_head():
    m = PERankFormer(_cfg(source_conditional_head=4))
    b = _batch(8)
    b["ctx_source_study"] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    m(b).sum().backward()
    for i, h in enumerate(m.source_heads):
        assert h.weight.grad is not None and h.weight.grad.abs().sum() > 0, f"head {i} got no gradient"


def test_only_the_final_projection_is_per_source():
    """The trunk and the head's hidden layers must stay shared, or the three sources
    drift into three separate models rather than one model with three readouts."""
    m = PERankFormer(_cfg(source_conditional_head=4))
    shared = [p for layer in list(m.head)[:-1] for p in layer.parameters()]
    assert sum(p.numel() for p in shared) > 0, "head hidden layers should exist"
    # Each per-source head is only the final projection: d_model -> n_out.
    for h in m.source_heads:
        assert h.out_features == len(THR)


def test_misconfiguration_rejected():
    with pytest.raises(ValueError, match="requires 'source_study'"):
        PERankFormerConfig(context_fields=("cell_type",), context_vocab_sizes={"cell_type": 3},
                           source_conditional_head=4)
    with pytest.raises(ValueError, match="meaningless without"):
        _cfg(tie_source_heads=True)


def test_default_is_unchanged():
    assert PERankFormerConfig().source_conditional_head == 0
    m = PERankFormer(_cfg())
    assert m.source_heads is None
    assert m(_batch()).shape == (8, len(THR))
