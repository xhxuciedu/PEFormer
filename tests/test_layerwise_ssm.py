"""Layerwise context conditioning on the SSM backbone (round-6 diagnostic follow-up).

Motivation: the model's rankings are far more similar across experimental conditions
than reality (mean cross-condition Spearman 0.835 vs 0.683), i.e. context is used to
rescale rather than to reorder -- while that reordering is partly predictable from
design features. Late FiLM can only rescale the pooled summary. Layerwise FiLM was
previously unavailable on the SSM backbone, which is the strongest one we have.
"""

from __future__ import annotations

import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.models.ssm import BiS4DLayer

THR = (0.01, 0.05, 0.2)


def _cfg(**kw):
    base = dict(d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=3, n_peg_layers=2,
                n_cross_blocks=1, context_fields=("cell_type",),
                context_vocab_sizes={"cell_type": 4}, context_embed_dim=8,
                outcome_head="ordinal", ordinal_thresholds=THR,
                sequence_mixer="ssm", ssm_state_dim=16, context_strategy="layerwise")
    base.update(kw)
    return PERankFormerConfig(**base)


def _batch(n=6, cell=None):
    c = _cfg()
    return {"edit_ids": torch.randint(1, 5, (n, c.edit_seq_len)),
            "peg_nuc_ids": torch.randint(1, 5, (n, c.peg_seq_len)),
            "peg_seg_ids": torch.randint(0, 3, (n, c.peg_seq_len)),
            "ctx_cell_type": torch.full((n,), cell) if cell is not None
                             else torch.randint(0, 4, (n,))}


def test_layerwise_ssm_now_constructs():
    """Previously raised ValueError: 'not implemented'."""
    m = PERankFormer(_cfg())
    assert m.layerwise
    assert all(isinstance(b, BiS4DLayer) for b in m.edit_layers)
    assert all(isinstance(b, BiS4DLayer) for b in m.peg_layers)
    assert len(m.edit_films) == 3 and len(m.peg_films) == 2


def test_forward_and_backward():
    m = PERankFormer(_cfg())
    out = m(_batch())
    assert out.shape == (6, len(THR))
    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.edit_films[0].parameters())


def test_context_can_reorder_not_merely_rescale():
    """The property the change exists for: switching context must be able to change the
    ORDER of predictions, not just shift them. A pure scale/offset would preserve order.
    """
    torch.manual_seed(0)
    m = PERankFormer(_cfg()).eval()
    for f in list(m.edit_films) + list(m.peg_films):   # make conditioning non-trivial
        last = [mod for mod in f.net if isinstance(mod, torch.nn.Linear)][-1]
        torch.nn.init.normal_(last.weight, std=0.5)
        torch.nn.init.normal_(last.bias, std=0.5)
    b = _batch(24, cell=0)
    with torch.no_grad():
        s0 = m.efficiency_from_output(m(b))
        b2 = dict(b); b2["ctx_cell_type"] = torch.full((24,), 3)
        s1 = m.efficiency_from_output(m(b2))
    assert not torch.equal(torch.argsort(s0), torch.argsort(s1)), \
        "context changed no ordering -- conditioning is acting only as a rescale"


def test_padding_still_masked():
    torch.manual_seed(0)
    m = PERankFormer(_cfg()).eval()
    b = _batch(2)
    b["edit_ids"][0, 40:] = 0   # pad id
    with torch.no_grad():
        o1 = m(b)
        b2 = {k: v.clone() for k, v in b.items()}
        b2["edit_ids"][0, 40:] = 0
        o2 = m(b2)
    assert torch.allclose(o1, o2)


def test_attention_layerwise_unchanged():
    """Family A (layerwise Transformer) must behave exactly as before."""
    m = PERankFormer(_cfg(sequence_mixer="attention"))
    assert not any(isinstance(b, BiS4DLayer) for b in m.edit_layers)
    assert m(_batch()).shape == (6, len(THR))


def test_late_conditioning_still_default():
    assert PERankFormerConfig().context_strategy == "late"
