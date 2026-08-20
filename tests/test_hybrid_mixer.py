"""Hybrid SSM/attention sequence mixer (round-5 spec §12)."""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.models.ssm import HybridStack


@pytest.mark.parametrize("mode", ["alternating", "parallel"])
def test_padding_does_not_leak(mode):
    """Same property demanded of the pure SSM stack: padded content must not reach the
    outputs at real positions."""
    torch.manual_seed(0)
    m = HybridStack(16, 2, 32, 0.0, n_layers=2, n_state=16, mode=mode).eval()
    x = torch.randn(2, 20, 16)
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[0, 12:] = True
    with torch.no_grad():
        out1 = m(x, pad)
        x2 = x.clone()
        x2[0, 12:] = torch.randn(8, 16) * 10
        out2 = m(x2, pad)
    assert torch.allclose(out1[0, :12], out2[0, :12], atol=1e-5)


def test_parallel_gate_starts_balanced():
    """Neither mechanism may be privileged at init, or the comparison is rigged."""
    m = HybridStack(16, 2, 32, 0.0, n_layers=3, n_state=16, mode="parallel")
    assert all(abs(g - 0.5) < 1e-6 for g in m.gate_summary())


def test_parallel_uses_both_branches():
    torch.manual_seed(0)
    m = HybridStack(16, 2, 32, 0.0, n_layers=1, n_state=16, mode="parallel").eval()
    x = torch.randn(1, 10, 16)
    with torch.no_grad():
        base = m(x)
        # Force the gate fully onto the SSM branch; output must change.
        m.gates[0].data.fill_(20.0)
        ssm_only = m(x)
    assert not torch.allclose(base, ssm_only, atol=1e-4)


def test_alternating_layer_composition():
    m = HybridStack(16, 2, 32, 0.0, n_layers=4, n_state=16, mode="alternating")
    kinds = [type(b).__name__ for b in m.blocks]
    assert kinds[0] == "BiS4DLayer" and kinds[1] == "TransformerEncoder"
    assert len(m.blocks) == 4


@pytest.mark.parametrize("mixer", ["hybrid_alt", "hybrid_par"])
def test_model_forward_and_backward(mixer):
    cfg = PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=1,
        context_fields=("cell_type",), context_vocab_sizes={"cell_type": 3}, context_embed_dim=8,
        sequence_mixer=mixer, ssm_state_dim=16, outcome_head="ordinal",
        ordinal_thresholds=(0.01, 0.05, 0.2),
    )
    model = PERankFormer(cfg)
    batch = {
        "edit_ids": torch.randint(1, 5, (4, cfg.edit_seq_len)),
        "peg_nuc_ids": torch.randint(1, 5, (4, cfg.peg_seq_len)),
        "peg_seg_ids": torch.randint(0, 3, (4, cfg.peg_seq_len)),
        "ctx_cell_type": torch.randint(0, 3, (4,)),
    }
    out = model(batch)
    assert out.shape == (4, 3)
    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.edit_encoder.parameters())


def test_hybrid_rejects_layerwise_context():
    with pytest.raises(ValueError, match="not implemented"):
        PERankFormerConfig(sequence_mixer="hybrid_par", context_strategy="layerwise")


def test_unknown_mixer_still_rejected():
    with pytest.raises(ValueError, match="unknown sequence_mixer"):
        PERankFormerConfig(sequence_mixer="hybrid_nonsense")
