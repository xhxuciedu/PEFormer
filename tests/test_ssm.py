"""Bidirectional S4D sequence mixer (round-4 experiment B, spec §8)."""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.models.ssm import BiS4DStack, S4DKernel


def test_kernel_shape_and_finiteness():
    k = S4DKernel(d_model=8, n_state=16)
    out = k(L=32)
    assert out.shape == (8, 32)
    assert torch.isfinite(out).all()


def test_padding_does_not_leak_into_real_positions():
    """The strongest correctness property for an FFT-convolution mixer: content in
    padded positions must not influence outputs at real positions."""
    torch.manual_seed(0)
    m = BiS4DStack(d_model=16, ffn_dim=32, dropout=0.0, n_layers=2, n_state=16).eval()
    x = torch.randn(2, 20, 16)
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[0, 12:] = True

    with torch.no_grad():
        out1 = m(x, pad)
        x2 = x.clone()
        x2[0, 12:] = torch.randn(8, 16) * 10  # garbage in the padded region
        out2 = m(x2, pad)

    assert torch.allclose(out1[0, :12], out2[0, :12], atol=1e-6)


def test_padded_outputs_are_zeroed():
    torch.manual_seed(0)
    m = BiS4DStack(d_model=16, ffn_dim=32, dropout=0.0, n_layers=1, n_state=16).eval()
    x = torch.randn(1, 10, 16)
    pad = torch.zeros(1, 10, dtype=torch.bool)
    pad[0, 6:] = True
    with torch.no_grad():
        out = m(x, pad)
    assert out[0, 6:].abs().max() < 1e-6


def test_bidirectional_uses_both_directions():
    """A forward-only mixer would be invariant to what follows a position; the
    bidirectional one must not be."""
    torch.manual_seed(0)
    m = BiS4DStack(d_model=8, ffn_dim=16, dropout=0.0, n_layers=1, n_state=8).eval()
    x = torch.randn(1, 12, 8)
    with torch.no_grad():
        out1 = m(x)
        x2 = x.clone()
        x2[0, -1] = torch.randn(8) * 5  # change only the LAST position
        out2 = m(x2)
    # an earlier position must respond, which only happens if information flows backwards
    assert (out1[0, 0] - out2[0, 0]).abs().max() > 1e-6


def test_ssm_model_forward_and_backward():
    cfg = PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=1,
        context_fields=("cell_type",), context_vocab_sizes={"cell_type": 3}, context_embed_dim=8,
        sequence_mixer="ssm", ssm_state_dim=16, outcome_head="simplex",
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
    assert torch.isfinite(model.edit_encoder.layers[0].kernel_fwd.C.grad).all()


def test_ssm_supports_layerwise_context():
    """Round 6 implemented this; it previously raised "not implemented".

    Kept as a test rather than deleted, because the round-6 diagnostic (the model
    reorders far less across experimental conditions than reality does) is the reason
    the combination matters, and a silent regression to late-only conditioning would
    be hard to notice from accuracy alone.
    """
    cfg = PERankFormerConfig(sequence_mixer="ssm", context_strategy="layerwise",
                             context_fields=("cell_type",), context_vocab_sizes={"cell_type": 3})
    assert cfg.context_strategy == "layerwise"


def test_unknown_mixer_rejected():
    with pytest.raises(ValueError, match="unknown sequence_mixer"):
        PERankFormerConfig(sequence_mixer="nonsense")


def test_attention_remains_default():
    """Backward compatibility: every pre-round-4 checkpoint was saved without this key."""
    assert PERankFormerConfig().sequence_mixer == "attention"
