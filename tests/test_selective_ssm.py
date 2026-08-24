"""Selective (Mamba-style) SSM mixer -- round-8 experiment A1.

The claim being tested is that this mixer is *content-dependent*, which is exactly
what S4D is not: S4DKernel.forward takes only the sequence length, so its convolution
is identical for every input. Several tests below exist specifically to prove the
difference is real rather than nominal.
"""

from __future__ import annotations

import pytest
import torch

from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
from pe_rankformer.models.ssm import BiSelectiveStack, S4DKernel, SelectiveScan


def test_scan_is_content_dependent():
    """The defining property. Changing the CONTENT of a position must change how other
    positions are mixed -- not merely shift the output by that content's own value."""
    torch.manual_seed(0)
    m = SelectiveScan(8, n_state=8).eval()
    x = torch.randn(1, 12, 8)
    with torch.no_grad():
        base = m(x)
        x2 = x.clone()
        x2[0, 3] += 5.0            # change content at t=3
        alt = m(x2)
    # Positions AFTER t=3 must respond; an LTI system would respond too, so the
    # sharper check is that the response is not a fixed linear function of the change:
    # doubling the perturbation must not exactly double the downstream delta.
    with torch.no_grad():
        x3 = x.clone(); x3[0, 3] += 10.0
        alt2 = m(x3)
    d1 = (alt - base)[0, 4:]
    d2 = (alt2 - base)[0, 4:]
    assert d1.abs().max() > 1e-6, "downstream positions did not respond at all"
    assert not torch.allclose(d2, 2 * d1, atol=1e-4), \
        "response is exactly linear in the perturbation -- the scan is not selective"


def test_s4d_is_not_content_dependent():
    """Contrast case, documenting the limitation rather than assuming it."""
    k = S4DKernel(d_model=8, n_state=16)
    assert torch.allclose(k(L=20), k(L=20)), "kernel should be deterministic"
    # The kernel signature takes only L -- there is no input argument to depend on.
    import inspect
    assert list(inspect.signature(k.forward).parameters) == ["L"]


def test_frozen_control_has_same_params_but_no_selection_grad():
    """The round-8 control: identical capacity, selection projections frozen."""
    live = SelectiveScan(16, n_state=8, freeze_selection=False)
    ctrl = SelectiveScan(16, n_state=8, freeze_selection=True)
    assert sum(p.numel() for p in live.parameters()) == sum(p.numel() for p in ctrl.parameters())
    assert ctrl.x_proj.weight.requires_grad is False
    assert ctrl.dt_proj.weight.requires_grad is False
    assert live.x_proj.weight.requires_grad is True


def test_padding_does_not_leak_into_real_positions():
    torch.manual_seed(0)
    m = BiSelectiveStack(16, 32, 0.0, n_layers=2, n_state=8).eval()
    x = torch.randn(2, 20, 16)
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[0, 12:] = True
    with torch.no_grad():
        o1 = m(x, pad)
        x2 = x.clone(); x2[0, 12:] = torch.randn(8, 16) * 10
        o2 = m(x2, pad)
    assert torch.allclose(o1[0, :12], o2[0, :12], atol=1e-5)


def test_padded_outputs_zeroed():
    m = BiSelectiveStack(16, 32, 0.0, n_layers=1, n_state=8).eval()
    x = torch.randn(1, 10, 16)
    pad = torch.zeros(1, 10, dtype=torch.bool); pad[0, 6:] = True
    with torch.no_grad():
        out = m(x, pad)
    assert out[0, 6:].abs().max() < 1e-6


def test_bidirectional():
    torch.manual_seed(0)
    m = BiSelectiveStack(8, 16, 0.0, n_layers=1, n_state=8).eval()
    x = torch.randn(1, 12, 8)
    with torch.no_grad():
        o1 = m(x)
        x2 = x.clone(); x2[0, -1] = torch.randn(8) * 5
        o2 = m(x2)
    assert (o1[0, 0] - o2[0, 0]).abs().max() > 1e-6, "no backward information flow"


def test_recurrence_is_stable():
    """A must stay negative or the scan diverges over long sequences."""
    m = SelectiveScan(8, n_state=8)
    assert (-torch.exp(m.log_neg_A) < 0).all()
    x = torch.randn(1, 102, 8)          # full edit-stream length
    assert torch.isfinite(m(x)).all()


@pytest.mark.parametrize("mixer", ["selective", "selective_frozen"])
def test_model_forward_backward(mixer):
    cfg = PERankFormerConfig(
        d_model=32, n_heads=2, ffn_dim=64, n_edit_layers=2, n_peg_layers=2, n_cross_blocks=1,
        context_fields=("cell_type",), context_vocab_sizes={"cell_type": 3}, context_embed_dim=8,
        sequence_mixer=mixer, ssm_state_dim=8, outcome_head="ordinal",
        ordinal_thresholds=(0.01, 0.05, 0.2),
    )
    model = PERankFormer(cfg)
    batch = {
        "edit_ids": torch.randint(1, 5, (2, cfg.edit_seq_len)),
        "peg_nuc_ids": torch.randint(1, 5, (2, cfg.peg_seq_len)),
        "peg_seg_ids": torch.randint(0, 3, (2, cfg.peg_seq_len)),
        "ctx_cell_type": torch.randint(0, 3, (2,)),
    }
    out = model(batch)
    assert out.shape == (2, 3)
    out.sum().backward()


def test_prior_mixers_unchanged():
    assert PERankFormerConfig().sequence_mixer == "attention"
    with pytest.raises(ValueError, match="unknown sequence_mixer"):
        PERankFormerConfig(sequence_mixer="selective_nonsense")
