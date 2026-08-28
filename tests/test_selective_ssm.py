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


def test_chunked_scan_matches_the_naive_recurrence():
    """The chunked closed form must reproduce the step-by-step recurrence exactly.

    This is the test that matters for the optimisation: the speedup is only legitimate
    if the maths is unchanged, and a subtle indexing error here would silently train a
    different model.
    """
    torch.manual_seed(0)
    m = SelectiveScan(8, n_state=8, chunk=8).double().eval()
    x = torch.randn(2, 37, 8, dtype=torch.double)   # length not a multiple of the chunk

    with torch.no_grad():
        fast = m(x)

        # Reference: literal per-step loop.
        B, L, H = x.shape
        N = m.n_state
        A = -torch.exp(m.log_neg_A)
        proj = m.x_proj(x)
        dt, Bm, Cm = torch.split(proj, [m.dt_rank, N, N], dim=-1)
        dt = torch.nn.functional.softplus(m.dt_proj(dt))
        h = x.new_zeros(B, H, N)
        ys = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)
            h = torch.exp(dt_t * A) * h + dt_t * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        ref = torch.stack(ys, dim=1) + x * m.D

    assert torch.allclose(fast, ref, atol=1e-8), \
        f"chunked scan diverges from the recurrence (max {(fast - ref).abs().max():.2e})"


def test_chunk_size_does_not_change_the_result():
    torch.manual_seed(0)
    x = torch.randn(1, 30, 8, dtype=torch.double)
    outs = []
    for c in (4, 8, 16):
        torch.manual_seed(0)
        m = SelectiveScan(8, n_state=8, chunk=c).double().eval()
        with torch.no_grad():
            outs.append(m(x))
    assert torch.allclose(outs[0], outs[1], atol=1e-8)
    assert torch.allclose(outs[1], outs[2], atol=1e-8)


def test_chunked_scan_stable_under_bf16_autocast():
    """Regression test for a real failure: the chunked scan trained to NaN under bf16.

    The float64 equivalence test passed, so correctness of the maths was never the
    issue -- the chunk formula divides by a small cumulative product, and bf16's ~3
    decimal digits cannot survive that. The scan therefore forces float32 internally.
    This test exercises the precision training actually uses.
    """
    if not torch.cuda.is_available():
        import pytest as _pytest
        _pytest.skip("needs CUDA for bf16 autocast")
    torch.manual_seed(0)
    m = SelectiveScan(64, n_state=16).cuda()
    x = torch.randn(8, 102, 64, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = m(x)
        loss = y.square().mean()
    loss.backward()
    assert torch.isfinite(y).all(), "forward produced non-finite values under bf16"
    for n_, p_ in m.named_parameters():
        if p_.grad is not None:
            assert torch.isfinite(p_.grad).all(), f"non-finite gradient in {n_}"


def test_scan_survives_large_dt_where_the_product_underflows():
    """The regime that actually broke training.

    With a learned dt large enough that the within-chunk cumulative product underflows,
    the earlier division-based form produced inf and then NaN. The bf16 test above did
    not catch it because random initialisation keeps dt moderate -- training does not.
    """
    torch.manual_seed(0)
    m = SelectiveScan(16, n_state=8)
    with torch.no_grad():
        m.dt_proj.bias.fill_(6.0)       # softplus(6) ~ 6, so dt*A reaches ~ -100
    x = torch.randn(4, 102, 16)
    y = m(x)
    assert torch.isfinite(y).all(), "scan produced non-finite values at large dt"
    y.square().mean().backward()
    for n_, p_ in m.named_parameters():
        if p_.grad is not None:
            assert torch.isfinite(p_.grad).all(), f"non-finite gradient in {n_} at large dt"
