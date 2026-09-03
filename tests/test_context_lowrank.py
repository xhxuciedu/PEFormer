"""Round-9 C1: context-conditioned non-diagonal correction (ContextLowRank).

The mechanism under test is that context can mix channels *differently* per condition,
which is what FiLM structurally cannot do (round-7 diagnosis). These tests pin the two
properties the experiment relies on: the block is the identity at initialisation, so the
architecture is a strict superset of the late-FiLM baseline; and the control keeps every
parameter while being provably unable to depend on context.
"""

import pytest
import torch

from pe_rankformer.models.pe_rankformer import (
    ContextLowRank,
    FiLM,
    PERankFormer,
    PERankFormerConfig,
)

CTX_DIM, FEAT_DIM, RANK, B = 24, 32, 4, 16


def _hc():
    torch.manual_seed(0)
    return torch.randn(B, FEAT_DIM), torch.randn(B, CTX_DIM)


def test_identity_at_initialisation():
    """U is zero-initialised, so the block starts as an exact no-op."""
    m = ContextLowRank(CTX_DIM, FEAT_DIM, RANK).requires_grad_(False)
    h, c = _hc()
    assert torch.equal(m(h, c), h)


def test_depends_on_context_once_trained():
    m = ContextLowRank(CTX_DIM, FEAT_DIM, RANK).requires_grad_(False)
    torch.nn.init.normal_(m.U.weight, std=0.5)
    h, c = _hc()
    c2 = torch.randn(B, CTX_DIM)
    assert not torch.allclose(m(h, c), m(h, c2), atol=1e-6)


def test_control_cannot_depend_on_context():
    """Same parameters and FLOPs, but the gain network is fed zeros."""
    m = ContextLowRank(CTX_DIM, FEAT_DIM, RANK, frozen=True)
    torch.nn.init.normal_(m.U.weight, std=0.5)
    h, c = _hc()
    for _ in range(5):
        assert torch.equal(m(h, c), m(h, torch.randn(B, CTX_DIM)))


def test_control_has_identical_parameter_count():
    a = ContextLowRank(CTX_DIM, FEAT_DIM, RANK, frozen=False)
    b = ContextLowRank(CTX_DIM, FEAT_DIM, RANK, frozen=True)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_is_not_diagonal_unlike_film():
    """The point of the mechanism: it mixes channels, FiLM cannot.

    A diagonal map cannot change the ORDER of two rows' projections onto a fixed
    direction in a way that depends on context; a non-diagonal one can. Here we assert
    the weaker, sufficient property that the correction is not expressible as a
    per-channel rescale, i.e. h'/h is not constant across channels.
    """
    m = ContextLowRank(CTX_DIM, FEAT_DIM, RANK).requires_grad_(False)
    torch.nn.init.normal_(m.U.weight, std=0.5)
    h, c = _hc()
    delta = m(h, c) - h
    # a per-channel scale would make delta[i] proportional to h[i] channel-wise
    ratio = delta / h
    assert ratio.std(dim=-1).min() > 1e-3


def test_effective_map_is_non_diagonal_where_films_is_diagonal():
    """The precise structural difference the mechanism is for.

    FiLM conditions through a *diagonal* map: h -> diag(1 + gamma(c)) h + beta(c). Its
    conditioning can therefore only reweight the existing coordinates, never form new
    directions in representation space. ContextLowRank conditions through
    I + U diag(a(c)) V^T, which is non-diagonal, so the readout direction it presents to
    the head can differ between conditions by more than a per-channel rescale.
    """
    m = ContextLowRank(CTX_DIM, FEAT_DIM, RANK).requires_grad_(False)
    torch.nn.init.normal_(m.U.weight, std=0.5)
    c = torch.randn(1, CTX_DIM)
    basis = torch.eye(FEAT_DIM)
    effective = m(basis, c.expand(FEAT_DIM, -1))  # rows = images of basis vectors
    off_diagonal = effective - torch.diag(torch.diagonal(effective))
    assert off_diagonal.abs().max() > 1e-3, "conditioning is diagonal, i.e. FiLM-equivalent"


def test_different_contexts_induce_genuinely_different_maps():
    """Not merely a scaled version of one map, which is all a global gain would give."""
    m = ContextLowRank(CTX_DIM, FEAT_DIM, RANK).requires_grad_(False)
    torch.nn.init.normal_(m.U.weight, std=0.5)
    basis = torch.eye(FEAT_DIM)
    torch.manual_seed(1)
    m1 = m(basis, torch.randn(1, CTX_DIM).expand(FEAT_DIM, -1)) - basis
    m2 = m(basis, torch.randn(1, CTX_DIM).expand(FEAT_DIM, -1)) - basis
    # best scalar alpha matching m2 to m1; a residual means the maps differ in direction
    alpha = (m1 * m2).sum() / (m2 * m2).sum().clamp_min(1e-12)
    residual = (m1 - alpha * m2).norm() / m1.norm().clamp_min(1e-12)
    assert float(residual) > 1e-2


def test_config_rejects_lowrank_without_context():
    with pytest.raises(ValueError, match="requires use_context"):
        PERankFormerConfig(context_lowrank=8, use_context=False)


def test_config_rejects_control_without_rank():
    with pytest.raises(ValueError, match="control set but"):
        PERankFormerConfig(context_lowrank=0, context_lowrank_control=True)


def test_config_rejects_negative_rank():
    with pytest.raises(ValueError, match="must be >= 0"):
        PERankFormerConfig(context_lowrank=-1)


def _model(**kw):
    return PERankFormer(PERankFormerConfig(
        context_fields=("cell_type",), context_vocab_sizes={"cell_type": 5},
        d_model=32, n_heads=4, ffn_dim=64, n_edit_layers=1, n_peg_layers=1,
        n_cross_blocks=1, outcome_head="ordinal",
        ordinal_thresholds=(0.1, 0.5, 0.9), sequence_mixer="ssm", ssm_state_dim=8, **kw))


def test_end_to_end_model_is_unchanged_at_init():
    """A full model with the block added scores identically to one without it."""
    torch.manual_seed(0)
    m = _model(context_lowrank=8)
    batch = {
        "edit_ids": torch.randint(1, 10, (4, 102)),
        "peg_nuc_ids": torch.randint(1, 5, (4, 90)),
        "peg_seg_ids": torch.randint(0, 4, (4, 90)),
        "ctx_cell_type": torch.randint(0, 5, (4,)),
    }
    m.eval()
    with torch.no_grad():
        with_block = m(batch)
        m.context_lowrank = None  # the only difference from the baseline architecture
        without_block = m(batch)
    a = with_block[0] if isinstance(with_block, tuple) else with_block
    b = without_block[0] if isinstance(without_block, tuple) else without_block
    assert torch.equal(a, b)
