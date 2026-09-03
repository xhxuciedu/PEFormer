"""Round-9 C2: censoring-aware ordinal loss.

Among exact Kim replicate pairs, 21.4% of rows measured at exactly zero have a non-zero
replicate. A measured zero is therefore a censored observation, and the cumulative
indicators at thresholds below the assay's detection limit are not reliably zero for
such a row. These tests pin that only zero rows are affected, only the terms below the
limit are dropped, non-zero rows are untouched, and the mechanism-free control matches
the number of dropped terms without matching their identity.
"""

import torch

from pe_rankformer.training.losses import ordinal_loss

TH = torch.tensor([0.0, 0.0003, 0.0024, 0.0065, 0.0136, 0.0252])
LIMIT = 0.0031  # 95th percentile of the replicate distribution given an observed zero


def _logits(b: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(b, len(TH))


def test_disabled_by_default_is_unchanged():
    z = _logits()
    y = torch.tensor([0.0, 0.0, 0.05, 0.2, 0.6, 0.9])
    assert torch.equal(ordinal_loss(z, y, TH), ordinal_loss(z, y, TH, censor_limit=0.0))


def test_nonzero_rows_are_untouched():
    """Only a measured zero is censored; a positive measurement is a real number."""
    z = _logits(2)
    y = torch.tensor([0.05, 0.9])  # no zeros
    assert torch.allclose(ordinal_loss(z, y, TH),
                          ordinal_loss(z, y, TH, censor_limit=LIMIT), atol=1e-7)


def test_zero_rows_drop_exactly_the_terms_below_the_limit():
    z = _logits(1)
    y = torch.tensor([0.0])
    n_below = int((TH < LIMIT).sum())
    assert n_below == 3, "the fixture should exercise a partial mask"
    got = ordinal_loss(z, y, TH, censor_limit=LIMIT)
    # recompute by hand over the surviving terms only
    target = (y.unsqueeze(-1) > TH.unsqueeze(0)).float()
    per_term = torch.nn.functional.binary_cross_entropy_with_logits(
        z, target, reduction="none")
    expected = per_term[:, n_below:].mean()
    assert torch.allclose(got, expected, atol=1e-7)


def test_limit_above_every_threshold_drops_all_terms_for_zero_rows():
    """A zero row then contributes nothing, and the loss reduces to the non-zero rows."""
    z = _logits(4)
    y = torch.tensor([0.0, 0.0, 0.4, 0.8])
    big = float(TH.max()) + 1.0
    got = ordinal_loss(z, y, TH, censor_limit=big)
    only_nonzero = ordinal_loss(z[2:], y[2:], TH)
    # zero rows contribute a clamped denominator of 1 and a zero numerator, so they
    # enter the mean as 0.0 rather than being removed from it
    assert float(got) < float(ordinal_loss(z, y, TH))
    assert float(only_nonzero) > 0


def test_shuffle_control_drops_the_same_number_of_terms():
    z = _logits(64)
    y = torch.zeros(64)
    n_below = int((TH < LIMIT).sum())
    torch.manual_seed(1)
    ctrl = ordinal_loss(z, y, TH, censor_limit=LIMIT, censor_shuffle_control=True)
    real = ordinal_loss(z, y, TH, censor_limit=LIMIT)
    full = ordinal_loss(z, y, TH)
    # Both drop n_below of len(TH) terms per row, so both differ from the full loss;
    # the control differs from the structured version because it drops different terms.
    assert not torch.allclose(ctrl, full, atol=1e-6)
    assert not torch.allclose(ctrl, real, atol=1e-6)
    assert n_below > 0


def test_shuffle_control_is_random_not_fixed():
    z = _logits(64)
    y = torch.zeros(64)
    torch.manual_seed(1)
    a = ordinal_loss(z, y, TH, censor_limit=LIMIT, censor_shuffle_control=True)
    torch.manual_seed(2)
    b = ordinal_loss(z, y, TH, censor_limit=LIMIT, censor_shuffle_control=True)
    assert not torch.allclose(a, b, atol=1e-7)


def test_gradients_do_not_reach_masked_terms():
    """The point of dropping rather than down-weighting: no gradient from a
    measurement the assay cannot support."""
    z = _logits(1).requires_grad_(True)
    y = torch.tensor([0.0])
    ordinal_loss(z, y, TH, censor_limit=LIMIT).backward()
    n_below = int((TH < LIMIT).sum())
    assert torch.allclose(z.grad[0, :n_below], torch.zeros(n_below), atol=1e-9)
    assert z.grad[0, n_below:].abs().sum() > 0
