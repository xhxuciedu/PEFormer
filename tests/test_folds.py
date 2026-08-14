"""Fold construction: leakage and determinism (task spec section 40)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pe_rankformer.data.folds import N_FOLDS, assign_folds, verify_no_leakage


def _toy_df(n_groups=20, rows_per_group=3, seed=0):
    rng = np.random.default_rng(seed)
    groups = [f"spacer_{i}" for i in range(n_groups)]
    rows = []
    for g in groups:
        for _ in range(rng.integers(1, rows_per_group + 1)):
            rows.append(g)
    return pd.DataFrame({"spacer": rows})


def test_all_records_for_same_protospacer_in_one_fold():
    df = _toy_df()
    df["fold"] = assign_folds(df["spacer"], seed=1)
    verify_no_leakage(df, group_col="spacer", fold_col="fold")


def test_leakage_detected_when_present():
    df = pd.DataFrame({"spacer": ["a", "a", "b"], "fold": [0, 1, 0]})
    with pytest.raises(AssertionError):
        verify_no_leakage(df, group_col="spacer", fold_col="fold")


def test_deterministic_across_runs():
    df = _toy_df()
    f1 = assign_folds(df["spacer"], seed=42)
    f2 = assign_folds(df["spacer"], seed=42)
    assert (f1 == f2).all()


def test_all_folds_used():
    df = _toy_df(n_groups=100)
    folds = assign_folds(df["spacer"], seed=7)
    assert set(folds.unique()) == set(range(N_FOLDS))


def test_train_val_test_disjoint_spacers():
    df = _toy_df(n_groups=50)
    df["fold"] = assign_folds(df["spacer"], seed=3)
    test = df[df.fold == 0]
    train = df[df.fold != 0]
    assert not (set(train.spacer) & set(test.spacer))
