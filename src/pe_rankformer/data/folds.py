"""Protospacer-disjoint fivefold cross-validation split (task spec section 10).

All observations sharing a protospacer are assigned to the same fold, using a fixed
seed so the assignment is generated once and reused across every run.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

N_FOLDS = 5
SEED = 20260812


def assign_folds(protospacers: pd.Series, seed: int = SEED, n_folds: int = N_FOLDS) -> pd.Series:
    """Deterministic group-disjoint fold assignment, keyed by protospacer.

    Uses a seeded permutation of the unique protospacer set rather than a raw hash so
    fold balance can be verified and adjusted independently of the hash function.
    """
    unique = np.sort(protospacers.unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    fold_of = {unique[perm[i]]: i % n_folds for i in range(len(unique))}
    return protospacers.map(fold_of)


def deterministic_hash(s: str, length: int = 10) -> str:
    """OptiPrime-compatible hash: sha256(s)[:length]. Used only for cross-checking,
    not for the fold assignment itself (see assign_folds)."""
    return hashlib.sha256(s.encode("ascii")).hexdigest()[:length]


def verify_no_leakage(df: pd.DataFrame, group_col: str = "protospacer", fold_col: str = "fold") -> None:
    """Raise if any group spans more than one fold."""
    n_folds_per_group = df.groupby(group_col)[fold_col].nunique()
    leaking = n_folds_per_group[n_folds_per_group > 1]
    if len(leaking):
        raise AssertionError(
            f"{len(leaking)} protospacers span multiple folds: {leaking.index[:5].tolist()}..."
        )
