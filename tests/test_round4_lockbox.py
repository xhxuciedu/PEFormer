"""Round-4 internal lockbox: isolation guarantees (spec §4).

The lockbox's value depends entirely on it never having been used for selection,
so these check the properties that make that true. Integration checks over real
pipeline output; skip if the data isn't built in this environment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
DEV = Path("data/processed/round3_dev_assignments.parquet")
LOCKBOX = Path("data/processed/round4_lockbox.parquet")

pytestmark = pytest.mark.skipif(
    not (CORPUS.exists() and DEV.exists() and LOCKBOX.exists()),
    reason="processed data not built in this environment",
)


@pytest.fixture(scope="module")
def lockbox():
    return pd.read_parquet(LOCKBOX)


def test_no_official_heldout_rows(lockbox):
    """The lockbox must not overlap the official held-out set (fold 0)."""
    assert (lockbox.fold != 0).all()


def test_no_schwank(lockbox):
    assert (lockbox.source_study != "pridict_pridict2").all()


def test_liu_kim_ratio_matches_official_heldout(lockbox):
    liu_frac = (lockbox.source_study == "hsu2026").mean()
    assert abs(liu_frac - 9175 / (9175 + 11334)) < 0.02


def test_disjoint_from_every_dev_validation_set(lockbox):
    """The defining property: no model has been selected using these rows."""
    corp = pd.read_parquet(CORPUS, columns=["record_id", "spacer", "fold"])
    dev = pd.read_parquet(DEV)
    devcols = [c for c in dev.columns if c.startswith("round3_dev_fold_")]
    m = corp.merge(dev[["record_id"] + devcols], on="record_id")
    ever_val_spacers = set(m.loc[(m[devcols] == "val").any(axis=1), "spacer"])
    assert not (set(lockbox.spacer) & ever_val_spacers)


def test_spans_all_official_folds_for_oof_scoring(lockbox):
    """OOF scoring needs every row to have a checkpoint that held its fold out."""
    assert set(lockbox.fold.unique()) == {1, 2, 3, 4, 5}


def test_protospacer_grouping_is_intact(lockbox):
    """Every row of a selected protospacer should be present, not a partial slice."""
    corp = pd.read_parquet(CORPUS, columns=["record_id", "spacer", "source_study", "fold"])
    pool = corp[(corp.fold != 0) & corp.source_study.isin(["hsu2026", "deepprime"])]
    counts_in_pool = pool.groupby("spacer").size()
    counts_in_lockbox = lockbox.groupby("spacer").size()
    for sp, n in counts_in_lockbox.items():
        assert n == counts_in_pool[sp], f"protospacer {sp} only partially included"
