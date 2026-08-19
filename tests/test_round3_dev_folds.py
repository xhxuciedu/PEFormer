"""Round-3 Liu+Kim-matched development folds: regression checks on the assembled
assignment file (spec §5). Integration checks over real pipeline output, not pure
unit tests -- skips if the data hasn't been built in this environment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

CORPUS_PATH = Path("data/processed/optiprime_official_318471.parquet")
DEV_PATH = Path("data/processed/round3_dev_assignments.parquet")

pytestmark = pytest.mark.skipif(
    not CORPUS_PATH.exists() or not DEV_PATH.exists(),
    reason="processed data not built in this environment",
)

N_DEV_FOLDS = 3


@pytest.fixture(scope="module")
def merged():
    corpus = pd.read_parquet(CORPUS_PATH, columns=["record_id", "spacer", "source_study", "fold"])
    dev = pd.read_parquet(DEV_PATH)
    dev_cols = ["record_id"] + [f"round3_dev_fold_{i}" for i in range(N_DEV_FOLDS)]
    return corpus.merge(dev[dev_cols], on="record_id")


def test_covers_exactly_the_training_pool(merged):
    corpus = pd.read_parquet(CORPUS_PATH, columns=["record_id", "fold"])
    assert len(merged) == (corpus.fold != 0).sum()
    assert (merged.fold != 0).all()


@pytest.mark.parametrize("i", range(N_DEV_FOLDS))
def test_no_schwank_in_validation(merged, i):
    col = f"round3_dev_fold_{i}"
    val = merged[merged[col] == "val"]
    assert (val.source_study != "pridict_pridict2").all()


@pytest.mark.parametrize("i", range(N_DEV_FOLDS))
def test_protospacer_disjoint(merged, i):
    col = f"round3_dev_fold_{i}"
    val_spacers = set(merged.loc[merged[col] == "val", "spacer"])
    train_spacers = set(merged.loc[merged[col] == "train", "spacer"])
    assert not (val_spacers & train_spacers)


@pytest.mark.parametrize("i", range(N_DEV_FOLDS))
def test_all_schwank_rows_are_training(merged, i):
    col = f"round3_dev_fold_{i}"
    schwank = merged[merged.source_study == "pridict_pridict2"]
    assert (schwank[col] == "train").all()


@pytest.mark.parametrize("i", range(N_DEV_FOLDS))
def test_liu_kim_ratio_matches_heldout_composition(merged, i):
    """Held-out composition is exactly 9175 Liu / 11334 Kim = 44.74% / 55.26%."""
    col = f"round3_dev_fold_{i}"
    val = merged[merged[col] == "val"]
    liu_frac = (val.source_study == "hsu2026").mean()
    assert abs(liu_frac - 9175 / (9175 + 11334)) < 0.01  # within 1 percentage point


def test_three_folds_give_independent_validation_sets(merged):
    """Repeated-holdout folds may overlap, but shouldn't be near-identical."""
    val_sets = [
        set(merged.loc[merged[f"round3_dev_fold_{i}"] == "val", "record_id"])
        for i in range(N_DEV_FOLDS)
    ]
    for i in range(N_DEV_FOLDS):
        for j in range(i + 1, N_DEV_FOLDS):
            jaccard = len(val_sets[i] & val_sets[j]) / len(val_sets[i] | val_sets[j])
            assert jaccard < 0.5, f"fold {i} and {j} validation sets are too similar"
