"""Merge the reconstructed PRIDICT2.0 K562 partition (bar 9) into the training corpus.

Fold assignment here is deliberately NOT a re-run of `build_folds.py`. That script
permutes the sorted unique protospacer set, so introducing new protospacers reshuffles
the fold of every existing one -- which would scramble the locked test fold, invalidate
the OptiPrime baseline predictions already computed on it (expensive, CPU-only JAX), and
break comparability with every result reported so far.

Instead: every protospacer already in the corpus keeps its existing fold, and only
genuinely new protospacers are assigned, filling the smallest folds first to preserve
balance. The script asserts that the locked test fold is bit-identical afterwards.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.folds import N_FOLDS, SEED, verify_no_leakage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("integrate_bar9")

CORPUS = Path("data/processed/optiprime_full_297962.parquet")
BAR9 = Path("data/interim/bar9_pridict2_k562.parquet")
FOLD_MAP = Path("data/processed/fold_assignments.parquet")


def main() -> None:
    corpus = pd.read_parquet(CORPUS)
    new = pd.read_parquet(BAR9)
    logger.info("corpus=%d rows, incoming bar9=%d rows", len(corpus), len(new))

    test_before = corpus[corpus.fold == 0].record_id.sort_values().tolist()
    hsu_test_before = corpus[(corpus.fold == 0) & (corpus.source_study == "hsu2026")]
    logger.info("locked test fold before: %d rows (%d Hsu)", len(test_before), len(hsu_test_before))

    # --- fold assignment: inherit, then fill ------------------------------------
    fold_of = dict(zip(corpus.spacer, corpus.fold))
    known = new.spacer.isin(fold_of)
    logger.info(
        "incoming protospacers: %d already in corpus (inherit fold), %d new",
        known.sum(), (~known).sum(),
    )

    new_spacers = np.sort(new.loc[~known, "spacer"].unique())
    counts = corpus.fold.value_counts().reindex(range(N_FOLDS), fill_value=0).to_dict()
    rng = np.random.default_rng(SEED)
    rows_per_spacer = new.loc[~known].groupby("spacer").size()
    for sp in rng.permutation(new_spacers):
        target = min(counts, key=lambda f: (counts[f], f))  # smallest fold first
        fold_of[sp] = target
        counts[target] += int(rows_per_spacer[sp])
    logger.info("assigned %d new protospacers", len(new_spacers))

    new["fold"] = new.spacer.map(fold_of).astype(int)
    merged = pd.concat([corpus, new[corpus.columns]], ignore_index=True)

    # --- invariants --------------------------------------------------------------
    verify_no_leakage(merged, group_col="spacer", fold_col="fold")
    logger.info("verified: zero protospacer leakage across folds")

    test_after = merged[merged.fold == 0].record_id.sort_values().tolist()
    hsu_after = merged[(merged.fold == 0) & (merged.source_study == "hsu2026")]
    assert len(hsu_after) == len(hsu_test_before), "Hsu test-fold row count changed"
    assert set(test_before).issubset(set(test_after)), "existing test-fold rows were reassigned"
    assert (
        hsu_after.record_id.sort_values().tolist()
        == hsu_test_before.record_id.sort_values().tolist()
    ), "Hsu test fold is not identical -- OptiPrime comparison would be invalidated"
    logger.info("verified: Hsu test fold unchanged (%d rows) -- baseline stays valid", len(hsu_after))

    for f in range(N_FOLDS):
        n = (merged.fold == f).sum()
        logger.info("fold %d: %d rows (%.1f%%)", f, n, 100 * n / len(merged))

    merged.to_parquet(CORPUS, index=False)
    logger.info("wrote %s (%d rows, was %d)", CORPUS, len(merged), len(corpus))

    fm = merged[["spacer", "fold"]].drop_duplicates("spacer").reset_index(drop=True)
    fm.to_parquet(FOLD_MAP, index=False)
    logger.info("wrote %s (%d protospacers)", FOLD_MAP, len(fm))


if __name__ == "__main__":
    main()
