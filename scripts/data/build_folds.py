"""Build data/processed/fold_assignments.parquet and verify zero leakage."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.folds import N_FOLDS, SEED, assign_folds, verify_no_leakage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_folds")


def main() -> None:
    df = pd.read_parquet("data/processed/optiprime_full_297962.parquet")
    logger.info("loaded %d rows, %d unique spacers", len(df), df["spacer"].nunique())

    df["fold"] = assign_folds(df["spacer"], seed=SEED, n_folds=N_FOLDS)
    verify_no_leakage(df, group_col="spacer", fold_col="fold")
    logger.info("verified: zero leakage across %d folds (seed=%d)", N_FOLDS, SEED)

    for f in range(N_FOLDS):
        n = (df["fold"] == f).sum()
        n_sp = df.loc[df["fold"] == f, "spacer"].nunique()
        logger.info("fold %d: %d rows (%.1f%%), %d spacers", f, n, 100 * n / len(df), n_sp)

    fold_assignments = (
        df[["spacer", "fold"]].drop_duplicates(subset=["spacer"]).reset_index(drop=True)
    )
    out = Path("data/processed/fold_assignments.parquet")
    fold_assignments.to_parquet(out, index=False)
    logger.info("wrote %s (%d spacer->fold rows)", out, len(fold_assignments))

    df.to_parquet("data/processed/optiprime_full_297962.parquet", index=False)
    logger.info("updated corpus parquet with `fold` column")


if __name__ == "__main__":
    main()
