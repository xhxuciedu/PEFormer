"""Build 3 Liu+Kim-matched, protospacer-disjoint development folds from the 297,962
training rows (round-3 spec §5).

Round 2's central finding: Schwank is 58% of the training/CV pool but 0% of the
official held-out set (Liu 44.7% + Kim 55.3%, exactly). Every round-1/round-2
validation split was drawn uniformly from all of folds 1-5 and was therefore
majority-Schwank, while the actual target is Liu+Kim only -- round-2's confirmed,
stable validation gain (+0.0028, std 0.0011 across 5 folds) did not transfer to a
held-out improvement, plausibly because of this mismatch.

Design: repeated random sub-sampling (not an exhaustive 3-way partition -- each
fold's own train/val split is disjoint, but the 3 folds' validation sets may
overlap with each other, which is standard for repeated-holdout CV and much
simpler than forcing a joint partition). For dev fold i: shuffle Liu protospacers
and Kim protospacers independently (seeded per fold), and take a prefix of each,
by CUMULATIVE ROW COUNT (not protospacer count), until the validation set reaches
the target ~44.7% Liu / 55.3% Kim row ratio at the target validation size.
Row-count weighting matters: Liu averages 65.8 rows/protospacer vs Kim's 38.7, so
a protospacer-count-balanced split would skew ~57% Liu by rows despite drawing
equal protospacer shares -- confirmed empirically before this fix shipped.

Any protospacer that also appears in Schwank is pinned to train unconditionally
in every fold (Schwank is otherwise never excluded from training, and letting a
shared protospacer sit in one source's validation set while its Schwank copy
sits in train would be a real leak).

The held-out fold (fold==0) is entirely excluded -- these are folds of the
297,962 *training* rows only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_round3_dev_folds")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
OUT = Path("data/processed/round3_dev_assignments.parquet")
N_DEV_FOLDS = 3
BASE_SEED = 20260819
VAL_FRACTION = 0.12  # ~12% of the training pool per dev fold's validation set
TARGET_LIU_FRAC = 9175 / (9175 + 11334)  # exact held-out composition: 44.74% Liu
TARGET_KIM_FRAC = 1 - TARGET_LIU_FRAC


def _select_val_spacers(
    row_counts: pd.Series, target_rows: float, seed: int
) -> set[str]:
    """Shuffle spacers (seeded) and take a cumulative-row-count prefix closest to
    `target_rows`, without exceeding it by more than one spacer's worth."""
    rng = np.random.default_rng(seed)
    spacers = row_counts.index.to_numpy()
    rng.shuffle(spacers)
    cum = 0
    chosen = []
    for sp in spacers:
        n = row_counts[sp]
        if cum + n > target_rows and chosen:
            break
        chosen.append(sp)
        cum += n
    return set(chosen)


def main() -> None:
    df = pd.read_parquet(CORPUS, columns=["record_id", "spacer", "source_study", "fold"])
    train_pool = df[df.fold != 0].copy()
    logger.info("training pool (fold != 0): %d rows", len(train_pool))
    logger.info("  by source: %s", train_pool.source_study.value_counts().to_dict())

    liu = train_pool.source_study == "hsu2026"
    kim = train_pool.source_study == "deepprime"
    schwank = train_pool.source_study == "pridict_pridict2"
    assert (liu | kim | schwank).all()

    schwank_spacers = set(train_pool.loc[schwank, "spacer"])
    liu_eligible = train_pool.loc[liu & ~train_pool.spacer.isin(schwank_spacers)]
    kim_eligible = train_pool.loc[kim & ~train_pool.spacer.isin(schwank_spacers)]
    n_excluded = int((liu | kim).sum() - len(liu_eligible) - len(kim_eligible))
    logger.info(
        "Liu/Kim rows whose protospacer touches Schwank (excluded from val eligibility): %d",
        n_excluded,
    )

    liu_row_counts = liu_eligible.groupby("spacer").size()
    kim_row_counts = kim_eligible.groupby("spacer").size()

    target_val_rows = VAL_FRACTION * len(train_pool)
    target_liu_rows = target_val_rows * TARGET_LIU_FRAC
    target_kim_rows = target_val_rows * TARGET_KIM_FRAC
    logger.info(
        "target per fold: ~%.0f val rows (Liu %.0f [%.1f%%], Kim %.0f [%.1f%%])",
        target_val_rows, target_liu_rows, 100 * TARGET_LIU_FRAC, target_kim_rows, 100 * TARGET_KIM_FRAC,
    )

    out = train_pool[["record_id", "source_study"]].copy()
    for i in range(N_DEV_FOLDS):
        val_liu_spacers = _select_val_spacers(liu_row_counts, target_liu_rows, seed=BASE_SEED + 10 * i + 1)
        val_kim_spacers = _select_val_spacers(kim_row_counts, target_kim_rows, seed=BASE_SEED + 10 * i + 2)
        is_val = (liu & train_pool.spacer.isin(val_liu_spacers)) | (kim & train_pool.spacer.isin(val_kim_spacers))
        out[f"round3_dev_fold_{i}"] = np.where(is_val, "val", "train")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logger.info("wrote %s (%d rows)", OUT, len(out))

    # --- verification -------------------------------------------------------------
    dev_cols = [f"round3_dev_fold_{i}" for i in range(N_DEV_FOLDS)]
    check = train_pool[["record_id", "spacer", "source_study"]].merge(
        out[["record_id"] + dev_cols], on="record_id"
    )
    for i in range(N_DEV_FOLDS):
        col = f"round3_dev_fold_{i}"
        val = check[check[col] == "val"]
        train = check[check[col] == "train"]
        assert (val.source_study != "pridict_pridict2").all(), "Schwank leaked into a dev validation set"
        overlap = set(val.spacer) & set(train.spacer)
        assert not overlap, f"{col}: {len(overlap)} protospacers appear in both train and val"
        assert (train.source_study == "pridict_pridict2").sum() == schwank.sum(), "Schwank row dropped from train"

        n_liu, n_kim = (val.source_study == "hsu2026").sum(), (val.source_study == "deepprime").sum()
        n_val = len(val)
        logger.info(
            "%s: train=%d val=%d (Liu %d [%.1f%%], Kim %d [%.1f%%])",
            col, len(train), n_val, n_liu, 100 * n_liu / n_val, n_kim, 100 * n_kim / n_val,
        )

    logger.info("verified: protospacer-disjoint per fold, zero Schwank in any dev validation set")


if __name__ == "__main__":
    main()
