"""Build the round-4 internal lockbox (spec §4).

Three-level evaluation hierarchy for round 4:

  1. matched dev folds  (data/processed/round3_dev_assignments.parquet)
     -- used freely for Stage-A exploration and model selection.
  2. round-4 internal lockbox  (this file)
     -- touched ONCE, to screen the shortlist before full 5-fold training.
  3. official held-out, 20,509 rows (fold 0)
     -- touched once, after the final freeze.

Why a lockbox is needed: the official held-out set has now been examined after
rounds 1, 2 and 3. Each examination is a (small) selection pressure. The lockbox
gives an untouched intermediate checkpoint so round-4 candidates can be screened
without spending the official set again.

**Construction and its one real subtlety.** Lockbox rows are drawn from the
297,962 training rows, so any model trained on the official 5-fold split *has*
trained on lockbox rows that fall in folds other than its own held-out fold.
The lockbox is therefore clean in exactly the sense the dev folds are: it must
be scored **out-of-fold** (each row scored only by the checkpoint that held that
row's official fold out). What makes it a genuine lockbox is not a stronger
data-isolation property, but a *usage* property -- it is disjoint from every
dev-fold validation set, so no model has ever been selected, early-stopped, or
weighted using these rows.

Eligibility: Liu+Kim protospacers that (a) never appear in any round-3 dev-fold
validation set, and (b) do not also appear in Schwank (which is unconditionally
training data, so a shared protospacer would straddle train and lockbox).
Selection is row-count-weighted to reproduce the official held-out set's exact
44.74% Liu / 55.26% Kim composition -- protospacer-count weighting would skew
Liu-heavy, since Liu averages far more rows per protospacer than Kim.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_round4_lockbox")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
DEV = Path("data/processed/round3_dev_assignments.parquet")
OUT = Path("data/processed/round4_lockbox.parquet")

SEED = 20260820
TARGET_ROWS = 18_000
TARGET_LIU_FRAC = 9175 / (9175 + 11334)  # 0.4474, the official held-out composition


def _select_by_row_count(row_counts: pd.Series, target_rows: float, seed: int) -> set[str]:
    rng = np.random.default_rng(seed)
    spacers = row_counts.index.to_numpy().copy()
    rng.shuffle(spacers)
    cum, chosen = 0, []
    for sp in spacers:
        n = int(row_counts[sp])
        if cum + n > target_rows and chosen:
            break
        chosen.append(sp)
        cum += n
    return set(chosen)


def main() -> None:
    corp = pd.read_parquet(CORPUS, columns=["record_id", "spacer", "source_study", "fold"])
    dev = pd.read_parquet(DEV)
    devcols = [c for c in dev.columns if c.startswith("round3_dev_fold_")]

    train_pool = corp[corp.fold != 0]
    m = train_pool.merge(dev[["record_id"] + devcols], on="record_id")

    schwank_spacers = set(m.loc[m.source_study == "pridict_pridict2", "spacer"])
    lk = m[m.source_study.isin(["hsu2026", "deepprime"])]

    ever_val = (lk[devcols] == "val").any(axis=1)
    used_as_val = set(lk.loc[ever_val, "spacer"])

    eligible = lk[~lk.spacer.isin(used_as_val) & ~lk.spacer.isin(schwank_spacers)]
    logger.info(
        "eligible pool: %d rows over %d protospacers (never a dev-val protospacer, no Schwank collision)",
        len(eligible), eligible.spacer.nunique(),
    )
    logger.info("  eligible composition: %s", eligible.source_study.value_counts().to_dict())

    liu = eligible[eligible.source_study == "hsu2026"]
    kim = eligible[eligible.source_study == "deepprime"]
    target_liu = TARGET_ROWS * TARGET_LIU_FRAC
    target_kim = TARGET_ROWS * (1 - TARGET_LIU_FRAC)
    logger.info("target: %d rows (Liu %.0f, Kim %.0f)", TARGET_ROWS, target_liu, target_kim)

    liu_sel = _select_by_row_count(liu.groupby("spacer").size(), target_liu, SEED + 1)
    kim_sel = _select_by_row_count(kim.groupby("spacer").size(), target_kim, SEED + 2)
    selected = liu_sel | kim_sel

    lock = eligible[eligible.spacer.isin(selected)]
    out = lock[["record_id", "spacer", "source_study", "fold"]].copy()
    out["round4_lockbox"] = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    # --- verification -------------------------------------------------------------
    n_liu = int((out.source_study == "hsu2026").sum())
    n_kim = int((out.source_study == "deepprime").sum())
    logger.info(
        "lockbox: %d rows, %d protospacers (Liu %d [%.1f%%], Kim %d [%.1f%%])",
        len(out), out.spacer.nunique(), n_liu, 100 * n_liu / len(out), n_kim, 100 * n_kim / len(out),
    )
    assert abs(n_liu / len(out) - TARGET_LIU_FRAC) < 0.02, "Liu fraction off target"
    assert (out.source_study != "pridict_pridict2").all(), "Schwank in lockbox"
    assert (out.fold != 0).all(), "official held-out row leaked into lockbox"
    assert not (set(out.spacer) & used_as_val), "lockbox protospacer was used as dev validation"
    assert not (set(out.spacer) & schwank_spacers), "lockbox protospacer also appears in Schwank"

    # every official fold should be represented, else OOF scoring can't cover the set
    per_fold = out.fold.value_counts().sort_index().to_dict()
    logger.info("  rows per official fold (for OOF scoring): %s", per_fold)
    assert set(per_fold) == {1, 2, 3, 4, 5}, f"lockbox does not span all official folds: {per_fold}"

    logger.info("verified: disjoint from every dev-fold validation set, no Schwank, no held-out rows")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
