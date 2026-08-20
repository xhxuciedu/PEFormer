"""Emit the round-4 lockbox in the fold-file format `evaluate_on_devfolds.py` reads.

That script selects rows by `column == "val"`, while the lockbox parquet marks
membership with a boolean. This writes an equivalent file with a `round4_lockbox_val`
column so the lockbox can be scored by the same evaluation path as the dev folds --
including its `--oof` mode, which is what makes the lockbox usable at all.

**Why OOF is mandatory here, not optional.** The lockbox holds protospacers that
never appeared in any dev-fold *validation* set, so it is fresh for model *selection*
after rounds 3-4 wore the dev folds out. It is not fresh for *training*: its rows are
spread across official folds 1-5 (3,303-4,166 rows each), so every member trained on
the official split saw roughly four fifths of them. Scored naively, a Phase-2 member
would be reporting largely in-sample performance. Scored OOF -- each row predicted
only by the checkpoint that held its official fold out -- it is clean, and directly
comparable to the incumbents, which are OOF by construction.

Verified at build time: no fold-0 (held-out) rows, no record or protospacer overlap
with any dev-fold validation set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_lockbox_foldfile")

LOCKBOX = Path("data/processed/round4_lockbox.parquet")
DEV = Path("data/processed/round3_dev_assignments.parquet")
CORPUS = Path("data/processed/optiprime_official_318471.parquet")
OUT = Path("data/processed/round4_lockbox_folds.parquet")


def main() -> None:
    lb = pd.read_parquet(LOCKBOX)
    dev = pd.read_parquet(DEV)
    src = pd.read_parquet(CORPUS, columns=["record_id", "spacer"])

    # Re-assert the invariants rather than trusting the upstream build: this file is
    # the gate that decides what gets promoted, so a silent contamination here would
    # propagate straight into the final model choice.
    assert not (lb.fold == 0).any(), "lockbox contains held-out fold-0 rows"
    dev_cols = [c for c in dev.columns if c.startswith("round3_dev_fold")]
    val_ids = set()
    for c in dev_cols:
        val_ids |= set(dev.loc[dev[c] == "val", "record_id"])
    assert not (set(lb.record_id) & val_ids), "lockbox overlaps a dev validation row"
    dev_spacers = set(src.loc[src.record_id.isin(val_ids), "spacer"])
    assert not (set(lb.spacer) & dev_spacers), "lockbox shares a protospacer with dev val"

    out = pd.DataFrame({"record_id": lb.record_id, "round4_lockbox_val": "val"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logger.info(
        "wrote %s: %d rows, %d protospacers, folds %s",
        OUT, len(out), lb.spacer.nunique(), sorted(lb.fold.unique()),
    )
    logger.info("evaluate with --dev-folds-file %s --dev-fold-cols round4_lockbox_val --oof", OUT)


if __name__ == "__main__":
    main()
