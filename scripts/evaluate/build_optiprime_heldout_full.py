"""Build OptiPrime-loadable CSVs for the COMPLETE held-out test set (20,509 rows).

Supersedes `build_optiprime_heldout_csv.py`, which emitted only the 9,175 Liu rows.
The 11,334 Kim rows were initially skipped out of caution about reproducing OptiPrime's
Kim metadata conventions; in fact the authors' files are already named exactly as
`process_hkim` expects (`Kim_{cell}_{lib}_{details}_test.csv`), so they can be handed
to the loader unmodified.

One subtlety worth recording: in the Kim files the leading lowercase `g` of the spacer
is a *substitution* of the first protospacer base (the U6 transcription-start
requirement), not an extra base as in the Liu files. So `spacer[1:]` matches the genome
at offset 5 while the true protospacer sits at offset 4. Checking for the spacer
verbatim inside `full_unedited` therefore fails on ~77% of Kim rows and looks like a
format error when it is not.

record_id is attached from the official corpus parquet by (source_file, row order),
which is how that parquet was assembled, so predictions can be joined back to
PE-RankFormer's.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_heldout_full")

SRC = Path("data/optiprime_train_mix")
OUT_DIR = Path("data/interim/optiprime_heldout_full")
CORPUS = Path("data/processed/optiprime_official_318471.parquet")


def main() -> None:
    corpus = pd.read_parquet(CORPUS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Liu: reuse the already-verified per-context files -------------------------
    liu_src = Path("data/interim/optiprime_heldout_test")
    n_liu = 0
    for f in sorted(liu_src.glob("Liu_*.csv")):
        shutil.copy(f, OUT_DIR / f.name)
        n_liu += len(pd.read_csv(f))
    logger.info("copied %d Liu held-out rows from %s", n_liu, liu_src)

    # --- Kim: the authors' own *_test.csv files, plus record_id --------------------
    n_kim = 0
    for f in sorted(SRC.glob("Kim_*_test.csv")):
        stem = f.stem
        d = pd.read_csv(f)
        rows = corpus[corpus.source_file == stem]
        assert len(rows) == len(d), f"{stem}: {len(rows)} corpus rows vs {len(d)} file rows"
        d = d.copy()
        d["record_id"] = rows.record_id.to_numpy()  # same file, same order
        d.to_csv(OUT_DIR / f.name, index=False)
        n_kim += len(d)
    logger.info("wrote %d Kim held-out rows", n_kim)

    total = n_liu + n_kim
    logger.info("total held-out rows: %d", total)
    assert total == 20_509, f"expected 20,509 held-out rows, got {total}"
    logger.info("matches the official held-out count exactly")


if __name__ == "__main__":
    main()
