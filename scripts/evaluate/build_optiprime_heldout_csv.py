"""Build OptiPrime-loadable CSVs for OptiPrime's OWN held-out test rows.

Unlike `build_optiprime_test_csv.py`, this needs no padding workaround. The authors'
files are already in OptiPrime's native input format: every held-out Liu row places the
20nt protospacer at exactly PS20_OFFSET=4, verified below. That retires the 4bp-padding
caveat that qualified the earlier baseline numbers.

Only the Liu rows are emitted. OptiPrime's Kim/Schwank loaders key metadata off filename
conventions tied to their own library naming, and the Liu held-out set (9,175 rows) is
the partition the head-to-head has always been run on, so it keeps the comparison
directly commensurable with earlier results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_heldout_csv")

PS20_OFFSET = 4
OUT_DIR = Path("data/interim/optiprime_heldout_test")


def main() -> None:
    df = pd.read_parquet("data/processed/optiprime_official_318471.parquet")
    test = df[(df.fold == 0) & (df.source_study == "hsu2026")].copy()
    logger.info("OptiPrime held-out Liu test rows: %d", len(test))

    ps20 = test.spacer.str.replace("U", "T", regex=False).str[-20:]
    offsets = [u.find(p) for u, p in zip(test.full_unedited, ps20)]
    bad = sum(1 for o in offsets if o != PS20_OFFSET)
    assert bad == 0, f"{bad} rows do not place the protospacer at offset {PS20_OFFSET}"
    logger.info("verified: all rows already at PS20_OFFSET=%d -- no padding applied", PS20_OFFSET)

    out = pd.DataFrame(
        {
            "spacer": test.spacer,
            "rtt": test.rtt,
            "pbs": test.pbs,
            "full_unedited": test.full_unedited,
            "full_edited": test.full_edited,
            "edited_frac": test.edited,
            "indel_frac": test.indel,
            "record_id": test.record_id,
            "cell_type": test.cell_type,
            "pe_type": test.pe_type,
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for (cell, pe), g in out.groupby(["cell_type", "pe_type"]):
        # Name so OptiPrime's process_liu() reads cell type from part 1 and PE from part 3.
        fname = OUT_DIR / f"Liu_{cell}_HeldOut_{pe}.csv"
        g.drop(columns=["cell_type", "pe_type"]).to_csv(fname, index=False)
        logger.info("wrote %s (%d rows)", fname, len(g))


if __name__ == "__main__":
    main()
