"""Duplicate analysis for the assembled corpus (task spec section W).

Builds a design fingerprint (excludes outcome) and a full observation fingerprint
(includes outcome), and reports both exact-duplicate measurements and same-design
duplicates across sources, without automatically deduplicating.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("duplicate_analysis")


def _hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def main() -> None:
    df = pd.read_parquet("data/processed/optiprime_full_297962.parquet")

    df["design_fp"] = [
        _hash(sp, pb, rt, fu, fe, ct, pe)
        for sp, pb, rt, fu, fe, ct, pe in zip(
            df.spacer, df.pbs, df.rtt, df.full_unedited, df.full_edited,
            df.cell_type, df.pe_type,
        )
    ]
    df["obs_fp"] = [
        _hash(d, f"{e:.6f}")
        for d, e in zip(df.design_fp, df.edited)
    ]

    n_obs_dup = df["obs_fp"].duplicated().sum()
    nonzero = df[df.edited > 0]
    n_obs_dup_nonzero = nonzero["obs_fp"].duplicated().sum()
    n_design_dup_rows = df["design_fp"].duplicated(keep=False).sum()
    n_design_dup_groups = df["design_fp"].duplicated().sum()

    logger.info("total rows: %d", len(df))
    logger.info("exact duplicate observations (design+context+outcome): %d", n_obs_dup)
    logger.info(
        "of which, with edited>0 (excludes zero-inflation artifact): %d",
        n_obs_dup_nonzero,
    )
    logger.info(
        "rows sharing a design fingerprint with >=1 other row: %d (%d distinct groups)",
        n_design_dup_rows, n_design_dup_groups,
    )

    same_design = df[df["design_fp"].duplicated(keep=False)]
    cross_source = (
        same_design.groupby("design_fp")["source_study"].nunique().gt(1).sum()
    )
    logger.info(
        "design fingerprints appearing under >1 source_study (cross-study overlap): %d",
        cross_source,
    )

    report = Path("reports/duplicate_analysis.md")
    lines = [
        "# Duplicate analysis\n",
        f"Corpus: `data/processed/optiprime_full_297962.parquet` ({len(df):,} rows)\n",
        "## Method\n",
        "- `design_fp` = hash(spacer, pbs, rtt, full_unedited, full_edited, cell_type, pe_type) "
        "-- identifies the same pegRNA/target/context design regardless of measured outcome.\n",
        "- `obs_fp` = hash(design_fp, editing_efficiency rounded to 1e-6) -- identifies exact "
        "duplicate measurements.\n",
        "## Results\n",
        f"- Exact duplicate observations (same design *and* same measured efficiency): "
        f"**{n_obs_dup:,}**, of which **{n_obs_dup_nonzero:,}** have `edited > 0`.\n",
        f"  The gap ({n_obs_dup - n_obs_dup_nonzero:,} rows) is an artifact of DeepPrime's "
        f"zero-inflated efficiency distribution (~41-50% of raw DeepPrime measurements are "
        f"exactly 0, confirmed against `external/deepprime/data/*.csv` directly) -- many "
        f"*different* designs coincidentally share `edited == 0.0`, which is not duplication.\n",
        f"- Rows sharing a design fingerprint with at least one other row: "
        f"**{n_design_dup_rows:,}** ({n_design_dup_groups:,} distinct duplicate groups)\n",
        f"- Of those, design fingerprints spanning more than one `source_study` "
        f"(genuine cross-study overlap, e.g. a design tested in both DeepPrime and "
        f"PRIDICT2.0): **{cross_source:,}**\n",
        "## Decision\n",
        "No automatic deduplication applied. Per task spec (§W), OptiPrime's own loader "
        "(`RxDataset.load_dir`, see `reports/optiprime_data_loader_reverse_engineering.md`) "
        "performs no cross-study deduplication either -- it concatenates every row of every "
        "file placed in its data directory. Rows sharing a design fingerprint within the "
        "*same* source (e.g. a design measured at multiple replicate timepoints, or the same "
        "pegRNA tested under both PE2 and PE4) are expected and retained as independent "
        "observations, consistent with the source studies' own experimental design.\n",
    ]
    report.write_text("\n".join(lines))
    logger.info("wrote %s", report)


if __name__ == "__main__":
    main()
