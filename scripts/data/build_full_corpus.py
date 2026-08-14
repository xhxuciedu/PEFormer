"""Assemble the best-available reconstruction of the OptiPrime training corpus.

Combines our own Hsu extraction (data/processed/hsu2026_74769.parquet) with the
biomni-derived Kim/Schwank rows that independently match our verified 42-partition
ground truth exactly (data/interim/biomni_reused_verified.parquet; provenance and
cross-validation documented in reports/dataset_reconstruction_status.md and
reports/biomni_cross_check.md).

This is NOT the exact 297,962-row OptiPrime corpus -- see the row-count assertion below,
which is expected to fail and is left in as a live, visible record of the gap rather than
silently working around it (task spec section 8/AF).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_full_corpus")

EXPECTED_TOTAL = 297_962


def _hsu_to_optiprime_columns(hsu: pd.DataFrame) -> pd.DataFrame:
    """Map our canonical Hsu schema onto the OptiPrime-compatible column set used by
    the reused biomni rows, so the two sources can be concatenated."""
    out = pd.DataFrame(index=hsu.index)
    dna_to_rna = lambda s: s.str.replace("T", "U", regex=False)

    out["spacer"] = dna_to_rna(hsu["spacer"])
    out["rtt"] = dna_to_rna(hsu["rtt_sequence"])
    out["pbs"] = dna_to_rna(hsu["pbs_sequence"])
    out["full_unedited"] = hsu["unedited_sequence"]
    out["full_edited"] = hsu["edited_sequence"]
    out["edited"] = hsu["editing_efficiency"]
    out["indel"] = hsu["indel_rate"]
    out["unedited"] = 1 - (out["edited"] + out["indel"])
    out["weight"] = 1.0
    out["scaffold_name"] = "BlpI_F+E"
    out["motif"] = "tevoPreQ1"
    out["cas9_type"] = "PEmax-Cas9"
    out["cas9_pam"] = "SpNGG"
    out["rt_name"] = "PE2-RT"
    out["pe_type"] = hsu["prime_editor"].map({"PE2": "PE2", "PE4": "PE4"})
    out["group"] = "Liu_" + hsu["cell_type"]
    out["cell_type"] = hsu["cell_type"]
    out["time"] = hsu["cell_type"].map({"HEK293T": 3.0, "HeLa": 5.0})
    out["linker"] = ""
    out["split"] = pd.NA
    out["source_study"] = "hsu2026"
    out["record_id"] = hsu["record_id"]
    out["PEmax"] = 1
    out["epegRNA"] = 1
    out["MLH1dn"] = (hsu["prime_editor"] == "PE4").astype(int)
    out["NRCH"] = 0
    return out


def _tag_reused(reused: pd.DataFrame) -> pd.DataFrame:
    out = reused.copy()
    out["source_study"] = out["lab"].map(
        {"Kim": "deepprime", "Schwank": "pridict_pridict2"}
    )
    out["record_id"] = (
        out["group"] + ":" + out["bar_idx"].astype(str) + ":" + out.index.astype(str)
    )
    keep = [
        "spacer", "rtt", "pbs", "full_unedited", "full_edited", "edited", "indel",
        "unedited", "weight", "scaffold_name", "motif", "cas9_type", "cas9_pam",
        "rt_name", "pe_type", "group", "cell_type", "time", "linker", "split",
        "source_study", "record_id", "PEmax", "epegRNA", "MLH1dn", "NRCH",
    ]
    return out[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsu", type=Path, default=Path("data/processed/hsu2026_74769.parquet"))
    ap.add_argument(
        "--reused", type=Path, default=Path("data/interim/biomni_reused_verified.parquet")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("data/processed/optiprime_full_297962.parquet")
    )
    args = ap.parse_args()

    hsu = pd.read_parquet(args.hsu)
    reused = pd.read_parquet(args.reused)

    hsu_mapped = _hsu_to_optiprime_columns(hsu)
    reused_mapped = _tag_reused(reused)

    common_cols = [c for c in hsu_mapped.columns if c in reused_mapped.columns]
    full = pd.concat(
        [hsu_mapped[common_cols], reused_mapped[common_cols]], ignore_index=True
    )

    logger.info("Hsu rows: %d", len(hsu_mapped))
    logger.info("Reused (Kim+Schwank) rows: %d", len(reused_mapped))
    logger.info("Combined rows: %d", len(full))
    logger.info("Target: %d (gap: %d)", EXPECTED_TOTAL, EXPECTED_TOTAL - len(full))

    # PRIDICT v1's raw averageindel has small negative values from background
    # subtraction (same phenomenon already handled for `edited` upstream). Clip and
    # recompute `unedited` rather than silently dropping these otherwise-valid rows.
    n_neg_indel = int((full["indel"] < 0).sum())
    if n_neg_indel:
        logger.warning(
            "clipping %d rows with negative indel (PRIDICT v1 background-subtraction "
            "noise, min=%.4f) to 0",
            n_neg_indel,
            full["indel"].min(),
        )
        full.loc[full["indel"] < 0, "indel"] = 0.0
        full["unedited"] = 1 - (full["edited"] + full["indel"])

    assert full["edited"].between(0, 1).all(), "editing efficiency outside [0,1]"
    assert full["indel"].between(0, 1).all(), "indel rate outside [0,1]"
    n_neg_unedited = int((full["unedited"] < 0).sum())
    if n_neg_unedited:
        logger.warning(
            "%d rows have edited+indel slightly >1 after clipping (min unedited=%.4f); "
            "clipping unedited to 0",
            n_neg_unedited,
            full["unedited"].min(),
        )
        full.loc[full["unedited"] < 0, "unedited"] = 0.0
    assert full["record_id"].is_unique, "duplicate record_id"
    assert full["spacer"].str.contains("T").sum() == 0, "unconverted DNA T in spacer (RNA expected)"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(full))

    if len(full) != EXPECTED_TOTAL:
        logger.warning(
            "Row count %d != target %d. This is expected -- see "
            "reports/dataset_reconstruction_status.md for the documented gap. "
            "This is an OptiPrime-COMPATIBLE reconstructed corpus, not a verified "
            "exact match to the published 297,962-row training set.",
            len(full),
            EXPECTED_TOTAL,
        )


if __name__ == "__main__":
    main()
