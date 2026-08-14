"""Build an OptiPrime-loadable CSV for the locked Hsu test-fold rows.

OptiPrime's format_pe_df requires `full_unedited` to place the 20nt protospacer at a
fixed offset (PS20_OFFSET=4) with `proto30 = full_unedited[:30]` exactly 30nt. Our Hsu
extraction's LibMMR rows have the protospacer starting at offset 0 (no upstream context
available in the public Supplementary workbook), and LibCV rows start at offset 3-4.

This is the known limitation documented in
reports/optiprime_data_loader_reverse_engineering.md section 6: we do not have the true
upstream genomic context for LibMMR designs. We left-pad with a fixed non-genomic filler
('N' x k) to reach the required offset, and note this explicitly wherever these baseline
results are reported -- OptiPrime's PAM/seed-window features read from these padded
positions will not reflect real genomic sequence for LibMMR rows.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_optiprime_test_csv")

PS20_OFFSET = 4


def pad_to_offset(unedited: str, edited: str, protospacer: str) -> tuple[str, int]:
    """Left-pad both sequences so the protospacer starts at PS20_OFFSET. Returns
    (padded_pair, n_padded) where padded_pair is (unedited, edited)."""
    start = unedited.find(protospacer)
    assert start >= 0, "protospacer must already be a substring of unedited"
    pad_needed = PS20_OFFSET - start
    if pad_needed <= 0:
        # already has enough (or too much) upstream context; trim from the left instead
        trim = -pad_needed
        return unedited[trim:], edited[trim:], 0
    # OptiPrime's seq_encoding one-hots strictly over ACGT (no ambiguity codes), so the
    # filler must be a real base. 'A' is an arbitrary, clearly-non-genomic placeholder
    # documented here and in the run report -- not a claim about the true upstream
    # sequence, which the public Supplementary data doesn't provide (see module docstring).
    filler = "A" * pad_needed
    return filler + unedited, filler + edited, pad_needed


def main() -> None:
    hsu = pd.read_parquet("data/processed/hsu2026_74769.parquet")
    folds = pd.read_parquet("data/processed/fold_assignments.parquet")
    folds["spacer_dna"] = folds["spacer"].str.replace("U", "T", regex=False)
    merged = hsu.merge(folds, left_on="spacer", right_on="spacer_dna", how="left")
    test = merged[merged.fold == 0].copy()
    logger.info("test fold rows: %d", len(test))

    padded_u, padded_e, n_pad = [], [], []
    for u, e, ps in zip(test.unedited_sequence, test.edited_sequence, test.protospacer):
        pu, pe, n = pad_to_offset(u, e, ps)
        padded_u.append(pu)
        padded_e.append(pe)
        n_pad.append(n)
    test["full_unedited"] = padded_u
    test["full_edited"] = padded_e
    test["n_padded"] = n_pad

    logger.info("padding distribution:\n%s", test.n_padded.value_counts().sort_index().to_string())

    out = pd.DataFrame(
        {
            "spacer": test.spacer_x,
            "rtt": test.rtt_sequence,
            "pbs": test.pbs_sequence,
            "full_unedited": test.full_unedited,
            "full_edited": test.full_edited,
            "edited_frac": test.editing_efficiency,
            "indel_frac": test.indel_rate,
            "record_id": test.record_id,
            "cell_type": test.cell_type,
            "prime_editor": test.prime_editor,
            "n_padded": test.n_padded,
        }
    )

    out_dir = Path("data/interim/optiprime_compatible_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    for (cell, pe), g in out.groupby(["cell_type", "prime_editor"]):
        fname = out_dir / f"Liu_{cell}_TestFold_{pe}.csv"
        g.drop(columns=["cell_type", "prime_editor"]).to_csv(fname, index=False)
        logger.info("wrote %s (%d rows)", fname, len(g))


if __name__ == "__main__":
    main()
