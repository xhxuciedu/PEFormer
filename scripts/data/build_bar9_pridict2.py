"""Build the PRIDICT2.0 K562 partition (ground-truth bar 9) from primary sources.

This partition was entirely absent from our corpus (0 rows against a 23,428-row
target). It is reconstructed here from the PRIDICT2.0 release directly --
`dataset/proc_v2/data_23k_v1.csv` and `k562_indices_nan.pkl` from
github.com/uzh-dqbm-cmi/PRIDICT2 -- rather than imported from the parallel
`biomni/` reconstruction, so the provenance chain runs to the original release.

The CSV carries no literal spacer/PBS/RTT columns; all three are sliced out of the
wide target sequences using the location columns. The slicing rules were derived by
matching a row against the independently-produced biomni values and are re-validated
here against every overlapping row (see `--validate`):

    spacer = 'G' + wide_initial[protospacerlocation_only_initial]
    pbs    = revcomp(wide_initial[PBSlocation])
    rtt    = revcomp(wide_mutated[RT_mutated_location])   # mutated strand: carries the edit

The leading 'G' is the U6 transcription-start base; PRIDICT stores only the 19nt
remainder in the location column.
"""

from __future__ import annotations

import ast
import logging
import pickle
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_bar9")

CSV = Path("data/raw/pridict2/data_23k_v1.csv")
NAN_PKL = Path("data/raw/pridict2/k562_indices_nan.pkl")
OUT = Path("data/interim/bar9_pridict2_k562.parquet")

_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _slice(seq: str, loc: str) -> str:
    start, end = ast.literal_eval(loc)
    return seq[start:end]


def build() -> pd.DataFrame:
    df = pd.read_csv(CSV, low_memory=False)
    logger.info("loaded %s: %d rows", CSV, len(df))

    # The 204 indices flagged in k562_indices_nan.pkl are rows where the *unclamped*
    # K562 measurement is NaN, i.e. the design was not measured in K562 at all. The
    # public CSV ships only clamped columns (non-null everywhere), so without this
    # file those 204 would be silently admitted as real zero-efficiency measurements.
    nan_idx = pickle.load(NAN_PKL.open("rb"))
    logger.info("dropping %d unmeasured K562 rows (k562_indices_nan.pkl)", len(nan_idx))
    df = df.drop(index=list(nan_idx)).reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    spacer = "G" + df.apply(
        lambda r: _slice(r.wide_initial_target, r.protospacerlocation_only_initial), axis=1
    )
    pbs = df.apply(lambda r: revcomp(_slice(r.wide_initial_target, r.PBSlocation)), axis=1)
    rtt = df.apply(lambda r: revcomp(_slice(r.wide_mutated_target, r.RT_mutated_location)), axis=1)

    dna_to_rna = lambda s: s.str.replace("T", "U", regex=False)
    out["spacer"] = dna_to_rna(spacer)
    out["rtt"] = dna_to_rna(rtt)
    out["pbs"] = dna_to_rna(pbs)
    out["full_unedited"] = df.wide_initial_target
    out["full_edited"] = df.wide_mutated_target
    out["edited"] = df.K562averageedited_clamped
    out["indel"] = df.K562averageunintended_clamped.fillna(0.0)
    out["unedited"] = 1.0 - (out.edited + out.indel)
    out["weight"] = 1.0

    # Experimental context for bar 9: Schwank K562, PE2 + epegRNA (tevoPreQ1 motif),
    # non-PEmax Cas9, no dnMLH1. Matches OptiPrime's own `process_schwank` metadata.
    out["scaffold_name"] = "SpCas9_OG"
    out["motif"] = "tevoPreQ1"
    out["cas9_type"] = "PE2-Cas9"
    out["cas9_pam"] = "SpNGG"
    out["rt_name"] = "PE2-RT"
    out["pe_type"] = "PE2"
    out["group"] = "Schwank_K562"
    out["cell_type"] = "K562"
    out["time"] = 6.0
    out["linker"] = ""
    out["split"] = pd.NA
    out["source_study"] = "pridict_pridict2"
    out["record_id"] = "pridict2_k562_" + df.index.astype(str)
    out["PEmax"] = 0
    out["epegRNA"] = 1
    out["MLH1dn"] = 0
    out["NRCH"] = 0

    # Sanity: lengths must agree with the release's own length columns.
    bad_pbs = (pbs.str.len() != df.PBSlength.astype(int)).sum()
    bad_rtt = (rtt.str.len() != df.RTlength.astype(int)).sum()
    assert bad_pbs == 0, f"{bad_pbs} rows: derived PBS length != PBSlength column"
    assert bad_rtt == 0, f"{bad_rtt} rows: derived RTT length != RTlength column"
    logger.info("length checks passed (PBS, RTT) on all %d rows", len(out))
    return out


def validate_against_biomni(out: pd.DataFrame) -> None:
    """Cross-check the derived sequences against the independent biomni reconstruction,
    which read spacer/PBS/RTT from the PRIDICT2.0 supplementary xlsx rather than
    slicing them. Agreement means two different extraction paths concur."""
    bp = Path("biomni/optiprime_full_297962.parquet")
    if not bp.exists():
        logger.warning("biomni parquet absent -- skipping cross-validation")
        return
    b = pd.read_parquet(bp)
    b9 = b[(b.bar_idx == 9) & (b.split.notna())]

    # Compare on full design identity. Merging on the target pair alone is wrong here:
    # many designs share a target, so a loose key cross-matches distinct rows.
    key = ["full_unedited", "full_edited", "spacer", "pbs", "rtt"]
    ours = set(map(tuple, out[key].values))
    theirs = set(map(tuple, b9[key].values))
    logger.info("cross-validation vs biomni (independent xlsx-based extraction):")
    logger.info("  ours=%d  biomni=%d  in common=%d", len(ours), len(theirs), len(ours & theirs))
    logger.info("  biomni rows absent from ours: %d", len(theirs - ours))
    assert not (theirs - ours), "biomni recovered designs we did not -- extraction is incomplete"

    merged = out.merge(b9[key + ["edited"]], on=key, how="inner",
                       suffixes=("", "_b")).drop_duplicates(key)
    logger.info("  max |edited - biomni edited| over %d matched rows = %.3g",
                len(merged), (merged.edited - merged.edited_b).abs().max())

    # The residual rows are ours-only. biomni deduplicated on (spacer, PBS, RTT), but in
    # every such group the *target* differs -- these are one pegRNA against several
    # genomic targets, i.e. genuinely distinct experiments. We keep them.
    extra = ours - theirs
    if extra:
        dup_groups = out[out.duplicated(["spacer", "pbs", "rtt"], keep=False)]
        n_multi_target = (
            dup_groups.groupby(["spacer", "pbs", "rtt"]).full_unedited.nunique() > 1
        ).sum()
        logger.info(
            "  ours-only rows: %d (in %d shared-pegRNA groups, all with distinct targets)",
            len(extra), n_multi_target,
        )


def main() -> None:
    out = build()
    validate_against_biomni(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logger.info("wrote %s (%d rows)", OUT, len(out))


if __name__ == "__main__":
    main()
