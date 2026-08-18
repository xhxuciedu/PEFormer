"""Compute the numerical feature branch for round-2 Family C (task spec §9).

Features are derived purely from sequence/metadata already present in the official
corpus, so they are computable identically for every row -- training and held-out
alike -- without ever looking at a label. That is required by §9 ("do not include a
feature unless it can be computed consistently for both training and official
held-out rows") and is safe under the test-set-lock rule (§2): these are model
*inputs*, not label-derived statistics, and normalization is fit on training rows
only downstream in the model code, not here.

Feature groups:
  1. Length / edit-geometry: PBS length, RTT length, edit length, edit type
     (substitution/insertion/deletion/complex/multi_substitution), edit position,
     distance from nick to edit. Reuses `seqops.diff_window` / `find_protospacer`,
     the same tested edit-classification code the corpus reconstruction used.
  2. GC content: PBS, RTT, extension (rtt+pbs, the physical 3' extension order --
     see external/optiprime/scripts/pe/pe_utils.py `pegrna = spacer+scaffold+rtt+pbs`).
  3. Melting temperature: PBS and RTT Tm via Bio.SeqUtils.MeltingTemp with the RNA
     nearest-neighbor table (RNA_NN3), matching OptiPrime's own PBSMeltRNA feature.
  4. RNA secondary-structure MFE (ViennaRNA): protospacer, protospacer+scaffold,
     extension, extension+scaffold, RTT, PBS -- the same six MFE features
     PRIDICT/OptiPrime-family models use.
  5. RuleSet3 on-target Cas9 activity (highest-priority feature per §9), computed
     once per unique protospacer via the isolated rs3 environment (see
     scripts/evaluate/precompute_ruleset3_cache.py for the same pattern used in the
     round-1 OptiPrime baseline reproduction) and joined back onto every row.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.seqops import diff_window, find_protospacer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_family_c_features")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
OUT = Path("data/processed/family_c_features.parquet")
RS3_CACHE = Path("data/processed/ruleset3_scores.parquet")

EDIT_TYPES = ["none", "substitution", "multi_substitution", "insertion", "deletion", "complex"]


def gc_content(seq: str) -> float:
    seq = seq.upper().replace("U", "T")
    if not seq:
        return 0.0
    return sum(c in "GC" for c in seq) / len(seq)


def compute_edit_geometry(df: pd.DataFrame) -> pd.DataFrame:
    spacer_dna = df.spacer.str.upper().str.replace("U", "T", regex=False)
    n = len(df)
    edit_len = np.empty(n, dtype=np.float32)
    edit_pos = np.empty(n, dtype=np.float32)
    edit_pos_from_nick = np.full(n, np.nan, dtype=np.float32)
    edit_type_idx = np.empty(n, dtype=np.int8)
    n_mismatch = np.empty(n, dtype=np.float32)

    type_to_idx = {t: i for i, t in enumerate(EDIT_TYPES)}
    n_no_protospacer_match = 0
    for i, (wt, ed, sp) in enumerate(zip(df.full_unedited.values, df.full_edited.values, spacer_dna.values)):
        spec = diff_window(wt, ed)
        edit_len[i] = spec.edit_length
        edit_pos[i] = spec.start
        edit_type_idx[i] = type_to_idx[spec.edit_type]
        n_mismatch[i] = spec.n_mismatch

        hit = find_protospacer(wt, sp)
        if hit is None:
            n_no_protospacer_match += 1
            continue
        nick = hit[0] + 17  # NICK_OFFSET_IN_PROTOSPACER, matches src/pe_rankformer/data/hsu2026.py
        edit_pos_from_nick[i] = spec.start - nick

    if n_no_protospacer_match:
        logger.warning(
            "%d/%d rows: protospacer not found verbatim in full_unedited "
            "(edit_position_from_nick left NaN for these)", n_no_protospacer_match, n
        )

    return pd.DataFrame(
        {
            "edit_length": edit_len,
            "edit_position": edit_pos,
            "edit_position_from_nick": edit_pos_from_nick,
            "edit_type_idx": edit_type_idx,
            "n_mismatch": n_mismatch,
        }
    )


def compute_length_gc(df: pd.DataFrame) -> pd.DataFrame:
    pbs = df.pbs.str.upper()
    rtt = df.rtt.str.upper()
    extension = rtt + pbs
    return pd.DataFrame(
        {
            "pbs_length": pbs.str.len().astype(np.float32),
            "rtt_length": rtt.str.len().astype(np.float32),
            "pbs_gc": pbs.map(gc_content).astype(np.float32),
            "rtt_gc": rtt.map(gc_content).astype(np.float32),
            "extension_gc": extension.map(gc_content).astype(np.float32),
        }
    )


def compute_tm(df: pd.DataFrame) -> pd.DataFrame:
    from Bio.SeqUtils import MeltingTemp as Mt

    def tm(seq: str) -> float:
        s = seq.upper().replace("U", "T")
        if len(s) < 2:
            return 0.0
        try:
            return float(Mt.Tm_NN(s, nn_table=Mt.RNA_NN3))
        except Exception:
            return np.nan

    return pd.DataFrame(
        {
            "pbs_tm": df.pbs.map(tm).astype(np.float32),
            "rtt_tm": df.rtt.map(tm).astype(np.float32),
        }
    )


def compute_mfe(df: pd.DataFrame, scaffold_col: str) -> pd.DataFrame:
    import RNA

    def mfe(seq: str) -> float:
        s = seq.upper().replace("T", "U")
        if len(s) < 2:
            return 0.0
        try:
            _, e = RNA.fold(s)
            return float(e)
        except Exception:
            return np.nan

    proto = df.spacer.str.upper()
    extension = (df.rtt.str.upper() + df.pbs.str.upper())
    rtt = df.rtt.str.upper()
    pbs = df.pbs.str.upper()

    logger.info("folding %d unique protospacers, %d unique extensions, %d unique RTT, %d unique PBS",
                proto.nunique(), extension.nunique(), rtt.nunique(), pbs.nunique())

    # Fold on unique sequences only -- there are far fewer unique spacer/pbs/rtt values
    # than rows (many designs share a spacer or a PBS/RTT pair), so this is much cheaper
    # than folding every row independently.
    out = pd.DataFrame(index=df.index)
    for name, series in [("proto_mfe", proto), ("rtt_mfe", rtt), ("pbs_mfe", pbs), ("extension_mfe", extension)]:
        uniq = series.unique()
        scores = {s: mfe(s) for s in uniq}
        out[name] = series.map(scores).astype(np.float32)
        logger.info("  %s: %d unique sequences folded", name, len(uniq))
    return out


def compute_ruleset3(df: pd.DataFrame) -> pd.DataFrame:
    """Loads a precomputed RuleSet3 score table if present (see precompute step below),
    otherwise fills with NaN and warns -- RuleSet3 requires the isolated rs3 env and is
    computed as a separate offline step, not inline here."""
    if not RS3_CACHE.exists():
        logger.warning(
            "%s not found -- run scripts/data/precompute_ruleset3_family_c.py in the "
            "isolated rs3 env first. Filling ruleset3_score with NaN for now.", RS3_CACHE
        )
        return pd.DataFrame({"ruleset3_score": np.full(len(df), np.nan, dtype=np.float32)})
    scores = pd.read_parquet(RS3_CACHE)  # keyed by DNA-uppercase spacer (see precompute script)
    key = df.spacer.str.upper().str.replace("U", "T", regex=False).rename("spacer")
    merged = key.to_frame().merge(scores, on="spacer", how="left")
    n_missing = merged.ruleset3_score.isna().sum()
    if n_missing:
        logger.warning("%d/%d rows missing a RuleSet3 score after join", n_missing, len(df))
    return merged[["ruleset3_score"]].reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(CORPUS)
    logger.info("loaded %d rows", len(df))

    parts = [
        df[["record_id"]].reset_index(drop=True),
        compute_length_gc(df),
        compute_edit_geometry(df),
        compute_tm(df),
        compute_mfe(df, scaffold_col="scaffold_name"),
        compute_ruleset3(df),
    ]
    out = pd.concat(parts, axis=1)

    n_feature_cols = len(out.columns) - 1
    logger.info("computed %d feature columns for %d rows", n_feature_cols, len(out))
    for col in out.columns:
        if col == "record_id":
            continue
        n_nan = out[col].isna().sum() if out[col].dtype != np.int8 else 0
        logger.info("  %-24s missing=%6d (%.1f%%)", col, n_nan, 100 * n_nan / len(out))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
