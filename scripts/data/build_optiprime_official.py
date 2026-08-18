"""Build the corpus from OptiPrime's official training mix, supplied by the authors.

This supersedes the public-data reconstruction in `build_full_corpus.py`. The authors
provided the 58 CSVs that constitute their actual training mix, including the
train/validation/test splits and a `weight` column derived from read depth.

Two facts from those files retire long-standing open questions in this project:

  * The 297,962 figure counts TRAINING rows only. The Liu/Hsu partition is
    65,594 train + 9,175 test = 74,769 -- so the 9,175-row "excess" we could never
    explain with any QC filter was simply their held-out test set. No filter existed.
  * The Schwank K562 LibDiverse PE4 partition (20,378 rows), absent from every public
    release and from all four cited BioProjects, is present here.

Metadata is taken from the in-file columns where present and otherwise inferred from
the filename, mirroring OptiPrime's own `scripts/pe/pe_datasets.py`. Sequence columns
are matched to our canonical schema; the authors lowercase the appended U6
transcription-start base (`gCAGG...`), which we uppercase, and this is the only
difference between their Liu designs and ours (verified: 19,908/19,908 designs agree).

Fold convention written here:  test -> 0,  split 1 -> 1 (val),  splits 2-5 -> 2..5.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_official")

SRC = Path("data/optiprime_train_mix")
OUT = Path("data/processed/optiprime_official_318471.parquet")

# Their Schwank HEK diverse-library file carries a typo in the cell type.
CELL_FIX = {"HEK239T": "HEK293T"}
LAB_TO_STUDY = {"Liu": "hsu2026", "Kim": "deepprime", "Schwank": "pridict_pridict2"}
# Time in culture, per OptiPrime's process_* functions.
TIME = {"Liu": {"HEK293T": 3.0, "HeLa": 5.0}, "Schwank": 7.0, "Kim": 7.0}


def parse_name(stem: str) -> dict:
    p = stem.split("_")
    lab, cell, library, editor = p[0], p[1], p[2], p[3]
    cell = CELL_FIX.get(cell, cell)
    return {"lab": lab, "cell_type": cell, "library": library, "editor": editor,
            "is_test_file": stem.endswith("_test")}


def infer_meta(meta: dict) -> dict:
    """Filename-derived metadata, following OptiPrime's own conventions."""
    ed = meta["editor"]
    out = {
        "pe_type": ed[:3],
        "cas9_type": "PEmax-Cas9" if "max" in ed else "PE2-Cas9",
        "cas9_pam": "SpNRCH" if "NRCH" in ed else "SpNGG",
    }
    # epegRNA libraries carry the tevoPreQ1 3' motif; '-e' marks it explicitly in Kim.
    if meta["library"] in ("LibDiverse", "LibLarge") or ed.endswith("-e"):
        out["motif"] = "tevoPreQ1"
    else:
        out["motif"] = "none"
    out["scaffold_name"] = "BlpI_F+E" if meta["lab"] == "Liu" else "SpCas9_OG"
    return out


def main() -> None:
    files = sorted(glob.glob(str(SRC / "*.csv")))
    logger.info("reading %d files from %s", len(files), SRC)

    frames = []
    for f in files:
        stem = os.path.basename(f)[:-4]
        meta = parse_name(stem)
        d = pd.read_csv(f)
        inferred = infer_meta(meta)

        out = pd.DataFrame(index=d.index)
        out["spacer"] = d.spacer.str.upper()
        out["rtt"] = d.rtt
        out["pbs"] = d.pbs
        out["full_unedited"] = d.full_unedited
        out["full_edited"] = d.full_edited
        out["edited"] = d.edited_frac
        out["indel"] = d["indel_frac"] if "indel_frac" in d.columns else np.nan
        out["weight"] = d["weight"] if "weight" in d.columns else 1.0

        # in-file metadata is authoritative; fall back to the filename
        for col, val in inferred.items():
            out[col] = d[col] if col in d.columns else val
        out["scaffold_name"] = d["scaffold_name"] if "scaffold_name" in d.columns else inferred["scaffold_name"]

        out["cell_type"] = meta["cell_type"]
        out["source_study"] = LAB_TO_STUDY[meta["lab"]]
        out["group"] = f"{meta['lab']}_{meta['cell_type']}"
        t = TIME[meta["lab"]]
        out["time"] = t[meta["cell_type"]] if isinstance(t, dict) else t
        out["rt_name"] = "PE2-RT"
        out["linker"] = d["linker"] if "linker" in d.columns else ""
        out["source_file"] = stem
        out["target_name"] = d.target_name

        sp = d.split.astype(str).str.lower()
        out["fold"] = np.where(sp == "test", 0, pd.to_numeric(sp, errors="coerce"))
        frames.append(out)

    a = pd.concat(frames, ignore_index=True)
    a["fold"] = a.fold.astype(int)
    a["unedited"] = 1.0 - (a.edited + a.indel.fillna(0.0))
    a["record_id"] = "op_" + a.index.astype(str)
    a["PEmax"] = (a.cas9_type == "PEmax-Cas9").astype(int)
    a["epegRNA"] = (a.motif == "tevoPreQ1").astype(int)
    a["MLH1dn"] = (a.pe_type == "PE4").astype(int)
    a["NRCH"] = (a.cas9_pam == "SpNRCH").astype(int)
    a["split"] = pd.NA

    n_train = (a.fold != 0).sum()
    logger.info("rows: %d total | %d training | %d held-out test", len(a), n_train, (a.fold == 0).sum())
    assert n_train == 297_962, f"training rows = {n_train}, expected 297962"
    logger.info("TRAINING ROW COUNT MATCHES OptiPrime's 297,962 EXACTLY")

    logger.info("indel coverage: %d/%d (%.1f%%)", a.indel.notna().sum(), len(a), 100 * a.indel.notna().mean())
    logger.info("fold sizes: %s", a.fold.value_counts().sort_index().to_dict())
    logger.info("by study:\n%s", a.groupby(["source_study", a.fold == 0]).size().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    a.to_parquet(OUT, index=False)
    logger.info("wrote %s (%d rows, %d cols)", OUT, len(a), len(a.columns))


if __name__ == "__main__":
    main()
