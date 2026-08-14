"""Precompute OptiPrime's RuleSet3Score disk cache for our test-fold spacers.

Run with the isolated rs3 environment (rs3 requires scikit-learn<=1.0.2, incompatible
with the main .venv's numpy/torch stack): see reports/baseline_reproduction_notes.md.

This lets OptiPrime's real inference code run unmodified except for stubbing the
`rs3` import (never actually called once every needed hash is pre-cached) -- see
scripts/evaluate/run_optiprime_baseline.py.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/srv/disk01/xhx/tmp/claude-8385/-srv-disk01-xhx-git-PEFormer/4e2fa2db-72ff-4f67-b286-f80299d50afa/scratchpad/rs3env/lib/python3.10/site-packages")
from rs3.seq import predict_seq  # noqa: E402

DATA_DIR = Path("/srv/disk01/xhx/git/PEFormer/data/interim/optiprime_compatible_test")
CACHE_ROOT = DATA_DIR / "_disk_cache" / "RuleSet3Score"


def deterministic_hash(s: str, length: int = 10) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()[:length]


def main() -> None:
    frames = [pd.read_csv(p) for p in sorted(DATA_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df)} rows from {len(frames)} files")

    spacer_rna = df["spacer"].str.upper().str.replace("T", "U", regex=False)
    spacer_hash = spacer_rna.apply(deterministic_hash)
    proto30 = df["full_unedited"].str.upper().str.slice(0, 30)

    unique = pd.DataFrame({"spacer_hash": spacer_hash, "proto30": proto30}).drop_duplicates("spacer_hash")
    print(f"{len(unique)} unique spacer hashes to score")
    assert (unique.proto30.str.len() == 30).all(), "proto30 must be exactly 30nt"

    targets = []
    for p in unique.proto30:
        t = list(p)
        t[25] = t[26] = "G"
        targets.append("".join(t))

    scores = predict_seq(targets, sequence_tracr="Chen2013")
    unique = unique.assign(score=scores)
    print(unique.score.describe())

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict[str, object]] = defaultdict(dict)
    for h, s in zip(unique.spacer_hash, unique.score):
        shards[h[:2]][h] = float(s)

    for hash2, data in shards.items():
        dpath = CACHE_ROOT / f"{hash2}_DATA.pkl"
        mpath = CACHE_ROOT / f"{hash2}_META.pkl"
        existing_data = pickle.load(dpath.open("rb")) if dpath.is_file() else {}
        existing_meta = pickle.load(mpath.open("rb")) if mpath.is_file() else set()
        existing_data.update(data)
        existing_meta |= set(data.keys())
        pickle.dump(existing_data, dpath.open("wb"))
        pickle.dump(existing_meta, mpath.open("wb"))

    print(f"wrote {len(shards)} shards to {CACHE_ROOT}")


if __name__ == "__main__":
    main()
