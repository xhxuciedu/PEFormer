"""Pre-populate OptiPrime's RuleSet3Score disk cache for the reserved panel.

OptiPrime calls `rs3` at inference time for Cas9 on-target scoring. `rs3` pins
scikit-learn<=1.0.2, which will not build against the main environment's stack, so the
established procedure in `reports/baseline_reproduction_notes.md` is followed: compute the
scores in the isolated Python 3.10 environment, write them into OptiPrime's own on-disk
cache, and let inference run unmodified with the `rs3` import stubbed by a canary that
raises if the cache ever misses.

Unlike `scripts/evaluate/precompute_ruleset3_cache.py`, this reads the panel's own `proto30`
column instead of slicing `full_unedited[:30]`: 828 panel rows have a window shorter than
30 nt (the corpus has such rows too, down to 26), and slicing would silently hand `rs3` a
short sequence.

Usage: <rs3env>/bin/python explore_v2/precompute_rs3_panel.py
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

RS3_ENV = ("/srv/disk01/xhx/tmp/claude-8385/-srv-disk01-xhx-git-PEFormer/"
           "4e2fa2db-72ff-4f67-b286-f80299d50afa/scratchpad/rs3env/lib/python3.10/"
           "site-packages")
sys.path.insert(0, RS3_ENV)
from rs3.seq import predict_seq  # noqa: E402

DATA_DIR = Path("/srv/disk01/xhx/git/PEFormer/data/interim/reserved_panel_kim_large")
CACHE_ROOT = DATA_DIR / "_disk_cache" / "RuleSet3Score"


def deterministic_hash(s: str, length: int = 10) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()[:length]


def main() -> None:
    df = pd.concat([pd.read_csv(p) for p in sorted(DATA_DIR.glob("*.csv"))],
                   ignore_index=True)
    # Hash EXACTLY what OptiPrime hashes. `format_pe_df` applies `.str.replace('T','U')`
    # and nothing else, so the Kim convention's lowercase leading `g` survives into the
    # hash. Upper-casing it here (as the earlier Hsu-era script did, where the spacers were
    # already upper case) produces hashes that never match and trips the cache canary.
    spacer_rna = df["spacer"].str.replace("T", "U", regex=False)
    u = pd.DataFrame({"spacer_hash": spacer_rna.apply(deterministic_hash),
                      "proto30": df["proto30"].str.upper()}).drop_duplicates("spacer_hash")
    assert (u.proto30.str.len() == 30).all(), "proto30 must be exactly 30 nt"
    print(f"{len(df):,} rows, {len(u):,} unique spacer hashes")

    # OptiPrime forces the PAM to NGG before scoring, as its own pipeline does
    targets = []
    for p in u.proto30:
        t = list(p)
        t[25] = t[26] = "G"
        targets.append("".join(t))
    u = u.assign(score=predict_seq(targets, sequence_tracr="Chen2013"))
    print(u.score.describe().to_string())

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict[str, float]] = defaultdict(dict)
    for h, s in zip(u.spacer_hash, u.score):
        shards[h[:2]][h] = float(s)
    for h2, data in shards.items():
        dp, mp = CACHE_ROOT / f"{h2}_DATA.pkl", CACHE_ROOT / f"{h2}_META.pkl"
        ed = pickle.load(dp.open("rb")) if dp.is_file() else {}
        em = pickle.load(mp.open("rb")) if mp.is_file() else set()
        ed.update(data)
        em |= set(data)
        pickle.dump(ed, dp.open("wb"))
        pickle.dump(em, mp.open("wb"))
    print(f"wrote {len(shards)} shards to {CACHE_ROOT}")


if __name__ == "__main__":
    main()
