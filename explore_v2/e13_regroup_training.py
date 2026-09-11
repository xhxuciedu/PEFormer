"""E13 - the training objective's ranking term is grouped on a design artefact.

A quarter of the training objective is a pairwise ranking loss over "ranking groups", and
`src/pe_rankformer/data/dataset.py::ranking_group_key` builds those groups from the raw
`full_unedited | full_edited` string pair plus cell type and editor. E01 established that
the raw window pair is a *design* artefact: the window's extent depends on the row's RTT
length, so 32,181 target sites carry several WT windows that are one allele.

Alternative pegRNAs for one allele differ precisely in PBS/RTT geometry. So the key that is
supposed to put them in one ranking group is the key most likely to separate them, and the
ranking loss has been comparing mostly designs that happen to share an identical window --
i.e. designs of the same RTT length.

Measured on the 238,381 rows the model actually trains on:

| grouping | groups | singletons | rows in a multi-design group | within-group design pairs |
|---|---:|---:|---:|---:|
| current ranking key | 209,161 | 93.8% | 16.2% | 14,277 |
| canonical decision group | 132,066 | 68.0% | 62.3% | **368,307** |

The ranking term has had access to 14,277 of 368,307 *possible* comparisons. Measured
exposure, from E17a's replay of the real sampler, is 15,695 sampled pair instances per epoch
against 62,028 under the canonical key, and 100.0% of the current key's sampled pairs compare
two designs of identical RTT length.

This script measures that, then writes a re-grouped copy of the featurized corpus whose
`group_key` is the canonical decision group and nothing else changes. Training on it against
training on the original is therefore a one-variable experiment: same architecture, same
recipe, same seed, same code, one changed input array.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e13_regroup_training.py [--build]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import CANON_VERSION  # noqa: E402

SRC_NPZ = ROOT / "data/processed/featurized_official.npz"
OUT_NPZ = ROOT / "data/processed/featurized_official_canongroup.npz"


def group_hash(keys: pd.Series) -> np.ndarray:
    """Same 60-bit hash construction as `ranking_group_key`, over a different key."""
    return np.array([int(hashlib.sha256(k.encode()).hexdigest()[:15], 16) for k in keys],
                    dtype=np.int64)


def measure(man: pd.DataFrame) -> dict:
    corp = pd.read_parquet(C.CORPUS, columns=["record_id", "full_unedited", "full_edited",
                                              "cell_type", "pe_type"])
    d = man.merge(corp, on="record_id", validate="1:1")
    d["current_key"] = (d.full_unedited + "|" + d.full_edited + "|"
                        + d.cell_type.astype(str) + "|" + d.pe_type.astype(str))
    out = {}
    for label, key, scope in (("train_folds_2_5", "current_key", d.fold >= 2),
                              ("train_folds_2_5_canonical", "decision_group", d.fold >= 2)):
        t = d[scope]
        g = t.groupby(key, observed=True).design_key.nunique()
        sizes = t[key].map(g)
        out[label] = {
            "rows": int(len(t)), "groups": int(len(g)),
            "singleton_groups": int((g == 1).sum()),
            "singleton_fraction": float((g == 1).mean()),
            "rows_in_multi_design_groups": int(sizes.ge(2).sum()),
            "fraction_of_rows_in_multi_design_groups": float(sizes.ge(2).mean()),
            "within_group_design_pairs": int((g * (g - 1) // 2).sum()),
            "designs_per_group": {str(k): int(v) for k, v in
                                  g.value_counts().sort_index().head(6).items()},
        }
    a, b = out["train_folds_2_5"], out["train_folds_2_5_canonical"]
    out["pair_ratio_canonical_over_current"] = round(
        b["within_group_design_pairs"] / max(a["within_group_design_pairs"], 1), 1)
    out["fraction_of_available_pairs_the_current_key_sees"] = round(
        a["within_group_design_pairs"] / max(b["within_group_design_pairs"], 1), 4)
    out["note"] = (
        "These are counts of POSSIBLE within-group design pairs, not measured training "
        "exposure. The sampler takes at most 8 members per group per batch and keeps at "
        "most 4 pairs per group per batch subject to a 0.02 target gap, so the objective "
        "sees far fewer. E17a replays the real sampler: 15,695 sampled pair instances per "
        "epoch under the current key against 62,028 under the canonical key, a factor of "
        "3.95, with the share of training groups receiving any ranking update rising from "
        "3.2% to 19.1%.")
    return out


def build(man: pd.DataFrame) -> dict:
    src = np.load(SRC_NPZ, allow_pickle=True)
    rid = np.asarray(src["record_id"])
    key = man.set_index("record_id").decision_group.reindex(rid)
    assert key.notna().all(), "manifest does not cover every featurized row"
    new_group = group_hash(key)
    payload = {k: src[k] for k in src.files}
    old = payload["group_key"]
    payload["group_key"] = new_group
    np.savez_compressed(OUT_NPZ, **payload)
    return {"path": str(OUT_NPZ.relative_to(ROOT)),
            "rows": int(len(rid)),
            "distinct_group_key_before": int(len(np.unique(old))),
            "distinct_group_key_after": int(len(np.unique(new_group))),
            "arrays_copied_unchanged": [k for k in src.files if k != "group_key"],
            "size_mb": round(OUT_NPZ.stat().st_size / 1e6, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--build", action="store_true",
                    help="also write the re-grouped featurized corpus")
    args = ap.parse_args()

    man = pd.read_parquet(C.require_manifest())
    res = {"provenance": C.provenance([C.CORPUS, SRC_NPZ], args.seed),
           "canon_version": CANON_VERSION,
           "grouping_comparison": measure(man)}
    if args.build:
        res["regrouped_corpus"] = build(man)
    C.write_outputs("e13_regroup_training", res, render(res))


def render(r: dict) -> str:
    g = r["grouping_comparison"]
    a, b = g["train_folds_2_5"], g["train_folds_2_5_canonical"]
    L = ["# E13 - the ranking loss is grouped on a design artefact\n",
         "`ranking_group_key` builds the pairwise ranking loss's groups from the raw "
         "`full_unedited | full_edited` string pair plus cell type and editor. That window's "
         "extent depends on the row's RTT length, and alternative pegRNAs for one allele "
         "differ precisely in PBS/RTT geometry -- so the key meant to group them together is "
         "the one most likely to split them apart.\n",
         "> **What this table is and is not.** These are *possible* within-group design "
         "pairs. Measured exposure is a different and smaller thing, and E17a replays the "
         "real sampler to get it: 15,695 sampled pair instances per epoch under the current "
         "key against 62,028 under the canonical one (3.95x), with the share of training "
         "groups receiving any ranking update rising from 3.2% to 19.1%. The sharpest "
         "measured fact is that **100.0% of the current key's sampled pairs compare two "
         "designs of identical RTT length**, against 6.8% under the canonical key: for one "
         "allele the window's extent is set by the RTT, so an identical window pair implies "
         "an identical RTT length, and the ranking term has never once been asked to "
         "compare the geometry variation that distinguishes alternative pegRNAs.\n",
         "> **A confound in the arms below.** `corpus.group_key` is passed both to "
         "`GroupedBatchSampler` and to `sample_ranking_pairs`, so changing it changes batch "
         "composition *and* pair eligibility. E14/E15 therefore measure the combined effect, "
         "not the ranking channel alone.\n",
         f"Measured on the {a['rows']:,} rows the model trains on (folds 2-5):\n",
         "| grouping | groups | singletons | rows in a multi-design group | within-group design pairs |",
         "|---|---:|---:|---:|---:|",
         f"| current ranking key | {a['groups']:,} | {a['singleton_fraction']:.1%} | "
         f"{a['fraction_of_rows_in_multi_design_groups']:.1%} | {a['within_group_design_pairs']:,} |",
         f"| canonical decision group | {b['groups']:,} | {b['singleton_fraction']:.1%} | "
         f"{b['fraction_of_rows_in_multi_design_groups']:.1%} | "
         f"**{b['within_group_design_pairs']:,}** |",
         f"\nThe ranking term, whose loss coefficient is 0.25, has had access to "
         f"**{g['fraction_of_available_pairs_the_current_key_sees']:.1%}** of the *possible* "
         f"within-group design comparisons -- a factor of "
         f"{g['pair_ratio_canonical_over_current']} fewer. Measured exposure differs by 3.95x "
         "rather than 25.8x; both are reported because the first bounds the opportunity and "
         "the second is what the objective actually consumed.\n",
         "Designs per group, current key: " + str(a["designs_per_group"]) +
         "; canonical: " + str(b["designs_per_group"]) + "\n"]
    if "regrouped_corpus" in r:
        c = r["regrouped_corpus"]
        L.append("## The one-variable experiment\n")
        L.append(f"`{c['path']}` is a copy of the featurized corpus with `group_key` replaced "
                 f"by the canonical decision group ({c['distinct_group_key_before']:,} "
                 f"distinct values before, {c['distinct_group_key_after']:,} after) and every "
                 "other array copied unchanged. Training on it against training on the "
                 "original is a one-variable comparison: same architecture, same recipe, same "
                 "seed, same code state, one changed input array.\n")
        L.append("Both arms must be run fresh. The existing checkpoints were trained at a "
                 "different code state, so the published model is not a valid control.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
