"""Task 1.4 — the comparison on a leakage-free subset.

The manuscript's limitation 3 records that 196 of 649 replicate groups straddle the
official train/held-out split, so some held-out rows have an exact design-and-condition
twin in training. It argues this affects both models equally. That is probably right,
but the way to establish it is to remove those rows and recompute, which nobody did.

A "twin" is defined exactly as the replicate analysis defines a replicate group: all
sixteen design and condition covariates plus the target site identical
(`noise_ceiling_empirical.FULL_KEY`).

Descriptive re-analysis of frozen predictions. Nothing is trained, tuned or selected.

Usage: python revision/task_1_4_leakage_free.py [--seed 20260903] [--n-boot 5000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

FULL_KEY = ["spacer", "rtt", "pbs", "cell_type", "pe_type", "cas9_type", "cas9_pam",
            "motif", "scaffold_name", "linker", "rt_name", "time",
            "PEmax", "epegRNA", "MLH1dn", "NRCH", "target_name"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT_DEFAULT)
    args = ap.parse_args()

    corpus = pd.read_parquet(C.CORPUS, columns=["record_id", "fold"] + FULL_KEY)
    # groupby(...).ngroup() rather than a string join: `linker` is NaN for 17% of rows
    # and astype(str) does not reliably stringify a mixed object column, which silently
    # broke an earlier version of this script.
    corpus["_key"] = corpus.groupby(FULL_KEY, dropna=False).ngroup()
    train_keys = set(corpus.loc[corpus.fold != 0, "_key"].unique())
    held = corpus.loc[corpus.fold == 0, ["record_id", "_key"]].copy()
    held["has_train_twin"] = held._key.isin(train_keys)

    m = C.load_heldout().merge(held[["record_id", "has_train_twin"]],
                               on="record_id", validate="1:1")
    clean = m[~m.has_train_twin]
    dirty = m[m.has_train_twin]

    def delta(d: pd.DataFrame) -> float:
        return C.spearman(d.ours.to_numpy(), d.y.to_numpy()) - \
               C.spearman(d.op.to_numpy(), d.y.to_numpy())

    result = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
              "twin_definition": "all 16 design/condition covariates + target_name identical",
              "counts": {
                  "heldout_rows": int(len(m)),
                  "rows_with_a_training_twin": int(m.has_train_twin.sum()),
                  "frac_with_twin": float(m.has_train_twin.mean()),
                  "leakage_free_rows": int(len(clean)),
                  "leakage_free_protospacers": int(clean.spacer.nunique())},
              "subsets": {}}
    for name, sub in (("full_heldout", m), ("leakage_free", clean), ("twinned_only", dirty)):
        if len(sub) < 100:
            continue
        entry = {"n": int(len(sub)), "n_protospacers": int(sub.spacer.nunique()),
                 "frac_zero": float((sub.y == 0).mean()),
                 "ours": C.spearman(sub.ours.to_numpy(), sub.y.to_numpy()),
                 "optiprime": C.spearman(sub.op.to_numpy(), sub.y.to_numpy())}
        entry["delta"] = entry["ours"] - entry["optiprime"]
        entry["bootstrap_delta"] = C.cluster_bootstrap(sub, delta, args.seed, args.n_boot)
        entry["bootstrap_ours_absolute"] = C.cluster_bootstrap(
            sub, lambda d: C.spearman(d.ours.to_numpy(), d.y.to_numpy()),
            args.seed, args.n_boot)
        result["subsets"][name] = entry

    cn = result["counts"]
    rows = []
    for name, lab in (("full_heldout", "Full held-out set"),
                      ("leakage_free", "**Leakage-free subset**"),
                      ("twinned_only", "Twinned rows only")):
        e = result["subsets"].get(name)
        if not e:
            continue
        b, ba = e["bootstrap_delta"], e["bootstrap_ours_absolute"]
        rows.append(f"| {lab} | {e['n']:,} | {e['n_protospacers']} | {e['optiprime']:.4f} | "
                    f"{e['ours']:.4f} | {e['delta']:+.4f} | "
                    f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
                    f"[{ba['ci95'][0]:.4f}, {ba['ci95'][1]:.4f}] |")

    md = f"""# Task 1.4 — leakage-free subset

A held-out row is "twinned" if some training row matches it on all sixteen design and
condition covariates **and** the target site — the same key the replicate analysis uses.

{cn['rows_with_a_training_twin']:,} of {cn['heldout_rows']:,} held-out rows
({cn['frac_with_twin']:.1%}) have such a twin in training. Removing them leaves
{cn['leakage_free_rows']:,} rows over {cn['leakage_free_protospacers']} protospacers.

| subset | rows | protospacers | OptiPrime | PE-RankFormer | Δρ | 95% CI on Δ | 95% CI on ours |
|---|---:|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

Protospacer-clustered bootstrap, {args.n_boot} resamples, seed {args.seed}.
Commit `{result['provenance']['git_commit']}`.
"""
    C.write_outputs("task_1_4_leakage_free", result, md)
    print(f"twinned rows: {cn['rows_with_a_training_twin']}/{cn['heldout_rows']} "
          f"({cn['frac_with_twin']:.1%})")
    for name, e in result["subsets"].items():
        b = e["bootstrap_delta"]
        print(f"{name:16s} n={e['n']:6d}  op {e['optiprime']:.4f}  ours {e['ours']:.4f}  "
              f"delta {e['delta']:+.4f} CI [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
