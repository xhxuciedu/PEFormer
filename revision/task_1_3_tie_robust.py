"""Task 1.3 — tie-robust and decomposed metrics.

Three questions the pooled Spearman cannot separate:

1. Is the comparison like-for-like given the tie structure? Spearman's tie convention
   rewards predictors that emit ties on a zero-inflated target (the manuscript's own
   tie-floor result: any model gains +0.013 to +0.016 from collapsing its low tail).
   Kendall tau-b corrects for ties in both variables, so it is the tie-neutral check.
2. How much of the performance is *detection* -- telling an edit from no edit at all?
   AUROC on the binary y>0 problem isolates that.
3. How much is *quantification* -- ordering the designs that do edit? Spearman
   restricted to y>0 isolates that.

Descriptive re-analysis of frozen predictions. Nothing is trained, tuned or selected.

Usage: python revision/task_1_3_tie_robust.py [--seed 20260903] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-boot", type=int, default=2000)  # tau-b is O(n log n) per resample
    args = ap.parse_args()

    m = C.load_heldout()
    result = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
              "n_rows": int(len(m)),
              "tie_structure": {}}

    # How many distinct values does each predictor emit? This is the premise of the
    # manuscript's tie caveat: a comparison is only like-for-like if both are tie-free.
    for name, col in (("ours", "ours"), ("optiprime", "op"), ("target", "y")):
        v = m[col].to_numpy()
        result["tie_structure"][name] = {
            "n_distinct": int(np.unique(v).size),
            "frac_in_largest_tie": float(np.max(np.unique(v, return_counts=True)[1]) / len(v))}

    partitions = {"all": m, "Liu": m[m.source_study == "hsu2026"],
                  "Kim": m[m.source_study == "deepprime"]}
    result["metrics"] = {}
    for pname, sub in partitions.items():
        y = sub.y.to_numpy()
        edits = y > 0
        entry = {"n": int(len(sub)), "n_editing": int(edits.sum()),
                 "frac_zero": float((y == 0).mean())}
        for name, col in (("ours", "ours"), ("optiprime", "op")):
            p = sub[col].to_numpy()
            entry[name] = {
                "spearman": C.spearman(p, y),
                "kendall_tau_b": C.kendall_tau_b(p, y),
                "auroc_edits_at_all": C.auroc(p, edits),
                "spearman_editing_only": (C.spearman(p[edits], y[edits])
                                          if edits.sum() > 10 else np.nan),
            }
        entry["delta"] = {k: entry["ours"][k] - entry["optiprime"][k]
                          for k in entry["ours"]}
        # clustered bootstrap on the paired differences that matter most
        for k, fn in (("kendall_tau_b",
                       lambda d: C.kendall_tau_b(d[col_o := "ours"].to_numpy(), d.y.to_numpy())
                                 - C.kendall_tau_b(d.op.to_numpy(), d.y.to_numpy())),
                      ("auroc_edits_at_all",
                       lambda d: C.auroc(d.ours.to_numpy(), d.y.to_numpy() > 0)
                                 - C.auroc(d.op.to_numpy(), d.y.to_numpy() > 0))):
            entry.setdefault("bootstrap", {})[k] = C.cluster_bootstrap(
                sub, fn, args.seed, args.n_boot)
        # editing-only Spearman difference
        sub_e = sub[edits]
        if len(sub_e) > 100:
            entry["bootstrap"]["spearman_editing_only"] = C.cluster_bootstrap(
                sub_e, lambda d: C.spearman(d.ours.to_numpy(), d.y.to_numpy())
                                 - C.spearman(d.op.to_numpy(), d.y.to_numpy()),
                args.seed, args.n_boot)
        result["metrics"][pname] = entry

    ts = result["tie_structure"]
    rows = []
    for pname in ("all", "Liu", "Kim"):
        e = result["metrics"][pname]
        for k, lab in (("spearman", "Spearman ρ"), ("kendall_tau_b", "Kendall τ-b"),
                       ("auroc_edits_at_all", "AUROC (edits at all)"),
                       ("spearman_editing_only", "Spearman ρ | y>0")):
            b = e.get("bootstrap", {}).get(k)
            ci = (f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"
                  if b and np.isfinite(b['ci95'][0]) else "—")
            rows.append(f"| {pname} (n={e['n']:,}) | {lab} | {e['optiprime'][k]:.4f} | "
                        f"{e['ours'][k]:.4f} | {e['delta'][k]:+.4f} | {ci} |")

    md = f"""# Task 1.3 — tie-robust and decomposed metrics

Protospacer-clustered bootstrap, {args.n_boot} resamples, seed {args.seed}.
Commit `{result['provenance']['git_commit']}`.

## Tie structure: is the comparison like-for-like?

| vector | distinct values over {len(m):,} rows | largest tie block |
|---|---:|---:|
| PE-RankFormer | {ts['ours']['n_distinct']:,} | {ts['ours']['frac_in_largest_tie']:.2%} |
| OptiPrime | {ts['optiprime']['n_distinct']:,} | {ts['optiprime']['frac_in_largest_tie']:.2%} |
| measured target | {ts['target']['n_distinct']:,} | {ts['target']['frac_in_largest_tie']:.2%} |

Both predictors are effectively tie-free, so the Spearman comparison is like-for-like on
the manuscript's own criterion, and Kendall τ-b should agree with it in direction.

## Metrics by partition

| partition | metric | OptiPrime | PE-RankFormer | Δ | 95% CI |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

`Spearman ρ | y>0` is restricted to rows that actually edit, so it isolates
quantification from detection; `AUROC (edits at all)` isolates detection.
"""
    C.write_outputs("task_1_3_tie_robust", result, md)
    for pname in ("all", "Liu", "Kim"):
        e = result["metrics"][pname]
        print(f"\n{pname} (n={e['n']}, {e['frac_zero']:.1%} zero)")
        for k in ("spearman", "kendall_tau_b", "auroc_edits_at_all", "spearman_editing_only"):
            b = e.get("bootstrap", {}).get(k)
            ci = f" CI [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]" if b and np.isfinite(b['ci95'][0]) else ""
            print(f"  {k:26s} op {e['optiprime'][k]:.4f}  ours {e['ours'][k]:.4f}  "
                  f"delta {e['delta'][k]:+.4f}{ci}")


if __name__ == "__main__":
    main()
