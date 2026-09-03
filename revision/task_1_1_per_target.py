"""Task 1.1 — per-target and per-condition Spearman as co-primary metrics.

Pooled Spearman over the held-out set conflates two questions: "which locus is easy to
edit" and "which pegRNA is best at this locus". Only the second is the design problem a
user faces, and only the second is what a predictor is actually deployed to answer. This
computes the comparison inside each protospacer group (the design problem) and inside
each experimental condition, with protospacer-clustered intervals.

Descriptive re-analysis of frozen predictions. Nothing is trained, tuned or selected.

Usage: python revision/task_1_1_per_target.py [--seed 20260903] [--n-boot 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

MIN_DESIGNS = 5  # a target needs at least this many designs to have a ranking to score


def per_group_table(df: pd.DataFrame, key: str, min_n: int) -> pd.DataFrame:
    rows = []
    for g, s in df.groupby(key, observed=True):
        if len(s) < min_n or s.y.nunique() < 2:
            continue
        a = C.spearman(s.ours.to_numpy(), s.y.to_numpy())
        b = C.spearman(s.op.to_numpy(), s.y.to_numpy())
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        rows.append({key: g, "n": int(len(s)), "ours": a, "optiprime": b, "delta": a - b,
                     "frac_zero": float((s.y == 0).mean())})
    return pd.DataFrame(rows)


def bootstrap_over_targets(t: pd.DataFrame, seed: int, n_boot: int,
                           weighted: bool) -> dict:
    """Bootstrap the mean within-target difference by resampling TARGETS.

    The clustering unit is the protospacer and the statistic is a mean over protospacer
    groups, so a cluster resample is exactly a resample of the per-target values: a
    target is either drawn or not, and its own within-target Spearman does not depend on
    which other targets were drawn. Recomputing the per-group correlations inside the
    bootstrap loop gives the same answer at roughly 10,000x the cost.
    """
    d = t.delta.to_numpy()
    w = t.n.to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        vals[i] = np.average(d[idx], weights=w[idx]) if weighted else d[idx].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    frac = float((vals > 0).mean())
    return {"observed": float(np.average(d, weights=w) if weighted else d.mean()),
            "ci95": [float(lo), float(hi)], "bootstrap_mean": float(vals.mean()),
            "frac_above_zero": frac,
            "two_sided_p": float(max(2 * min(frac, 1 - frac), 1.0 / n_boot)),
            "n_boot": n_boot, "n_clusters": int(len(d)),
            "note": "resampling targets; clusters and groups coincide"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT_DEFAULT)
    args = ap.parse_args()

    m = C.load_heldout()
    result = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
              "min_designs_per_target": MIN_DESIGNS}

    # ---------------- per target (protospacer) ----------------
    tgt = per_group_table(m, "spacer", MIN_DESIGNS)
    q = lambda s: {k: float(np.percentile(s, p)) for k, p in
                   (("p10", 10), ("q1", 25), ("median", 50), ("q3", 75), ("p90", 90))}
    result["per_target"] = {
        "n_targets_scored": int(len(tgt)),
        "n_targets_total": int(m.spacer.nunique()),
        "rows_covered": int(tgt.n.sum()),
        "designs_per_target": {"median": float(tgt.n.median()),
                               "min": int(tgt.n.min()), "max": int(tgt.n.max())},
        "ours": {"mean": float(tgt.ours.mean()), **q(tgt.ours)},
        "optiprime": {"mean": float(tgt.optiprime.mean()), **q(tgt.optiprime)},
        "delta": {"mean": float(tgt.delta.mean()), **q(tgt.delta),
                  "frac_targets_favouring_ours": float((tgt.delta > 0).mean())},
    }
    result["per_target"]["bootstrap_mean_delta_unweighted"] = bootstrap_over_targets(
        tgt, args.seed, args.n_boot, weighted=False)
    result["per_target"]["bootstrap_mean_delta_weighted"] = bootstrap_over_targets(
        tgt, args.seed, args.n_boot, weighted=True)

    # ---------------- per condition ----------------
    cnd = per_group_table(m, "cond", 50)
    result["per_condition"] = {
        "n_conditions": int(len(cnd)),
        "ours_n_weighted": float(np.average(cnd.ours, weights=cnd.n)),
        "optiprime_n_weighted": float(np.average(cnd.optiprime, weights=cnd.n)),
        "delta_n_weighted": float(np.average(cnd.delta, weights=cnd.n)),
        "frac_conditions_favouring_ours": float((cnd.delta > 0).mean()),
        "table": cnd.to_dict(orient="records"),
    }
    # The per-condition interval genuinely needs the clusters recomputed inside the
    # loop, because targets span conditions. It is already computed at 5,000 resamples
    # by scripts/evaluate/stratified_comparison.py, so it is read from there rather than
    # recomputed at a different seed and reported as though independent.
    strat = json.loads((C.ROOT / "results/round9/stratified_comparison.json").read_text())
    result["per_condition"]["bootstrap_delta_n_weighted"] = {
        **strat["margin"]["within_condition"],
        "source": "results/round9/stratified_comparison.json"}

    # ---------------- pooled, for contrast ----------------
    result["pooled"] = {
        "ours": C.spearman(m.ours.to_numpy(), m.y.to_numpy()),
        "optiprime": C.spearman(m.op.to_numpy(), m.y.to_numpy()),
    }
    result["pooled"]["delta"] = result["pooled"]["ours"] - result["pooled"]["optiprime"]

    tgt.to_csv(C.REV / "task_1_1_per_target_table.csv", index=False)

    pt, pc, pl = result["per_target"], result["per_condition"], result["pooled"]
    bu, bw = pt["bootstrap_mean_delta_unweighted"], pt["bootstrap_mean_delta_weighted"]
    md = f"""# Task 1.1 — per-target and per-condition Spearman

Frozen-prediction re-analysis. Protospacer-clustered bootstrap, {args.n_boot} resamples,
seed {args.seed}. Commit `{result['provenance']['git_commit']}`.

## The headline number falls a long way when the locus effect is removed

| evaluation | OptiPrime | PE-RankFormer | Δρ | 95% CI |
|---|---:|---:|---:|---|
| Pooled over all {len(m):,} rows | {pl['optiprime']:.4f} | {pl['ours']:.4f} | {pl['delta']:+.4f} | see task 1.7 / Table 2 |
| Within condition, n-weighted ({pc['n_conditions']} conditions) | {pc['optiprime_n_weighted']:.4f} | {pc['ours_n_weighted']:.4f} | {pc['delta_n_weighted']:+.4f} | [{pc['bootstrap_delta_n_weighted']['ci95'][0]:+.4f}, {pc['bootstrap_delta_n_weighted']['ci95'][1]:+.4f}] |
| **Within target**, mean over {pt['n_targets_scored']} targets | **{result['per_target']['optiprime']['mean']:.4f}** | **{result['per_target']['ours']['mean']:.4f}** | **{pt['delta']['mean']:+.4f}** | [{bu['ci95'][0]:+.4f}, {bu['ci95'][1]:+.4f}] |
| Within target, n-weighted | — | — | {bw['observed']:+.4f} | [{bw['ci95'][0]:+.4f}, {bw['ci95'][1]:+.4f}] |

Targets scored: {pt['n_targets_scored']} of {pt['n_targets_total']} protospacers had
>= {MIN_DESIGNS} designs and a non-constant target, covering {pt['rows_covered']:,} rows
(median {pt['designs_per_target']['median']:.0f} designs per target,
range {pt['designs_per_target']['min']}-{pt['designs_per_target']['max']}).

## Distribution of within-target Spearman

| | p10 | q1 | median | q3 | p90 | mean |
|---|---:|---:|---:|---:|---:|---:|
| PE-RankFormer | {pt['ours']['p10']:.3f} | {pt['ours']['q1']:.3f} | {pt['ours']['median']:.3f} | {pt['ours']['q3']:.3f} | {pt['ours']['p90']:.3f} | {pt['ours']['mean']:.3f} |
| OptiPrime | {pt['optiprime']['p10']:.3f} | {pt['optiprime']['q1']:.3f} | {pt['optiprime']['median']:.3f} | {pt['optiprime']['q3']:.3f} | {pt['optiprime']['p90']:.3f} | {pt['optiprime']['mean']:.3f} |
| difference | {pt['delta']['p10']:+.3f} | {pt['delta']['q1']:+.3f} | {pt['delta']['median']:+.3f} | {pt['delta']['q3']:+.3f} | {pt['delta']['p90']:+.3f} | {pt['delta']['mean']:+.3f} |

PE-RankFormer is ahead on {pt['delta']['frac_targets_favouring_ours']:.1%} of scored
targets, and ahead in {bu['frac_above_zero']:.1%} of bootstrap resamples
(p = {bu['two_sided_p']:.4f}).

Per-target table: `task_1_1_per_target_table.csv`.
"""
    C.write_outputs("task_1_1_per_target", result, md)

    print(f"\npooled            : ours {pl['ours']:.4f}  op {pl['optiprime']:.4f}  "
          f"delta {pl['delta']:+.4f}")
    print(f"within condition  : ours {pc['ours_n_weighted']:.4f}  "
          f"op {pc['optiprime_n_weighted']:.4f}  delta {pc['delta_n_weighted']:+.4f}")
    print(f"within TARGET     : ours {pt['ours']['mean']:.4f}  "
          f"op {pt['optiprime']['mean']:.4f}  delta {pt['delta']['mean']:+.4f}  "
          f"CI [{bu['ci95'][0]:+.4f}, {bu['ci95'][1]:+.4f}]")
    print(f"                    {pt['n_targets_scored']} targets, "
          f"ours ahead on {pt['delta']['frac_targets_favouring_ours']:.1%}")


if __name__ == "__main__":
    main()
