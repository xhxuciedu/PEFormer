"""Task 1.2 — deployment utility metrics per target.

Spearman answers "how well ordered is the whole list". A bench scientist asks something
narrower: if I synthesise the design this model puts first, do I get an edit? These
metrics answer that, per target, against OptiPrime and against picking a design at
random.

Descriptive re-analysis of frozen predictions. Nothing is trained, tuned or selected.

Usage: python revision/task_1_2_utility.py [--seed 20260903] [--n-boot 5000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

MIN_DESIGNS = 5
THRESHOLDS = (0.05, 0.20)  # "worth taking to the bench" and "a good edit"


def ndcg_at_k(pred: np.ndarray, y: np.ndarray, k: int) -> float:
    """NDCG@k with the measured efficiency as the gain, which is the natural choice
    here: the value of a recommendation is the efficiency you actually obtain."""
    k = min(k, len(y))
    order = np.argsort(-pred, kind="stable")
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((y[order][:k] * disc).sum())
    idcg = float((np.sort(y)[::-1][:k] * disc).sum())
    return dcg / idcg if idcg > 0 else np.nan


def per_target_utility(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for g, s in df.groupby("spacer", observed=True):
        if len(s) < MIN_DESIGNS:
            continue
        y = s.y.to_numpy()
        best_measured = float(y.max())
        rec = {"spacer": g, "n": int(len(s)), "best_measured": best_measured,
               "frac_zero": float((y == 0).mean())}
        top5_true = set(np.argsort(-y, kind="stable")[:min(5, len(y))].tolist())
        for name, pred in (("ours", s.ours.to_numpy()), ("op", s.op.to_numpy())):
            order = np.argsort(-pred, kind="stable")
            top1 = int(order[0])
            rec[f"{name}_p_at_1"] = float(top1 in set(np.argsort(-y, kind="stable")[:1]))
            k5 = min(5, len(y))
            rec[f"{name}_p_at_5"] = len(set(order[:k5].tolist()) & top5_true) / k5
            rec[f"{name}_ndcg_at_5"] = ndcg_at_k(pred, y, 5)
            rec[f"{name}_top1_eff"] = float(y[top1])
            rec[f"{name}_regret"] = best_measured - float(y[top1])
            for t in THRESHOLDS:
                rec[f"{name}_top1_ge_{t}"] = float(y[top1] >= t)
        # random selection: expectation over designs, computed exactly, not simulated
        rec["rand_top1_eff"] = float(y.mean())
        rec["rand_regret"] = best_measured - float(y.mean())
        rec["rand_p_at_1"] = 1.0 / len(y)
        k5 = min(5, len(y))
        rec["rand_p_at_5"] = k5 / len(y)          # E[|random k ∩ true top-k|]/k
        rec["rand_ndcg_at_5"] = float(np.mean([
            ndcg_at_k(rng.permutation(len(y)).astype(float), y, 5) for _ in range(40)]))
        for t in THRESHOLDS:
            rec[f"rand_top1_ge_{t}"] = float((y >= t).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def boot(vals_a: np.ndarray, vals_b: np.ndarray, seed: int, n_boot: int) -> dict:
    """Bootstrap the paired difference of two per-target vectors by resampling targets.

    Clusters are protospacers and each row here is one protospacer, so resampling rows
    of this table *is* the protospacer-clustered bootstrap.

    Targets where the metric is undefined for either model are dropped, not propagated:
    NDCG has no ideal DCG at a target whose every design measures exactly zero, and there
    are such targets in the Kim partition.
    """
    d = vals_a - vals_b
    keep = np.isfinite(d)
    n_dropped = int((~keep).sum())
    d = d[keep]
    if d.size == 0:
        return {"observed": np.nan, "ci95": [np.nan, np.nan], "n_dropped": n_dropped}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    frac = float((out > 0).mean())
    return {"observed": float(d.mean()), "ci95": [float(lo), float(hi)],
            "frac_above_zero": frac, "n_targets_used": int(d.size),
            "n_dropped_undefined": n_dropped,
            "two_sided_p": float(max(2 * min(frac, 1 - frac), 1.0 / n_boot))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT_DEFAULT)
    args = ap.parse_args()

    m = C.load_heldout()
    rng = np.random.default_rng(args.seed)
    t = per_target_utility(m, rng)
    t.to_csv(C.REV / "task_1_2_utility_table.csv", index=False)

    metrics = ["p_at_1", "p_at_5", "ndcg_at_5", "top1_eff", "regret"] + \
              [f"top1_ge_{x}" for x in THRESHOLDS]
    result = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
              "n_targets": int(len(t)), "min_designs": MIN_DESIGNS,
              "rows_covered": int(t.n.sum()), "metrics": {}}
    for mt in metrics:
        row = {"ours": float(np.nanmean(t[f"ours_{mt}"])),
               "optiprime": float(np.nanmean(t[f"op_{mt}"])),
               "random": float(np.nanmean(t[f"rand_{mt}"])),
               "n_defined": int(np.isfinite(t[f"ours_{mt}"]).sum())}
        row["ours_vs_optiprime"] = boot(t[f"ours_{mt}"].to_numpy(),
                                        t[f"op_{mt}"].to_numpy(), args.seed, args.n_boot)
        row["ours_vs_random"] = boot(t[f"ours_{mt}"].to_numpy(),
                                     t[f"rand_{mt}"].to_numpy(), args.seed, args.n_boot)
        result["metrics"][mt] = row

    lines = []
    names = {"p_at_1": "precision@1", "p_at_5": "precision@5", "ndcg_at_5": "NDCG@5",
             "top1_eff": "efficiency of the top-1 pick", "regret": "regret vs best design",
             "top1_ge_0.05": "top-1 achieves >= 5%", "top1_ge_0.2": "top-1 achieves >= 20%"}
    for mt in metrics:
        r = result["metrics"][mt]
        b = r["ours_vs_optiprime"]
        lines.append(
            f"| {names.get(mt, mt)} | {r['random']:.3f} | {r['optiprime']:.3f} | "
            f"**{r['ours']:.3f}** | {b['observed']:+.3f} | "
            f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | {b['two_sided_p']:.4f} | "
            f"{b.get('n_targets_used', r['n_defined'])} |")

    md = f"""# Task 1.2 — deployment utility per target

{result['n_targets']} targets with >= {MIN_DESIGNS} designs, covering
{result['rows_covered']:,} of the 20,509 held-out rows. Bootstrap resamples targets
(= protospacer clusters), {args.n_boot} resamples, seed {args.seed}. Commit
`{result['provenance']['git_commit']}`.

Random selection is the expectation over designs at that target, computed exactly rather
than simulated, except NDCG@5 which is averaged over 40 random permutations.

| metric | random | OptiPrime | PE-RankFormer | Δ | 95% CI | p | targets |
|---|---:|---:|---:|---:|---|---:|---:|
{chr(10).join(lines)}

Regret is in units of editing efficiency: how much efficiency is lost by taking the
model's first pick instead of the target's best design. Lower is better, so a negative Δ
favours PE-RankFormer for that row.

Per-target table: `task_1_2_utility_table.csv`.
"""
    C.write_outputs("task_1_2_utility", result, md)
    for mt in metrics:
        r = result["metrics"][mt]
        b = r["ours_vs_optiprime"]
        print(f"{names.get(mt, mt):32s} rand {r['random']:.3f}  op {r['optiprime']:.3f}  "
              f"ours {r['ours']:.3f}   delta {b['observed']:+.3f} "
              f"[{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
