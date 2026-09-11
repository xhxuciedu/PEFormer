"""E08 - power for the plan's proposed pilot, from grouped residuals in hand.

Research plan section 8A days 8-14 ("simulate pilot power from available grouped
residuals") and section 6, which powers the definitive study from a paired, per-edit
endpoint and offers 0.08 as an illustrative paired standard deviation.

That illustration can now be replaced with a measurement. The plan's primary endpoint is
the intended efficiency achieved by the best of k candidates nominated for each edit,
compared between two methods and paired by edit. Three quantities decide the sample size:

1. the SD across edits of the paired between-method difference,
2. how much of that is measurement error, which replication shrinks,
3. how much is the edit-to-edit spread of what is achievable, which replication does not.

(1) is measured directly on the held-out surface, where two real methods and their real
per-edit outcomes exist. (2) comes from the 654 metadata-identical replicate groups. (3)
comes from the development folds at candidate depth five and above, which is where a
best-of-three endpoint is defined.

Descriptive throughout. No model is fitted and the reserved panel is not touched.

Usage: .venv/bin/python explore_v2/e08_pilot_power.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402

Z = {0.80: 0.8416, 0.90: 1.2816}
Z_ALPHA = 1.96


def n_for(sd: float, delta: float, power: float = 0.80) -> float:
    """Edits needed for a two-sided 5% paired test, ignoring clustering and dropout."""
    return float(((Z_ALPHA + Z[power]) * sd / delta) ** 2)


def best_of_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score, kind="stable")[:k]
    return float(y[order].max())


def expected_best_of_k_random(y: np.ndarray, k: int) -> float:
    """Exact expectation of the best of k designs drawn without replacement.

    P(max of a random k-subset <= y_(i)) = C(i, k) / C(n, k) over the sorted outcomes, so
    the expectation is a finite sum and needs no simulation.
    """
    from math import comb
    ys = np.sort(y)
    n = len(ys)
    if k >= n:
        return float(ys[-1])
    tot = comb(n, k)
    out = 0.0
    prev = 0.0
    for i in range(k, n + 1):
        cdf = comb(i, k) / tot
        out += ys[i - 1] * (cdf - prev)
        prev = cdf
    return float(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    res = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed)}

    # ---- (1) paired between-method SD, measured on the held-out surface -------------
    t = pd.read_csv(C.OUT / "e02_pairwise_table.csv")
    t = t[~t.tie]
    dif = (t.ours_picked - t.op_picked).to_numpy()
    res["paired_between_method"] = {
        "surface": "held-out fold 0, two-candidate decision groups",
        "n_groups": int(len(t)),
        "mean_difference": float(dif.mean()),
        "sd_difference": float(dif.std(ddof=1)),
        "sd_difference_nonzero_only": float(dif[dif != 0].std(ddof=1)),
        "frac_groups_with_zero_difference": float((dif == 0).mean()),
        "note": ("Both methods pick the same design in most groups, so the paired "
                 "difference is exactly zero there. That is a real feature of the "
                 "endpoint, not a missing value: it lowers the SD and raises power, "
                 "which is why the SD over all groups is the one to use."),
    }

    # ---- (2) measurement error in one observed endpoint ----------------------------
    man = pd.read_parquet(C.require_manifest())
    g = man.groupby("replicate_key", observed=True).agg(n=("edited", "size"),
                                                        v=("edited", "var"))
    r = g[g.n > 1]
    sigma = float(np.sqrt((r.v * (r.n - 1)).sum() / (r.n - 1).sum()))
    res["measurement_error"] = {
        "sigma_raw_efficiency": sigma,
        "sd_of_endpoint_from_noise_1_replicate": sigma,
        "sd_of_endpoint_from_noise_3_replicates": sigma / np.sqrt(3),
        "sd_of_paired_difference_from_noise_3_replicates": sigma * np.sqrt(2 / 3),
        "note": ("A paired difference between two methods at one edit involves two "
                 "different designs, so two independent measurement errors, unless the "
                 "methods nominate overlapping candidates - in which case the shared "
                 "measurement cancels and the noise term is smaller than this."),
    }

    # ---- (3) edit-to-edit spread of the achievable endpoint ------------------------
    idx = pd.read_parquet(C.CACHE / "embed_index.parquet")
    d = man.merge(idx[["record_id", "oof_pred"]], on="record_id", validate="1:1")
    d = d[d.fold >= 1]
    d = d.groupby(["decision_group", "design_key", "cond3"], observed=True,
                  as_index=False).agg(y=("edited", "mean"), p=("oof_pred", "mean"))
    nd = d.groupby("decision_group", observed=True).design_key.transform("nunique")
    deep = d[nd >= 5]
    rows = []
    for gid, s in deep.groupby("decision_group", observed=True, sort=False):
        y, p = s.y.to_numpy(), s.p.to_numpy()
        rows.append({"cond3": s.cond3.iloc[0], "n": len(y),
                     "oracle": float(y.max()),
                     "model_best_of_3": best_of_k(y, p, 3),
                     "random_best_of_3": expected_best_of_k_random(y, 3),
                     "model_best_of_1": best_of_k(y, p, 1)})
    e = pd.DataFrame(rows)
    res["endpoint_spread"] = {
        "surface": "development folds 1-5, decision groups with >=5 candidates",
        "n_groups": int(len(e)), "n_contexts": int(e.cond3.nunique()),
        "mean_oracle": float(e.oracle.mean()),
        "mean_model_best_of_3": float(e.model_best_of_3.mean()),
        "mean_random_best_of_3": float(e.random_best_of_3.mean()),
        "sd_across_edits_of_model_best_of_3": float(e.model_best_of_3.std(ddof=1)),
        "sd_across_edits_of_model_minus_random_best_of_3": float(
            (e.model_best_of_3 - e.random_best_of_3).std(ddof=1)),
        "mean_model_minus_random_best_of_3": float(
            (e.model_best_of_3 - e.random_best_of_3).mean()),
    }

    # ---- sample sizes --------------------------------------------------------------
    plan_sd = 0.08
    sds = {
        "plan_illustration": plan_sd,
        "measured_between_method_paired_sd_depth2": res["paired_between_method"]["sd_difference"],
        "measured_model_minus_random_paired_sd_depth5plus":
            res["endpoint_spread"]["sd_across_edits_of_model_minus_random_best_of_3"],
        "measured_endpoint_sd_depth5plus_unpaired":
            res["endpoint_spread"]["sd_across_edits_of_model_best_of_3"],
    }
    table = []
    for name, sd in sds.items():
        for delta in (0.02, 0.04, 0.08):
            table.append({"sd_source": name, "sd": sd, "target_effect": delta,
                          "edits_for_80pct_power": n_for(sd, delta, 0.80),
                          "edits_for_90pct_power": n_for(sd, delta, 0.90)})
    res["sample_size"] = table
    res["caveats"] = [
        "These are the simplest paired two-sided calculations. Clustering by locus family, "
        "donor or plate, more than one primary comparison, and attrition all raise the "
        "requirement.",
        "The between-method SD is measured where both methods choose from only two "
        "candidates. A twelve-candidate panel gives both methods more room to differ, so "
        "its paired SD will be larger than the depth-2 estimate and closer to the "
        "depth-5 figure.",
        "The number of wells is not the denominator. Replicates reduce the measurement "
        "term only; the edit-to-edit spread is irreducible by replication.",
        "Nothing here powers a mechanistic reversal endpoint. E03 puts the supported "
        "reversal rate at 4.2%, so an experiment whose primary endpoint is reversal "
        "direction needs its own calculation on that base rate.",
    ]
    C.write_outputs("e08_pilot_power", res, render(res))


def render(r: dict) -> str:
    p, m, e = r["paired_between_method"], r["measurement_error"], r["endpoint_spread"]
    L = ["# E08 - power for the proposed pilot, from measured variability\n",
         "## 1. The paired between-method difference, measured\n",
         f"On {p['n_groups']:,} scorable two-candidate decision groups of the held-out "
         f"surface, the per-edit difference in achieved efficiency between PE-RankFormer "
         f"and OptiPrime has mean {p['mean_difference']:+.4f} and "
         f"**SD {p['sd_difference']:.4f}**. The two methods nominate the same design in "
         f"{p['frac_groups_with_zero_difference']:.0%} of groups, where the difference is "
         f"exactly zero; among the rest the SD is {p['sd_difference_nonzero_only']:.4f}.\n",
         p["note"] + "\n",
         "## 2. What replication buys\n",
         f"- per-measurement SD from {'654'} metadata-identical replicate groups: "
         f"{m['sigma_raw_efficiency']:.4f} efficiency units\n"
         f"- with three biological replicates per design: "
         f"{m['sd_of_endpoint_from_noise_3_replicates']:.4f}\n"
         f"- the noise part of a paired difference at three replicates: "
         f"{m['sd_of_paired_difference_from_noise_3_replicates']:.4f}\n",
         m["note"] + "\n",
         "## 3. Edit-to-edit spread at real candidate depth\n",
         f"{e['n_groups']:,} development decision groups with five or more candidates, over "
         f"{e['n_contexts']} contexts. Best-of-three achieved efficiency: model "
         f"{e['mean_model_best_of_3']:.4f}, random {e['mean_random_best_of_3']:.4f}, oracle "
         f"{e['mean_oracle']:.4f}. The model-minus-random paired difference averages "
         f"{e['mean_model_minus_random_best_of_3']:+.4f} with SD "
         f"**{e['sd_across_edits_of_model_minus_random_best_of_3']:.4f}**; the endpoint "
         f"itself has SD {e['sd_across_edits_of_model_best_of_3']:.4f} across edits.\n",
         "## 4. Edits required\n",
         "| paired SD source | SD | target effect | edits at 80% | edits at 90% |\n|---|---:|---:|---:|---:|"]
    for row in r["sample_size"]:
        L.append(f"| {row['sd_source']} | {row['sd']:.4f} | {row['target_effect']:.2f} | "
                 f"{row['edits_for_80pct_power']:.0f} | {row['edits_for_90pct_power']:.0f} |")
    L.append("\n## Caveats\n")
    for c in r["caveats"]:
        L.append(f"- {c}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
