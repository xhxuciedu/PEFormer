"""Task 1.8 — where does the training-weight asymmetry cost performance?

Round 9 found that OptiPrime's released code consumes the corpus `weight` column as a
per-row loss weight, and that every Kim row carries weight 0.1, giving the Kim study
2.0% of its training gradient against 55.3% of the held-out rows. Training
PE-RankFormer under those same weights costs 0.0148 Spearman on development fold 0.

If that mechanism is right, the loss must sit almost entirely in **Kim**, and Liu should
be roughly unaffected or slightly better (Liu's effective share rises from 22.0% to
24.8% under OptiPrime's weights). If the loss is spread evenly, the weighting story is
wrong and the drop is something else.

This scores the two matched runs -- `r9_opw` (OptiPrime's row weights) and `r9_ctrl`
(uniform, same seed, same fold, same recipe) -- on the development fold, by source.

Usage: python revision/task_1_8_weighting_by_partition.py [--seed 20260903]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PRED = "results/round9/devpreds/predictions_{}_round3_dev_fold_0.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT_DEFAULT)
    args = ap.parse_args()

    opw = pd.read_parquet(C.ROOT / PRED.format("r9_opw"))
    ctl = pd.read_parquet(C.ROOT / PRED.format("r9_ctrl"))
    m = ctl.rename(columns={"predicted_efficiency": "uniform"}).merge(
        opw[["record_id", "predicted_efficiency"]].rename(
            columns={"predicted_efficiency": "opweights"}),
        on="record_id", validate="1:1")
    spacer = pd.read_parquet(C.CORPUS, columns=["record_id", "spacer"])
    m = m.merge(spacer, on="record_id", validate="1:1")
    m["cond"] = m.source_study + "|" + m.cell_type + "|" + m.pe_type

    def delta(d: pd.DataFrame) -> float:
        """uniform minus OptiPrime-weighted: positive means uniform weighting is better."""
        return C.spearman(d.uniform.to_numpy(), d.true_efficiency.to_numpy()) - \
               C.spearman(d.opweights.to_numpy(), d.true_efficiency.to_numpy())

    result = {"provenance": C.provenance([C.CORPUS], args.seed),
              "runs": {"uniform": "r9_ctrl", "optiprime_weights": "r9_opw",
                       "matched_on": "seed, dev fold 0, recipe, code state"},
              "n_rows": int(len(m)), "partitions": {}}

    parts = {"all": m}
    for s in sorted(m.source_study.unique()):
        parts[s] = m[m.source_study == s]
    for name, sub in parts.items():
        if len(sub) < 100:
            continue
        e = {"n": int(len(sub)), "frac_zero": float((sub.true_efficiency == 0).mean()),
             "uniform": C.spearman(sub.uniform.to_numpy(), sub.true_efficiency.to_numpy()),
             "optiprime_weights": C.spearman(sub.opweights.to_numpy(),
                                             sub.true_efficiency.to_numpy())}
        e["delta"] = e["uniform"] - e["optiprime_weights"]
        e["bootstrap"] = C.cluster_bootstrap(sub, delta, args.seed, args.n_boot)
        result["partitions"][name] = e

    # per condition, to check the loss is not driven by one cell line
    cond = []
    for c, s in m.groupby("cond"):
        if len(s) < 200:
            continue
        cond.append({"cond": c, "n": int(len(s)),
                     "uniform": C.spearman(s.uniform.to_numpy(), s.true_efficiency.to_numpy()),
                     "optiprime_weights": C.spearman(s.opweights.to_numpy(),
                                                     s.true_efficiency.to_numpy())})
        cond[-1]["delta"] = cond[-1]["uniform"] - cond[-1]["optiprime_weights"]
    cond.sort(key=lambda r: -r["delta"])
    result["per_condition"] = cond

    prows = [f"| {n} | {v['n']:,} | {v['frac_zero']:.1%} | {v['optiprime_weights']:.4f} | "
             f"{v['uniform']:.4f} | {v['delta']:+.4f} | "
             f"[{v['bootstrap']['ci95'][0]:+.4f}, {v['bootstrap']['ci95'][1]:+.4f}] |"
             for n, v in result["partitions"].items()]
    crows = [f"| {r['cond']} | {r['n']:,} | {r['optiprime_weights']:.4f} | "
             f"{r['uniform']:.4f} | {r['delta']:+.4f} |" for r in cond]

    md = f"""# Task 1.8 — where the training-weight asymmetry costs performance

Two matched runs on development fold 0: `r9_ctrl` (uniform weights) and `r9_opw`
(the corpus `weight` column, i.e. OptiPrime's own per-row loss weights). Same seed, same
fold, same recipe, same code state, so the only difference is the weighting.
Protospacer-clustered bootstrap, {args.n_boot} resamples, seed {args.seed}.
Commit `{result['provenance']['git_commit']}`.

Positive Δ means uniform weighting is better.

| partition | n | zero-mass | OptiPrime weights | uniform | Δρ | 95% CI |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(prows)}

## By condition

| condition | n | OptiPrime weights | uniform | Δρ |
|---|---:|---:|---:|---:|
{chr(10).join(crows)}
"""
    C.write_outputs("task_1_8_weighting_by_partition", result, md)
    for n, v in result["partitions"].items():
        b = v["bootstrap"]
        print(f"{n:20s} n={v['n']:6d}  opw {v['optiprime_weights']:.4f}  "
              f"uniform {v['uniform']:.4f}  delta {v['delta']:+.4f} "
              f"CI [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
