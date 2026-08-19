"""Ensemble-diversity diagnostics for round-4 candidate screening (spec §5, §17, §18).

For each candidate M against the current ensemble E, computes the four quantities the
round-4 spec makes the basis of every promotion decision:

  S1  standalone Spearman            rho(y, yhat_M)
  S2  prediction-rank correlation    rho_rank(yhat_M, yhat_E)
  S2r residual correlation           rho(y - yhat_M, y - yhat_E)   [rank-space residuals]
  S3  incremental ensemble gain      rho(Ensemble(E, M), y) - rho(E, y)     <- main criterion

S3 is computed with the frozen combination rule (equal-weight rank average), so it
measures what adding M would actually do, not what an optimally-reweighted ensemble
could do.

Residuals are taken in rank space because the ensemble's own output is a rank average
and its members are not mutually calibrated -- a raw-efficiency residual would mostly
measure scale mismatch between members rather than which examples they get wrong.

Everything runs on matched dev folds or the lockbox. Never touches the official
held-out set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diversity_report")

DEV_FOLDS = ["round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"]


def _rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


def load_fold(pred_dir: Path, names: list[str], fold: str) -> pd.DataFrame:
    base = None
    for n in names:
        p = pred_dir / f"predictions_{n}_{fold}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)
        cols = ["record_id", "predicted_efficiency"]
        keep = df[["record_id", "source_study", "true_efficiency", "predicted_efficiency"]] \
            if base is None else df[cols]
        keep = keep.rename(columns={"predicted_efficiency": n})
        base = keep if base is None else base.merge(keep, on="record_id")
    return base


def analyse(df: pd.DataFrame, ensemble: list[str], candidate: str) -> dict:
    y = df.true_efficiency.to_numpy()
    ens_rank = np.mean([_rank01(df[m].to_numpy()) for m in ensemble], axis=0)
    cand = df[candidate].to_numpy()
    cand_rank = _rank01(cand)

    rho_E = spearmanr(y, ens_rank).statistic
    rho_M = spearmanr(y, cand).statistic

    # residuals in rank space (see module docstring)
    y_rank = _rank01(y)
    res_E = y_rank - ens_rank
    res_M = y_rank - cand_rank

    combined = np.mean([_rank01(df[m].to_numpy()) for m in ensemble] + [cand_rank], axis=0)
    rho_combined = spearmanr(y, combined).statistic

    return {
        "S1_standalone": float(rho_M),
        "S2_pred_corr_with_ensemble": float(spearmanr(cand, ens_rank).statistic),
        "S2r_residual_corr": float(spearmanr(res_M, res_E).statistic),
        "rho_ensemble": float(rho_E),
        "rho_ensemble_plus_candidate": float(rho_combined),
        "S3_ensemble_gain": float(rho_combined - rho_E),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=Path("results/round3/dev_recalibration"))
    ap.add_argument("--ensemble", nargs="+", required=True, help="current ensemble member names")
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/round4/diversity/diversity_report.json"))
    ap.add_argument("--folds", nargs="+", default=DEV_FOLDS)
    args = ap.parse_args()

    all_names = list(dict.fromkeys(args.ensemble + args.candidates))
    per_fold = {f: load_fold(args.pred_dir, all_names, f) for f in args.folds}

    logger.info("current ensemble: %s", " + ".join(args.ensemble))
    rho_E = [spearmanr(df.true_efficiency,
                       np.mean([_rank01(df[m].to_numpy()) for m in args.ensemble], axis=0)).statistic
             for df in per_fold.values()]
    logger.info("ensemble rho per fold: %s (mean %.4f)", [f"{r:.4f}" for r in rho_E], np.mean(rho_E))

    rows = []
    for cand in args.candidates:
        per = [analyse(df, args.ensemble, cand) for df in per_fold.values()]
        agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        agg["candidate"] = cand
        agg["S3_per_fold"] = [p["S3_ensemble_gain"] for p in per]
        agg["S3_positive_all_folds"] = all(p["S3_ensemble_gain"] > 0 for p in per)
        rows.append(agg)

    rows.sort(key=lambda r: -r["S3_ensemble_gain"])
    logger.info("")
    logger.info("%-26s %9s %9s %9s %11s %s", "candidate", "S1_solo", "S2_corr", "S2r_res", "S3_gain", "all_folds+")
    for r in rows:
        logger.info(
            "%-26s %9.4f %9.4f %9.4f %+11.4f %s",
            r["candidate"], r["S1_standalone"], r["S2_pred_corr_with_ensemble"],
            r["S2r_residual_corr"], r["S3_ensemble_gain"], r["S3_positive_all_folds"],
        )
    logger.info("")
    logger.info("promotion rule (§6/§18): advance if S3_gain >= +0.003 and positive on all folds,")
    logger.info("or standalone gain >= +0.005; reject if S2_corr > 0.99 and S3_gain < 0.001")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"ensemble": args.ensemble, "ensemble_rho_mean": float(np.mean(rho_E)), "candidates": rows},
        indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
