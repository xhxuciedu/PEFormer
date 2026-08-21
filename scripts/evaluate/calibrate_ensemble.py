"""Monotone calibration of the rank-averaged ensemble (round-5 spec §15).

The round-4 system rank-averages its members, which is excellent for Spearman but
destroys the output scale: its "prediction" is a normalised rank in [0,1], not an
editing efficiency. MAE and RMSE against it are meaningless, and a biologist cannot
read it as "this pegRNA edits ~12% of alleles".

This fits a monotone map g: rank -> efficiency and restores that reading.

**Why this is safe to apply to the held-out predictions.** g is monotone increasing,
so it cannot change the order of any pair, and Spearman is therefore preserved
exactly (up to tie handling). It is fitted *entirely* on out-of-fold development
predictions -- no held-out row participates in fitting. Applying it is a change of
units on an already-frozen prediction vector, not a new model selection, so it exerts
no selection pressure on the benchmark. The rank metric is reported before and after
to demonstrate this rather than assert it.

Calibrators compared (selected on dev folds only):
  - isotonic regression: nonparametric, exactly monotone, can fit the sharp mass at
    zero that efficiency data has;
  - monotone cubic (PCHIP) through binned medians: smoother, less prone to the
    piecewise-constant plateaus isotonic produces on ties.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import rankdata, spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calibrate")

DEV_FOLDS = ["round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"]
PRED = Path("results/round3/dev_recalibration")
MEMBERS = ["r4p2_ordSSM_oof", "r4p2_ssm_oof", "r4p2_ordC_oof", "r4p2_ordA_oof", "r3_familyA_oof"]


def _rank01(x) -> np.ndarray:
    x = np.asarray(x)
    return rankdata(x) / len(x)


def load_fold(fold: str) -> pd.DataFrame:
    base = None
    for n in MEMBERS:
        d = pd.read_parquet(PRED / f"predictions_{n}_{fold}.parquet")
        keep = (d[["record_id", "true_efficiency", "predicted_efficiency"]] if base is None
                else d[["record_id", "predicted_efficiency"]]).rename(
            columns={"predicted_efficiency": n})
        base = keep if base is None else base.merge(keep, on="record_id")
    base["ens_rank"] = np.mean([_rank01(base[m]) for m in MEMBERS], axis=0)
    return base


def fit_pchip(s: np.ndarray, y: np.ndarray, n_bins: int = 60) -> PchipInterpolator:
    """Monotone cubic through binned medians. Bins are quantiles of the score, so each
    carries equal support; medians resist the heavy right tail of efficiency."""
    qs = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    idx = np.clip(np.searchsorted(qs, s, side="right") - 1, 0, len(qs) - 2)
    xs, ys = [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() >= 20:
            xs.append(s[m].mean())
            ys.append(np.median(y[m]))
    xs, ys = np.array(xs), np.maximum.accumulate(np.array(ys))  # enforce monotonicity
    return PchipInterpolator(xs, ys, extrapolate=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "spearman": float(spearmanr(y, p).statistic),
        "pearson": float(pearsonr(y, p)[0]),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-pred", default="results/round4/heldout/predictions_round4_final.parquet")
    ap.add_argument("--out", type=Path, default=Path("results/round5/calibration.json"))
    ap.add_argument("--apply-heldout", action="store_true",
                    help="apply the dev-fitted calibrator to the frozen held-out predictions")
    args = ap.parse_args()

    folds = {f: load_fold(f) for f in DEV_FOLDS}
    out: dict = {"dev": {}}

    # Nested selection: fit on two folds, score on the third. Protospacer overlap is
    # not a concern here the way it was for stacking -- a 1-D monotone map has
    # essentially no capacity to memorise individual rows -- but the nesting is kept so
    # the reported dev numbers are honestly out-of-sample.
    for name in ("isotonic", "pchip"):
        rows = []
        for held in DEV_FOLDS:
            te = folds[held]
            tr = pd.concat([folds[f] for f in DEV_FOLDS if f != held]).drop_duplicates("record_id")
            s_tr, y_tr = tr.ens_rank.to_numpy(), tr.true_efficiency.to_numpy()
            s_te, y_te = te.ens_rank.to_numpy(), te.true_efficiency.to_numpy()
            if name == "isotonic":
                g = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(s_tr, y_tr)
                p = g.predict(s_te)
            else:
                g = fit_pchip(s_tr, y_tr)
                p = np.clip(g(s_te), 0.0, 1.0)
            rows.append(metrics(y_te, p))
        out["dev"][name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        raw = [metrics(folds[f].true_efficiency.to_numpy(), folds[f].ens_rank.to_numpy())
               for f in DEV_FOLDS]
        out["dev"]["uncalibrated"] = {k: float(np.mean([r[k] for r in raw])) for k in raw[0]}

    logger.info("--- dev folds (nested, mean of 3) ---")
    logger.info("%-14s %9s %9s %9s %9s", "calibrator", "spearman", "pearson", "MAE", "RMSE")
    for k, v in out["dev"].items():
        logger.info("%-14s %9.4f %9.4f %9.4f %9.4f", k, v["spearman"], v["pearson"], v["mae"], v["rmse"])

    best = max(("isotonic", "pchip"), key=lambda n: out["dev"][n]["pearson"])
    out["selected"] = best
    logger.info("selected on dev by Pearson (Spearman is invariant to a monotone map): %s", best)

    if args.apply_heldout:
        # Fit on ALL dev folds, apply once to the frozen held-out prediction vector.
        allf = pd.concat(folds.values()).drop_duplicates("record_id")
        s_all, y_all = allf.ens_rank.to_numpy(), allf.true_efficiency.to_numpy()
        g = (IsotonicRegression(out_of_bounds="clip", increasing=True).fit(s_all, y_all)
             if best == "isotonic" else fit_pchip(s_all, y_all))

        ho = pd.read_parquet(args.heldout_pred)
        s_ho = _rank01(ho.predicted_efficiency.to_numpy())
        p_ho = g.predict(s_ho) if best == "isotonic" else np.clip(g(s_ho), 0.0, 1.0)
        y_ho = ho.true_efficiency.to_numpy()

        out["heldout_uncalibrated"] = metrics(y_ho, ho.predicted_efficiency.to_numpy())
        out["heldout_calibrated"] = metrics(y_ho, p_ho)
        logger.info("--- official held-out (calibrator fitted on dev only) ---")
        for k in ("heldout_uncalibrated", "heldout_calibrated"):
            v = out[k]
            logger.info("%-22s spearman=%.4f pearson=%.4f MAE=%.4f RMSE=%.4f",
                        k.replace("heldout_", ""), v["spearman"], v["pearson"], v["mae"], v["rmse"])
        # Assert the actual property -- that g is non-decreasing -- rather than using
        # Spearman as a proxy for it. Isotonic regression is piecewise *constant*, so it
        # maps distinct scores onto identical values; those ties move Spearman by a tiny
        # amount without any pair ever being inverted. Checking the map directly
        # distinguishes "created ties" (fine, expected) from "reordered rows" (a bug).
        order = np.argsort(s_ho, kind="stable")
        assert np.all(np.diff(p_ho[order]) >= -1e-9), \
            "calibrator is not monotone: it reordered rows, which would invalidate Spearman"
        d = abs(out["heldout_calibrated"]["spearman"] - out["heldout_uncalibrated"]["spearman"])
        out["spearman_shift_from_ties"] = float(d)
        logger.info("map verified non-decreasing (no pair inverted); Spearman shifted %.2e "
                    "purely from isotonic ties", d)

        ho = ho.assign(calibrated_efficiency=p_ho)
        ho.to_parquet("results/round5/heldout_calibrated.parquet", index=False)
        logger.info("wrote results/round5/heldout_calibrated.parquet")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
