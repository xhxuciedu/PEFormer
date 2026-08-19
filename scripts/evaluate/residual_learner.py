"""Residual learner over the current ensemble (round-4 spec §11).

Trains a model on what the ensemble *misses*: r = rank(y) - yhat_E, then predicts
yhat = yhat_E + eta * rhat. Inputs are the 16 engineered Family-C features (lengths,
GC, Tm, MFE, edit geometry, RuleSet3) plus experimental context -- deliberately a
different information source from the members themselves, which see only sequence
plus categorical context.

Distinct from the stacker in §10: the stacker reweights *member predictions*, so it
can only recombine what the members already encode. This model sees engineered
features directly and can in principle add information no member has. That is the
whole point of §11.

Evaluation is nested and **protospacer-disjoint**, for the reason documented in
context_gated_ensemble.py: the round-3 dev folds are repeated subsamples with 57-60%
row overlap, so a naive fit-on-2/score-on-1 split lets the learner fit its own test
rows. eta is tuned on the fitting folds only, never on the scoring fold.

Never touches the official held-out set or the round-4 lockbox.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("residual_learner")

DEV_FOLDS = ["round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"]
CONTEXT_COLS = ["source_study", "cell_type", "pe_type"]
ETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def _rank01(x) -> np.ndarray:
    return rankdata(np.asarray(x)) / len(np.asarray(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=Path("results/round3/dev_recalibration"))
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--corpus", default="data/processed/optiprime_official_318471.parquet")
    ap.add_argument("--features", default="data/processed/family_c_features.parquet")
    ap.add_argument("--out", type=Path, default=Path("results/round4/residual_learner.json"))
    args = ap.parse_args()

    from pe_rankformer.data.family_c_features import FEATURE_COLS  # noqa: E402

    meta = pd.read_parquet(args.corpus, columns=["record_id", "spacer"] + CONTEXT_COLS)
    feats = pd.read_parquet(args.features)

    folds = {}
    for f in DEV_FOLDS:
        base = None
        for n in args.members:
            df = pd.read_parquet(args.pred_dir / f"predictions_{n}_{f}.parquet")
            keep = (df[["record_id", "true_efficiency", "predicted_efficiency"]] if base is None
                    else df[["record_id", "predicted_efficiency"]]).rename(
                columns={"predicted_efficiency": n})
            base = keep if base is None else base.merge(keep, on="record_id")
        folds[f] = base.merge(meta, on="record_id").merge(feats, on="record_id")

    ctx_template = pd.get_dummies(
        pd.concat([folds[f][CONTEXT_COLS] for f in DEV_FOLDS]).astype(str)
    ).columns

    def design(df: pd.DataFrame) -> np.ndarray:
        ctx = pd.get_dummies(df[CONTEXT_COLS].astype(str)).reindex(columns=ctx_template, fill_value=0)
        return np.column_stack([df[FEATURE_COLS].to_numpy(dtype=float), ctx.to_numpy(dtype=float)])

    base_scores, resid_scores, chosen_etas = [], [], []
    for held in DEV_FOLDS:
        test = folds[held]
        train = pd.concat([folds[f] for f in DEV_FOLDS if f != held], ignore_index=True)
        train = train.drop_duplicates("record_id")
        train = train[~train.spacer.isin(set(test.spacer))]

        ens_tr = np.mean([_rank01(train[m]) for m in args.members], axis=0)
        ens_te = np.mean([_rank01(test[m]) for m in args.members], axis=0)
        r_tr = _rank01(train.true_efficiency) - ens_tr

        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=5, random_state=0
        ).fit(design(train), r_tr)

        # eta tuned on the FITTING folds only (via their own residual fit), then applied
        # unchanged to the scoring fold.
        rhat_tr = model.predict(design(train))
        best_eta = max(ETAS, key=lambda e: spearmanr(train.true_efficiency, ens_tr + e * rhat_tr).statistic)
        chosen_etas.append(best_eta)

        rhat_te = model.predict(design(test))
        base_rho = spearmanr(test.true_efficiency, ens_te).statistic
        new_rho = spearmanr(test.true_efficiency, ens_te + best_eta * rhat_te).statistic
        base_scores.append(base_rho)
        resid_scores.append(new_rho)
        logger.info("%s: ensemble=%.4f  +residual(eta=%.2f)=%.4f  delta=%+.4f",
                    held, base_rho, best_eta, new_rho, new_rho - base_rho)

    delta = float(np.mean(resid_scores) - np.mean(base_scores))
    logger.info("")
    logger.info("ensemble mean          : %.4f", np.mean(base_scores))
    logger.info("ensemble + residual    : %.4f", np.mean(resid_scores))
    logger.info("mean delta             : %+.4f  (promotion threshold §6: +0.003)", delta)
    logger.info("positive on all folds  : %s",
                all(n > b for n, b in zip(resid_scores, base_scores)))
    logger.info("chosen etas            : %s", chosen_etas)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "members": args.members,
        "ensemble_per_fold": [float(x) for x in base_scores],
        "residual_per_fold": [float(x) for x in resid_scores],
        "mean_delta": delta,
        "chosen_etas": chosen_etas,
        "positive_all_folds": bool(all(n > b for n, b in zip(resid_scores, base_scores))),
    }, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
