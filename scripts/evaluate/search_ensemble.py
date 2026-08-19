"""Search ensemble compositions/weights on matched-dev OOF predictions (round-3 §25).

Consumes the per-fold prediction parquets written by evaluate_on_devfolds.py and
evaluates candidate ensembles. Never touches the official held-out set -- every
number here comes from Liu+Kim-matched dev-fold validation rows only.

Combination rules (§25):
  - global mean
  - rank average (round-3 finding: consistently beats mean; see research log)
  - nonnegative OOF-optimised weights, simplex-constrained
  - source-specific weights (separate weight vectors for Liu and Kim)

Selection rule for this round: a candidate must improve on **every** dev fold,
not just on the mean. The Stage-0 diagnosis showed fold-to-fold sign disagreement
at the ~0.003 scale, so a mean-only improvement is not evidence of anything.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("search_ensemble")

DEV_FOLDS = ["round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"]


def load_member(pred_dir: Path, model_name: str, fold: str) -> pd.DataFrame:
    p = pred_dir / f"predictions_{model_name}_{fold}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def assemble(pred_dir: Path, model_names: list[str], fold: str) -> pd.DataFrame:
    base = load_member(pred_dir, model_names[0], fold)[
        ["record_id", "source_study", "true_efficiency", "predicted_efficiency"]
    ].rename(columns={"predicted_efficiency": model_names[0]})
    for m in model_names[1:]:
        other = load_member(pred_dir, m, fold)[["record_id", "predicted_efficiency"]].rename(
            columns={"predicted_efficiency": m}
        )
        base = base.merge(other, on="record_id")
    return base


def _rank_norm(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


def combine(df: pd.DataFrame, members: list[str], rule: str, weights: np.ndarray | None = None) -> np.ndarray:
    mat = df[members].to_numpy()
    if rule == "mean":
        return mat.mean(axis=1)
    if rule == "rank":
        return np.mean([_rank_norm(mat[:, i]) for i in range(mat.shape[1])], axis=0)
    if rule == "weighted_rank":
        r = np.stack([_rank_norm(mat[:, i]) for i in range(mat.shape[1])], axis=1)
        return r @ weights
    raise ValueError(rule)


def fit_weights(df: pd.DataFrame, members: list[str]) -> np.ndarray:
    """Nonnegative, sum-to-1 weights maximising Spearman on this dataframe.

    Optimises rank-space weights (the rank rule dominates mean on this data), via
    a softmax reparameterisation so the simplex constraint is automatic.
    """
    y = df.true_efficiency.to_numpy()
    r = np.stack([_rank_norm(df[m].to_numpy()) for m in members], axis=1)

    def neg_rho(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        return -spearmanr(y, r @ w).statistic

    best = None
    for seed in range(3):  # a few restarts; the objective is not convex
        z0 = np.zeros(len(members)) if seed == 0 else np.random.default_rng(seed).normal(0, 0.5, len(members))
        res = minimize(neg_rho, z0, method="Nelder-Mead",
                       options={"maxiter": 400 * len(members), "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    w = np.exp(best.x - best.x.max())
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=Path("results/round3/dev_recalibration"))
    ap.add_argument("--members", nargs="+", required=True,
                    help="model-name prefixes as passed to evaluate_on_devfolds.py --model-name")
    ap.add_argument("--out", type=Path, default=Path("results/round3/ensemble_search.json"))
    ap.add_argument("--max-subset-size", type=int, default=None)
    args = ap.parse_args()

    per_fold = {f: assemble(args.pred_dir, args.members, f) for f in DEV_FOLDS}
    logger.info("loaded %d members over %d dev folds", len(args.members), len(DEV_FOLDS))

    # ---- individual members -------------------------------------------------------
    logger.info("--- individual members (mean over folds) ---")
    singles = {}
    for m in args.members:
        rhos = [spearmanr(df.true_efficiency, df[m]).statistic for df in per_fold.values()]
        singles[m] = float(np.mean(rhos))
        logger.info("  %-28s %.4f   per-fold %s", m, singles[m], [f"{r:.4f}" for r in rhos])
    best_single = max(singles.values())

    # ---- all subsets, mean vs rank ------------------------------------------------
    results = []
    max_k = args.max_subset_size or len(args.members)
    for k in range(2, max_k + 1):
        for subset in itertools.combinations(args.members, k):
            for rule in ("mean", "rank"):
                rhos = [
                    spearmanr(df.true_efficiency, combine(df, list(subset), rule)).statistic
                    for df in per_fold.values()
                ]
                results.append(
                    {
                        "members": list(subset), "rule": rule,
                        "per_fold": [float(r) for r in rhos], "mean": float(np.mean(rhos)),
                        # the round-3 selection rule: must beat the best single on EVERY fold
                        "beats_best_single_all_folds": bool(
                            all(r > max(spearmanr(df.true_efficiency, df[m]).statistic for m in subset)
                                for r, df in zip(rhos, per_fold.values()))
                        ),
                    }
                )

    results.sort(key=lambda d: -d["mean"])
    logger.info("--- top 10 subsets/rules ---")
    for r in results[:10]:
        logger.info(
            "  %.4f %s %-6s all-folds-win=%s  %s",
            r["mean"], [f"{x:.4f}" for x in r["per_fold"]], r["rule"],
            r["beats_best_single_all_folds"], "+".join(r["members"]),
        )

    # ---- fitted weights on the full member set ------------------------------------
    # Fit on each fold, evaluate on the other two, to avoid reporting an in-sample
    # weight-fitting number as if it were a validation result.
    if len(args.members) >= 2:
        logger.info("--- fitted rank weights (fit on one fold, evaluated on the other two) ---")
        held_scores = []
        for fit_fold in DEV_FOLDS:
            w = fit_weights(per_fold[fit_fold], args.members)
            others = [f for f in DEV_FOLDS if f != fit_fold]
            rhos = [
                spearmanr(per_fold[f].true_efficiency,
                          combine(per_fold[f], args.members, "weighted_rank", w)).statistic
                for f in others
            ]
            held_scores.extend(rhos)
            logger.info("  fit on %s -> w=%s | eval %s = %s",
                        fit_fold, np.round(w, 3).tolist(), others, [f"{r:.4f}" for r in rhos])
        logger.info("  mean held-fold score with fitted weights: %.4f", float(np.mean(held_scores)))
        logger.info("  (compare: best equal-weight rank subset above)")

    summary = {
        "members": args.members,
        "singles": singles,
        "best_single_mean": best_single,
        "top_results": results[:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
