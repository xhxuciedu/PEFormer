"""Context-gated ensemble weights and nonlinear stacking (round-4 spec §9, §10).

Round 3 established that *globally* fitted ensemble weights were unstable and never
beat equal weighting. §9 asks the narrower question: do weights that depend on
**experimental context** (source, cell type, PE system, Cas9/PAM) help, where global
weights did not? §10 asks whether a nonlinear stacker over OOF predictions beats
simple rank averaging.

Both are evaluated with **nested, protospacer-disjoint** dev-fold cross-validation:
fit on two dev folds, score on the third, rotate. A gate or stacker scored on the
folds it was fit on would be guaranteed to look good, which is exactly the failure
mode that made round-3's global weight fitting misleading.

**Critical subtlety, found the hard way.** The three round-3 dev folds are *repeated
random subsamples*, not a partition, so their validation sets overlap heavily:
57-60% of each fold's rows also appear in the other two. A naive "fit on 2, score on
1" split therefore lets the stacker fit on the majority of its own test rows. The
first run of this script did exactly that and reported a spurious +0.0030 gain for
GBM stacking. Fixed here by dropping, from the fitting set, every row whose
**protospacer** appears in the scoring fold -- protospacer-level rather than row-level
because rows sharing a protospacer are strongly correlated.

Everything uses OOF dev predictions only -- no official held-out rows, and no lockbox
rows (the lockbox is disjoint from every dev validation set by construction).

Compared here (§9):
  - equal-weight rank average  (the frozen round-3 rule, the bar to beat)
  - equal-weight score average
  - source-specific fixed weights (separate weight vector for Liu and Kim)
  - context-gated weights: linear-softmax gate, and 1-hidden-layer MLP gate,
    both regularised toward equal weights by lambda * sum_k (w_k - 1/K)^2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("context_gated_ensemble")

DEV_FOLDS = ["round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"]
CONTEXT_COLS = ["source_study", "cell_type", "pe_type"]
META_COLS = CONTEXT_COLS + ["spacer"]


def _rank01(x) -> np.ndarray:
    x = np.asarray(x)
    return rankdata(x) / len(x)


def load_fold(pred_dir: Path, names: list[str], fold: str, meta: pd.DataFrame) -> pd.DataFrame:
    base = None
    for n in names:
        df = pd.read_parquet(pred_dir / f"predictions_{n}_{fold}.parquet")
        keep = (df[["record_id", "true_efficiency", "predicted_efficiency"]] if base is None
                else df[["record_id", "predicted_efficiency"]])
        keep = keep.rename(columns={"predicted_efficiency": n})
        base = keep if base is None else base.merge(keep, on="record_id")
    return base.merge(meta, on="record_id")


class Gate(nn.Module):
    """Maps one-hot experimental context to nonnegative ensemble weights summing to 1."""

    def __init__(self, n_context: int, n_members: int, hidden: int | None = None):
        super().__init__()
        self.net = (nn.Linear(n_context, n_members) if hidden is None
                    else nn.Sequential(nn.Linear(n_context, hidden), nn.ReLU(), nn.Linear(hidden, n_members)))
        # Init at ~zero logits => starts exactly at equal weights, so any departure
        # from equal weighting has to be earned by the data.
        last = self.net if hidden is None else self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(c), dim=-1)


def fit_gate(train: pd.DataFrame, members: list[str], ctx_cols: list[str],
             hidden: int | None, lam: float, epochs: int = 300, seed: int = 0) -> tuple[Gate, list]:
    torch.manual_seed(seed)
    cats = pd.get_dummies(train[ctx_cols].astype(str), columns=ctx_cols)
    columns = list(cats.columns)
    C = torch.tensor(cats.to_numpy(dtype=np.float32))
    R = torch.tensor(np.stack([_rank01(train[m]) for m in members], axis=1), dtype=torch.float32)
    y = torch.tensor(_rank01(train.true_efficiency), dtype=torch.float32)

    gate = Gate(C.shape[1], len(members), hidden)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
    K = len(members)
    for _ in range(epochs):
        w = gate(C)
        pred = (w * R).sum(dim=1)
        # Soft-rank surrogate for Spearman: MSE in rank space, which is monotone-
        # consistent with rank correlation and differentiable, unlike Spearman itself.
        loss = ((pred - y) ** 2).mean() + lam * ((w - 1.0 / K) ** 2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return gate, columns


def apply_gate(gate: Gate, columns: list, test: pd.DataFrame, members: list[str],
               ctx_cols: list[str]) -> np.ndarray:
    cats = pd.get_dummies(test[ctx_cols].astype(str), columns=ctx_cols)
    cats = cats.reindex(columns=columns, fill_value=0)  # unseen context -> all-zero -> equal weights
    C = torch.tensor(cats.to_numpy(dtype=np.float32))
    R = np.stack([_rank01(test[m]) for m in members], axis=1)
    with torch.no_grad():
        w = gate(C).numpy()
    return (w * R).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=Path("results/round3/dev_recalibration"))
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--corpus", default="data/processed/optiprime_official_318471.parquet")
    ap.add_argument("--out", type=Path, default=Path("results/round4/gating_stacking.json"))
    ap.add_argument("--lam", type=float, default=0.1)
    args = ap.parse_args()

    meta = pd.read_parquet(args.corpus, columns=["record_id"] + META_COLS)
    folds = {f: load_fold(args.pred_dir, args.members, f, meta) for f in DEV_FOLDS}
    K = len(args.members)
    logger.info("members (K=%d): %s", K, ", ".join(args.members))

    results: dict[str, list[float]] = {k: [] for k in
                                       ["equal_rank", "equal_score", "source_weights",
                                        "gate_linear", "gate_mlp", "stack_ridge", "stack_gbm"]}

    for held in DEV_FOLDS:
        test = folds[held]
        # Protospacer-disjoint fit set: the dev folds overlap heavily (57-60% of rows
        # recur across folds), so without this the stacker fits on most of its own
        # test rows. See module docstring.
        train = pd.concat([folds[f] for f in DEV_FOLDS if f != held], ignore_index=True)
        train = train.drop_duplicates("record_id")
        n_before = len(train)
        train = train[~train.spacer.isin(set(test.spacer))]
        logger.info("  fold %s: fit rows %d -> %d after protospacer-disjoint filtering (test %d)",
                    held, n_before, len(train), len(test))
        y_test = test.true_efficiency.to_numpy()

        results["equal_rank"].append(
            spearmanr(y_test, np.mean([_rank01(test[m]) for m in args.members], axis=0)).statistic)
        results["equal_score"].append(
            spearmanr(y_test, np.mean([test[m].to_numpy() for m in args.members], axis=0)).statistic)

        # --- source-specific fixed weights (grid over the simplex, fit per source) ----
        pred = np.zeros(len(test))
        for src in test.source_study.unique():
            tr_s, te_s = train[train.source_study == src], (test.source_study == src).to_numpy()
            best_w, best_rho = None, -2
            for w in np.stack(np.meshgrid(*[np.arange(0, 1.01, 0.1)] * K), -1).reshape(-1, K):
                if abs(w.sum() - 1) > 1e-9:
                    continue
                r = np.stack([_rank01(tr_s[m]) for m in args.members], axis=1) @ w
                rho = spearmanr(tr_s.true_efficiency, r).statistic
                if rho > best_rho:
                    best_rho, best_w = rho, w
            pred[te_s] = np.stack([_rank01(test[m]) for m in args.members], axis=1)[te_s] @ best_w
        results["source_weights"].append(spearmanr(y_test, pred).statistic)

        # --- context-gated weights ---------------------------------------------------
        for name, hidden in [("gate_linear", None), ("gate_mlp", 16)]:
            gate, cols = fit_gate(train, args.members, CONTEXT_COLS, hidden, args.lam)
            results[name].append(spearmanr(y_test, apply_gate(gate, cols, test, args.members, CONTEXT_COLS)).statistic)

        # --- nonlinear stacking (§10) -------------------------------------------------
        feat_cols = args.members
        Xtr = np.column_stack([np.stack([_rank01(train[m]) for m in feat_cols], axis=1),
                               pd.get_dummies(train[CONTEXT_COLS].astype(str)).to_numpy(dtype=float)])
        dummies_te = pd.get_dummies(test[CONTEXT_COLS].astype(str)).reindex(
            columns=pd.get_dummies(train[CONTEXT_COLS].astype(str)).columns, fill_value=0)
        Xte = np.column_stack([np.stack([_rank01(test[m]) for m in feat_cols], axis=1),
                               dummies_te.to_numpy(dtype=float)])
        ytr = _rank01(train.true_efficiency)

        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
        results["stack_ridge"].append(spearmanr(y_test, ridge.predict(Xte)).statistic)

        from sklearn.ensemble import HistGradientBoostingRegressor
        gbm = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                            max_depth=4, random_state=0).fit(Xtr, ytr)
        results["stack_gbm"].append(spearmanr(y_test, gbm.predict(Xte)).statistic)

        logger.info("held-out fold %s done", held)

    logger.info("")
    logger.info("%-16s %8s   %s", "method", "mean", "per-fold (nested: fit on 2, score on 1)")
    baseline = float(np.mean(results["equal_rank"]))
    for k, v in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        logger.info("%-16s %8.4f   %s   %+.4f vs equal_rank",
                    k, np.mean(v), [f"{x:.4f}" for x in v], np.mean(v) - baseline)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"members": args.members, "lambda": args.lam,
         "results": {k: [float(x) for x in v] for k, v in results.items()},
         "means": {k: float(np.mean(v)) for k, v in results.items()}}, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
