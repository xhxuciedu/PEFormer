"""E07 - does an explicit context-conditioned design correction improve the choice?

E04 established that the design-by-context interaction is predictable at held-out loci
from the frozen representation and the context identity (R^2 = +0.40 against +0.01 for a
structured shuffle). E05 established that it cannot be recovered from 12-192 reference
measurements in a context the model has never seen. Those two results are compatible and
they point somewhere the research plan does not: the interaction is learnable for contexts
that *are* in the training corpus, and the binding constraint is not the context labels.

This script tests the consequence directly, on the decision rather than on D. Held-out
loci, contexts known:

    score(d, c) = frozen_base(d, c) + [x(d) - mean_allele x] ^T W onehot(c)

with W fitted by ridge on training loci only. The feature block is the allele-centred
frozen representation, i.e. the design-contrast subspace, because only design-varying
terms can reorder candidates inside a decision group.

Controls: the frozen base alone; the same correction with no context term (a better design
model, not a context model); the same correction with context labels permuted at the locus
level; and an oracle-free check that the fitted correction is monotone-free.

Splits are locus-grouped, where a locus is a connected component of the (target site,
canonical allele) graph. Fold 0 is not touched.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e07_context_correction.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import e04_interaction_predictability as E4  # noqa: E402
import e05_reference_panel as E5  # noqa: E402

N_PCA = 32
LAMBDAS = (1e1, 1e2, 1e3, 1e4, 1e5)
N_FOLDS = 5


def build() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    man = pd.read_parquet(C.require_manifest())
    idx = pd.read_parquet(C.CACHE / "embed_index.parquet")
    E = np.load(C.CACHE / "embeddings.npy")
    man = man.merge(idx[["record_id", "row", "oof_pred"]], on="record_id", validate="1:1")
    man = man.assign(locus_group=E4.locus_groups(man))
    d = man[man.fold >= 1].copy()
    d["z"] = E5.arcsine(d.edited.to_numpy())
    d["base_z"] = E5.arcsine(d.oof_pred.to_numpy())
    grp = ["decision_group", "design_key", "edit_key", "context_key", "cond3",
           "locus_group"]
    d = d.groupby(grp, observed=True, as_index=False).agg(
        y=("edited", "mean"), z=("z", "mean"), base=("oof_pred", "mean"),
        base_z=("base_z", "mean"), row=("row", "first"))
    # only groups that pose a choice
    nd = d.groupby("decision_group", observed=True).design_key.transform("nunique")
    d = d[nd >= 2].sort_values(["decision_group"], kind="stable").reset_index(drop=True)
    d["cpos"] = np.arange(len(d))
    Ec = E5.allele_centred(E, d.rename(columns={"edit_key": "edit_key"}))
    return d, E, Ec


def design_blocks(d: pd.DataFrame, Ec: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Allele-centred features and the context one-hot, ready for an outer product."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Ec), min(60000, len(Ec)), replace=False)
    _, _, Vt = np.linalg.svd(Ec[sub], full_matrices=False)
    X = (Ec @ Vt[:N_PCA].T).astype(np.float64)
    X /= X.std(0, keepdims=True)
    ctx = sorted(d.cond3.unique())
    Cm = np.zeros((len(d), len(ctx)))
    Cm[np.arange(len(d)), [ctx.index(c) for c in d.cond3]] = 1.0
    return X, Cm


def group_centre(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Subtract each decision group's own mean.

    Only within-group variation can reorder candidates, so the correction is fitted and
    evaluated on within-group deviations. This also removes the locus-level component of
    the residual, which is large, irreducible from design features, and would otherwise
    dominate the least-squares fit.
    """
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    cnt = np.diff(np.r_[starts, len(g)])
    means = np.add.reduceat(v, starts, axis=0) / cnt.reshape(-1, *([1] * (v.ndim - 1)))
    return v - np.repeat(means, cnt, axis=0)


def fit_predict(Xtr, ytr, Xte, lam_grid, rng) -> np.ndarray:
    """Ridge with the penalty chosen on an inner random split of the training rows."""
    hold = rng.random(len(Xtr)) < 0.2
    best, blam = np.inf, lam_grid[0]
    for lam in lam_grid:
        w = E4.ridge_fit(Xtr[~hold], ytr[~hold], lam)
        mse = float(np.mean((ytr[hold] - Xtr[hold] @ w) ** 2))
        if mse < best:
            best, blam = mse, lam
    return Xte @ E4.ridge_fit(Xtr, ytr, blam)


def top1_accuracy(score: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """Fraction of decision groups whose top-ranked candidate is the measured best.

    Written out rather than reusing E05.group_metrics because the alpha and selectivity
    searches call it about a hundred times on 150,000 rows and do not need the per-group
    Spearman loop.
    """
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ymax = np.maximum.reduceat(y, starts)
    keep = ymax > np.minimum.reduceat(y, starts)
    order = np.lexsort((-score, g))
    picks = order[np.flatnonzero(np.r_[True, g[order][1:] != g[order][:-1]])]
    if not keep.any():
        return np.nan
    return float((y[picks][keep] == ymax[keep]).mean())


def group_magnitude(corr: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Each group's predicted reordering size, broadcast back to its rows."""
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    cnt = np.diff(np.r_[starts, len(g)])
    return np.repeat(np.maximum.reduceat(corr, starts)
                     - np.minimum.reduceat(corr, starts), cnt)


def choose_selectivity(base: np.ndarray, corr: np.ndarray, y: np.ndarray, g: np.ndarray,
                       alpha: float, grid=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0)) -> float:
    """Fraction of decision groups the correction is allowed to touch.

    E04 found the interaction's sign is predicted correctly on 94% of the top 5% of
    quartets ranked by predicted magnitude, against near-chance overall. That asymmetry
    argues for using the correction selectively: reorder only the groups where a large
    reordering is predicted, and leave the frozen ranking alone everywhere else. The
    fraction is chosen on training loci, and 0.0 recovers the frozen model.
    """
    mag = group_magnitude(corr, g)
    best, bq = -np.inf, 0.0
    for q in grid:
        sc = base if q == 0.0 else base + alpha * corr * (mag >= np.quantile(mag, 1 - q))
        acc = top1_accuracy(sc, y, g)
        if np.isfinite(acc) and acc > best:
            best, bq = acc, q
    return bq


def choose_alpha(base: np.ndarray, corr: np.ndarray, y: np.ndarray, g: np.ndarray,
                 grid=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0)) -> float:
    """Blend weight for the correction, chosen on training rows by the decision metric.

    Without this the comparison is unfair in the other direction: a correction fitted by
    least squares against a residual whose bulk is irreducible has no reason to arrive at
    the scale that helps a ranking, and alpha = 0 recovers the frozen model exactly, so a
    genuinely useless correction can be switched off rather than made to look harmful.
    """
    best, balpha = -np.inf, 0.0
    for a in grid:
        acc = top1_accuracy(base + a * corr, y, g)
        if np.isfinite(acc) and acc > best:
            best, balpha = acc, a
    return balpha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d, E, Ec = build()
    X, Cm = design_blocks(d, Ec, args.seed)
    gcode = pd.factorize(d.decision_group)[0]
    # within-group deviations; the centred base score is carried as its own covariate so
    # the correction cannot smuggle in a rescaling of the frozen score
    resid = group_centre((d.z - d.base_z).to_numpy(), gcode)
    X = group_centre(X, gcode)
    base_c = group_centre(d.base_z.to_numpy(), gcode).reshape(-1, 1)

    inter = np.hstack([E4.interaction_matrix(X, Cm), base_c])
    X = np.hstack([X, base_c])
    # locus-level permutation of the context block: every context keeps its marginal
    # share and every locus keeps its residuals, only the pairing dies
    perm = np.arange(len(d))
    for _, ii in pd.Series(np.arange(len(d))).groupby(d.locus_group.to_numpy()):
        v = ii.to_numpy()
        perm[v] = rng.permutation(v)
    inter_shuf = np.hstack([E4.interaction_matrix(X[:, :-1], Cm[perm]), base_c])

    loci = d.locus_group.to_numpy()
    uniq = np.unique(loci)
    assign = dict(zip(uniq, rng.integers(0, N_FOLDS, len(uniq))))
    fold = np.array([assign[g] for g in loci])

    arms = {"design_only": X, "design_x_context": inter,
            "design_x_context_SHUFFLED": inter_shuf}
    pred = {k: np.full(len(d), np.nan) for k in arms}
    pred_sel = {k: np.zeros(len(d)) for k in arms}
    alphas = {k: [] for k in arms}
    sel = {k: [] for k in arms}
    base_all = d.base_z.to_numpy()
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        for name, M in arms.items():
            pred[name][te] = fit_predict(M[tr], resid[tr], M[te], LAMBDAS, rng)
            # alpha is chosen on TRAINING loci only, using in-sample corrections there
            corr_tr = fit_predict(M[tr], resid[tr], M[tr], LAMBDAS, rng)
            ytr, gtr = d.y.to_numpy()[tr], gcode[tr]
            a = choose_alpha(base_all[tr], corr_tr, ytr, gtr)
            q = choose_selectivity(base_all[tr], corr_tr, ytr, gtr, a)
            alphas[name].append(a)
            sel[name].append(q)
            mag_te = group_magnitude(pred[name][te], gcode[te])
            gate = ((mag_te >= np.quantile(mag_te, 1 - q)) if q > 0
                    else np.zeros(int(te.sum()), bool))
            pred_sel[name][te] = a * pred[name][te] * gate
            pred[name][te] *= a
        print(f"fold {f} done, alphas "
              f"{ {k: v[-1] for k, v in alphas.items()} }", flush=True)

    d = d.assign(_gcode=gcode)
    scores = {"base": base_all}
    for name in arms:
        scores[name] = base_all + pred[name]
    for name in list(arms):
        scores[name + "_SELECTIVE"] = base_all + pred_sel[name]

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "setup": {"rows": int(len(d)),
                     "decision_groups": int(d.decision_group.nunique()),
                     "locus_groups": int(len(uniq)),
                     "contexts": int(d.cond3.nunique()),
                     "n_pca": N_PCA, "folds": N_FOLDS,
                     "lambda_grid": list(LAMBDAS)},
           "blend_alpha_by_fold": {k: [float(x) for x in v] for k, v in alphas.items()},
           "selectivity_by_fold": {k: [float(x) for x in v] for k, v in sel.items()},
           "residual_prediction": {
               name: {"spearman": C.spearman(pred[name], resid),
                      "r2_vs_predict_zero": float(
                          1 - np.mean((resid - pred[name]) ** 2) / np.mean(resid ** 2))}
               for name in arms},
           "overall": {name: E5.group_metrics(d, s) for name, s in scores.items()}}

    per_ctx = []
    for c, s in d.groupby("cond3", observed=True):
        s = s.assign(_gcode=pd.factorize(s.decision_group)[0])
        row = {"cond3": c, "rows": int(len(s))}
        for name, sc in scores.items():
            m = E5.group_metrics(s, sc[s.index.to_numpy()])
            row[f"{name}_acc"] = m.get("accuracy", np.nan)
            row[f"{name}_regret"] = m.get("regret", np.nan)
            row["n_groups"] = m.get("n_groups", 0)
        per_ctx.append(row)
    res["per_context"] = sorted(per_ctx, key=lambda r: -r["rows"])

    # locus-clustered paired interval on the two headline deltas
    res["paired"] = {}
    for name in [k for k in scores if k != "base"]:
        res["paired"][name] = paired_locus_bootstrap(d, scores["base"], scores[name],
                                                     args.seed)
    C.write_outputs("e07_context_correction", res, render(res))


def paired_locus_bootstrap(d: pd.DataFrame, base: np.ndarray, other: np.ndarray,
                           seed: int, n_boot: int = 300) -> dict:
    """Per-decision-group paired differences, resampled by locus group."""
    g = d._gcode.to_numpy()
    y = d.y.to_numpy()
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    cnt = np.diff(np.r_[starts, len(y)])
    ymax = np.maximum.reduceat(y, starts)
    keep = ymax > np.minimum.reduceat(y, starts)

    def picks(s: np.ndarray) -> np.ndarray:
        order = np.lexsort((-s, g))
        return order[np.flatnonzero(np.r_[True, g[order][1:] != g[order][:-1]])]

    pb, po = y[picks(base)], y[picks(other)]
    d_acc = ((po == ymax).astype(float) - (pb == ymax).astype(float))[keep]
    d_reg = ((ymax - po) - (ymax - pb))[keep]
    locus = d.locus_group.to_numpy()[starts][keep]
    rng = np.random.default_rng(seed)
    ul, inv = np.unique(locus, return_inverse=True)
    buckets = [np.flatnonzero(inv == i) for i in range(len(ul))]
    a, r = [], []
    for _ in range(n_boot):
        idx = np.concatenate([buckets[j] for j in rng.integers(0, len(ul), len(ul))])
        a.append(float(d_acc[idx].mean()))
        r.append(float(d_reg[idx].mean()))
    return {"n_groups": int(keep.sum()), "n_locus_clusters": int(len(ul)),
            "d_accuracy": float(d_acc.mean()),
            "d_accuracy_ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "d_regret": float(d_reg.mean()),
            "d_regret_ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


def render(r: dict) -> str:
    s, ov, pa = r["setup"], r["overall"], r["paired"]
    L = ["# E07 - an explicit context-conditioned design correction, on held-out loci\n",
         f"{s['rows']:,} candidate measurements in {s['decision_groups']:,} decision groups "
         f"over {s['locus_groups']:,} locus groups and {s['contexts']} contexts. "
         f"Locus-grouped {s['folds']}-fold; ridge penalty chosen inside each training split; "
         "fold 0 untouched.\n",
         "## Held-out residual prediction\n",
         "| model | Spearman | R^2 vs predicting zero |\n|---|---:|---:|"]
    for k, v in r["residual_prediction"].items():
        L.append(f"| {k} | {v['spearman']:.4f} | {v['r2_vs_predict_zero']:+.4f} |")
    L.append("\nBlend weight chosen on training loci by the decision metric, per fold: "
             f"{r['blend_alpha_by_fold']}. A weight of zero recovers the frozen model "
             "exactly, so a useless correction can be switched off rather than forced on. "
             "The `_SELECTIVE` arms additionally choose, on training loci, what fraction of "
             "decision groups the correction may touch at all: "
             f"{r['selectivity_by_fold']}.\n")
    L.append("\n## The decision\n")
    L.append("| score | groups | top-1 accuracy | selected efficiency | regret | mean Spearman (>=3) |\n|---|---:|---:|---:|---:|---:|")
    for k, v in ov.items():
        L.append(f"| {k} | {v['n_groups']:,} | {v['accuracy']:.4f} | "
                 f"{v['selected_efficiency']:.4f} | {v['regret']:.5f} | "
                 f"{v['mean_spearman_ge3']:.4f} |")
    b = ov["base"]
    L.append(f"\nOracle within these groups is {b['oracle_efficiency']:.4f} and a random pick "
             f"gets {b['random_efficiency']:.4f}, so the whole decision is worth "
             f"{b['oracle_efficiency'] - b['random_efficiency']:.4f} efficiency.\n")
    L.append("## Paired against the frozen model, resampling locus groups\n")
    L.append("| correction | groups | delta accuracy | 95% CI | delta regret | 95% CI |\n|---|---:|---:|---|---:|---|")
    for k, v in pa.items():
        L.append(f"| {k} | {v['n_groups']:,} | {v['d_accuracy']:+.4f} | "
                 f"[{v['d_accuracy_ci95'][0]:+.4f}, {v['d_accuracy_ci95'][1]:+.4f}] | "
                 f"{v['d_regret']:+.5f} | "
                 f"[{v['d_regret_ci95'][0]:+.5f}, {v['d_regret_ci95'][1]:+.5f}] |")
    L.append("\n## By context\n")
    L.append("| context | groups | base acc | design x context acc | selective acc | shuffled selective acc |\n|---|---:|---:|---:|---:|---:|")
    for c in r["per_context"]:
        L.append(f"| {c['cond3']} | {c['n_groups']:,} | {c['base_acc']:.4f} | "
                 f"{c['design_x_context_acc']:.4f} | "
                 f"{c['design_x_context_SELECTIVE_acc']:.4f} | "
                 f"{c['design_x_context_SHUFFLED_SELECTIVE_acc']:.4f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
