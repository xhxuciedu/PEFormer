"""E04 - can the design-by-context interaction be predicted on unseen loci?

Research plan section 2, item 3 ("predictive identifiability") and Gate A: an interaction
only earns a place in a model if it can be predicted at held-out loci better than an
additive model and better than a structured shuffle. Lowering a model's cross-context
correlation is not evidence; adding noise would do that too.

Target: the quartet contrast D on the arcsine-root scale, from E03's matched support.
D is antisymmetric under swapping the two designs and under swapping the two contexts, so
the model class is chosen to be antisymmetric in both by construction:

    D_hat = sum_{f,k} beta_{f,k} * [x_f(d1) - x_f(d2)] * [onehot_k(c1) - onehot_k(c2)]

i.e. a feature-by-context interaction, fitted by ridge. Two feature sets are compared: the
17 interpretable pegRNA/edit features used by the manuscript's gradient-boosted baseline,
and the first 32 principal directions of the frozen ordinal-S4D representation.

Splits are locus-disjoint: loci are the connected components of the (target site, canonical
allele) graph, so an allele reachable from two protospacers and two alleles at one site all
travel together.

Controls: predict-zero (the additive model), design features with no context term,
context-pair labels permuted at the locus level, and D permuted within context pair. A
positive control injects a synthetic interaction of known size and checks the same pipeline
recovers it, which distinguishes "no signal" from "no power".

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e04_interaction_predictability.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402

FEATS = ["pbs_length", "rtt_length", "pbs_gc", "rtt_gc", "extension_gc", "edit_length",
         "edit_position", "edit_position_from_nick", "edit_type_idx", "n_mismatch",
         "pbs_tm", "rtt_tm", "proto_mfe", "rtt_mfe", "pbs_mfe", "extension_mfe",
         "ruleset3_score"]
N_EMB_PCA = 32
LAMBDAS = (1e1, 1e2, 1e3, 1e4, 1e5)
N_FOLDS = 5


# --------------------------------------------------------------------------- #
# locus grouping
# --------------------------------------------------------------------------- #
def locus_groups(man: pd.DataFrame) -> pd.Series:
    """Connected components of the (target site, canonical allele) bipartite graph.

    Holding out an exact design, or even an allele, is not enough: two alleles at one
    target site share 40 bp of sequence, and one allele reached by two protospacers appears
    under two site names. A union-find over both relations gives the coarsest honest
    holdout unit available without genomic coordinates.
    """
    t_codes, t_uniq = pd.factorize(man.target_name)
    e_codes, e_uniq = pd.factorize(man.edit_key)
    n_t = len(t_uniq)
    parent = np.arange(n_t + len(e_uniq))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(t_codes, e_codes + n_t):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(int(a)) for a in t_codes])
    return pd.Series(roots, index=man.index)


# --------------------------------------------------------------------------- #
# design matrices
# --------------------------------------------------------------------------- #
def interaction_matrix(fd: np.ndarray, cd: np.ndarray) -> np.ndarray:
    """Row-wise outer product of the feature contrast and the context contrast."""
    return (fd[:, :, None] * cd[:, None, :]).reshape(len(fd), -1)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
    return np.linalg.solve(A, X.T @ y)


def cv_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam_grid,
            seed: int) -> dict:
    """Locus-grouped K-fold with the ridge penalty chosen inside each training split.

    No intercept: D is antisymmetric, so a nonzero intercept would be a statement that
    design 1 beats design 2 in general, which is an artefact of key ordering.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    assign = dict(zip(uniq, rng.integers(0, N_FOLDS, len(uniq))))
    fold = np.array([assign[g] for g in groups])
    pred = np.full(len(y), np.nan)
    chosen = []
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() < 50:
            continue
        # inner split on loci for the penalty
        gtr = np.unique(groups[tr])
        inner = set(rng.choice(gtr, max(1, len(gtr) // 5), replace=False).tolist())
        im = np.array([g in inner for g in groups[tr]])
        best, blam = -np.inf, lam_grid[0]
        for lam in lam_grid:
            w = ridge_fit(X[tr][~im], y[tr][~im], lam)
            p = X[tr][im] @ w
            sc = -float(np.mean((y[tr][im] - p) ** 2))
            if sc > best:
                best, blam = sc, lam
        w = ridge_fit(X[tr], y[tr], blam)
        pred[te] = X[te] @ w
        chosen.append(blam)
    ok = np.isfinite(pred)
    mse_null = float(np.mean(y[ok] ** 2))
    mse = float(np.mean((y[ok] - pred[ok]) ** 2))
    # Sign agreement over every quartet is dominated by the ones whose D is mostly noise,
    # so it is also reported on the deciles where the interaction is actually large. A
    # model can be strongly correlated with D and still look near-chance on signs if the
    # correlation lives in the tails, which is exactly the regime a user cares about.
    strat = {}
    absy = np.abs(y[ok])
    absp = np.abs(pred[ok])
    for q0 in (0.5, 0.8, 0.9, 0.95):
        for lbl, v in (("by_true", absy), ("by_pred", absp)):
            m = v >= np.quantile(v, q0)
            strat[f"sign_acc_{lbl}_top_{int((1-q0)*100)}pct"] = float(
                (np.sign(pred[ok][m]) == np.sign(y[ok][m])).mean())
    return {"n": int(ok.sum()), "lambdas": chosen, "sign_accuracy_stratified": strat,
            "spearman": C.spearman(pred[ok], y[ok]),
            "pearson": float(np.corrcoef(pred[ok], y[ok])[0, 1]),
            "r2_vs_predict_zero": float(1 - mse / mse_null),
            "mse": mse, "mse_predict_zero": mse_null,
            "sign_accuracy": float((np.sign(pred[ok]) == np.sign(y[ok])).mean()),
            "pred": pred}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--quartets", type=Path,
                    default=C.cache_path("quartets_full"))
    ap.add_argument("--max-quartets", type=int, default=400000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    q = pd.read_parquet(args.quartets)
    man = pd.read_parquet(C.require_manifest())
    feats = pd.read_parquet(C.ROOT / "data/processed/family_c_features.parquet")
    idx = pd.read_parquet(C.CACHE / "embed_index.parquet")
    E = np.load(C.CACHE / "embeddings.npy")

    # one representative row per (allele, design): features are context-free
    man = man.merge(feats, on="record_id", validate="1:1").merge(
        idx[["record_id", "row"]], on="record_id", validate="1:1")
    man = man.assign(locus_group=locus_groups(man))
    rep = man.drop_duplicates(subset=["edit_key", "design_key"]).copy()
    # A pair key as one string: a two-level MultiIndex .loc over 110,000 tuples is
    # pathologically slow here (hours), and a reindex on a string key is not.
    rep["pair_key"] = rep.edit_key + "\x00" + rep.design_key
    rep = rep.set_index("pair_key")
    locus_of = man.drop_duplicates("edit_key").set_index("edit_key").locus_group

    # subsample quartets at the LOCUS level so a train/test split is never split
    q = q.assign(locus_group=q.edit_key.map(locus_of))
    q = q[q.locus_group.notna()]
    if len(q) > args.max_quartets:
        # Subsample whole loci, so no train/test split is ever cut through a locus.
        counts = q.locus_group.value_counts()
        order = counts.sample(frac=1.0, random_state=args.seed).index
        keep = set(order[counts.loc[order].cumsum() <= args.max_quartets].tolist())
        q = q[q.locus_group.isin(keep)]

    pos = pd.Series(np.arange(len(rep)), index=rep.index)
    i1 = pos.reindex((q.edit_key + "\x00" + q.d1).to_numpy()).to_numpy()
    i2 = pos.reindex((q.edit_key + "\x00" + q.d2).to_numpy()).to_numpy()
    assert np.isfinite(i1).all() and np.isfinite(i2).all(), "quartet design not in manifest"
    i1, i2 = i1.astype(int), i2.astype(int)
    F = rep[FEATS].to_numpy(dtype=np.float64)
    f1, f2 = F[i1], F[i2]
    fd = np.nan_to_num(f1 - f2)
    mu = np.nanmean(np.vstack([f1, f2]), 0)
    sd = np.nanstd(np.vstack([f1, f2]), 0)
    sd[sd == 0] = 1.0
    fd = fd / sd

    rows = rep.row.to_numpy()
    emb_d = E[rows[i1]].astype(np.float32) - E[rows[i2]].astype(np.float32)
    # PCA of the *difference* space, which is what the interaction model consumes
    sub = rng.choice(len(emb_d), min(40000, len(emb_d)), replace=False)
    _, _, Vt = np.linalg.svd(emb_d[sub] - emb_d[sub].mean(0), full_matrices=False)
    Vk = Vt[:N_EMB_PCA]
    ed = (emb_d @ Vk.T).astype(np.float64)
    ed /= ed.std(0, keepdims=True)

    ctx = sorted(set(q.c1) | set(q.c2))
    ci = {c: i for i, c in enumerate(ctx)}
    cd = np.zeros((len(q), len(ctx)))
    cd[np.arange(len(q)), [ci[c] for c in q.c1]] = 1.0
    cd[np.arange(len(q)), [ci[c] for c in q.c2]] -= 1.0

    y = q.D_z.to_numpy(dtype=np.float64)
    groups = q.locus_group.to_numpy()

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "support": {"n_quartets": int(len(q)),
                       "n_loci": int(len(np.unique(groups))),
                       "n_contexts": len(ctx),
                       "sd_D_z": float(y.std())},
           "arms": {}}

    def run(name: str, X: np.ndarray, yy: np.ndarray, gg: np.ndarray, seed: int) -> None:
        out = cv_eval(X, yy, gg, LAMBDAS, seed)
        out.pop("pred")
        res["arms"][name] = out
        print("ARM", name, {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in out.items() if k != "lambdas"}, flush=True)

    res["arms"]["additive_predict_zero"] = {
        "n": int(len(y)), "spearman": np.nan, "pearson": np.nan,
        "r2_vs_predict_zero": 0.0, "mse": float(np.mean(y ** 2)),
        "mse_predict_zero": float(np.mean(y ** 2)), "sign_accuracy": 0.5}

    run("feature_x_context", interaction_matrix(fd, cd), y, groups, args.seed)
    run("embedding_x_context", interaction_matrix(ed, cd), y, groups, args.seed)
    run("design_features_only", fd, y, groups, args.seed)

    # ---- structured shuffles ------------------------------------------------
    # (a) permute the context-pair identity across quartets, within locus: the marginal
    #     distribution of context pairs and of D are untouched, only their alignment dies.
    perm = np.arange(len(q))
    for _, ii in pd.Series(np.arange(len(q))).groupby(groups):
        v = ii.to_numpy()
        perm[v] = rng.permutation(v)
    run("feature_x_context_SHUFFLED_context", interaction_matrix(fd, cd[perm]), y,
        groups, args.seed)
    # (b) permute D within context pair, which keeps every context pair's D distribution
    cp = pd.Series([f"{a}->{b}" for a, b in zip(q.c1, q.c2)])
    yperm = y.copy()
    for _, ii in pd.Series(np.arange(len(q))).groupby(cp.to_numpy()):
        v = ii.to_numpy()
        yperm[v] = y[rng.permutation(v)]
    run("feature_x_context_SHUFFLED_D", interaction_matrix(fd, cd), yperm, groups,
        args.seed)

    # ---- positive control ---------------------------------------------------
    # Inject a synthetic interaction whose SD is a stated multiple of the interaction SD
    # E03 estimated from variance components, using a random sparse coefficient vector.
    Xf = interaction_matrix(fd, cd)
    beta = np.zeros(Xf.shape[1])
    live = rng.choice(Xf.shape[1], max(1, Xf.shape[1] // 8), replace=False)
    beta[live] = rng.normal(size=live.size)
    signal = Xf @ beta
    signal /= signal.std()
    inj = {}
    for frac in (0.05, 0.10, 0.25, 0.50):
        target_sd = frac * y.std()
        out = cv_eval(Xf, y + target_sd * signal, groups, LAMBDAS, args.seed)
        out.pop("pred")
        inj[str(frac)] = {"injected_sd_as_frac_of_sd_D": frac,
                          "injected_sd": float(target_sd), **out}
        print("inject", frac, round(out["spearman"], 4), round(out["r2_vs_predict_zero"], 4),
              flush=True)
    res["injected_interaction_positive_control"] = inj

    C.write_outputs("e04_interaction_predictability", res, render(res))


def render(r: dict) -> str:
    s = r["support"]
    L = ["# E04 - is the interaction predictable on held-out loci?\n",
         f"{s['n_quartets']:,} matched quartets over {s['n_loci']:,} locus groups and "
         f"{s['n_contexts']} contexts; SD of D on the arcsine scale is {s['sd_D_z']:.4f}. "
         "Locus-grouped 5-fold, ridge penalty chosen inside each training split, no "
         "intercept (D is antisymmetric).\n",
         "## Held-out prediction of D\n",
         "| model | Spearman | Pearson | R^2 vs predicting zero | sign accuracy |\n|---|---:|---:|---:|---:|"]
    for name, v in r["arms"].items():
        sp = "n/a" if not np.isfinite(v["spearman"]) else f"{v['spearman']:.4f}"
        pe = "n/a" if not np.isfinite(v["pearson"]) else f"{v['pearson']:.4f}"
        L.append(f"| {name} | {sp} | {pe} | {v['r2_vs_predict_zero']:+.4f} | "
                 f"{v['sign_accuracy']:.3f} |")
    ref = r["arms"].get("embedding_x_context", {}).get("sign_accuracy_stratified")
    if ref:
        L.append("\n### Where the signal sits\n")
        L.append("Sign agreement for `embedding_x_context`, restricted to the largest "
                 "quartets by true \\|D\\| and by predicted \\|D\\|.\n")
        L.append("| subset | sign accuracy |\n|---|---:|")
        for k, v in ref.items():
            L.append(f"| {k.replace('_', ' ')} | {v:.3f} |")
    L.append("\n## Injected-interaction positive control\n")
    L.append("Same features, same splits, same penalty search, with a synthetic "
             "feature-by-context interaction added at a known size.\n")
    L.append("| injected SD (fraction of SD of D) | Spearman | R^2 | sign accuracy |\n|---|---:|---:|---:|")
    for k, v in r["injected_interaction_positive_control"].items():
        L.append(f"| {float(k):.2f} | {v['spearman']:.4f} | {v['r2_vs_predict_zero']:+.4f} | "
                 f"{v['sign_accuracy']:.3f} |")
    L.append("\nThe positive control says what effect size this analysis could have found. "
             "A null result above is informative only down to the smallest injected size "
             "the pipeline still recovers.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
