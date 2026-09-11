"""E09 - E04's central claim, retested with the three defects in E04 repaired.

`NEXT_STEPS_PLAN.md` sections 2.1, 2.2 and 2.4 identify three problems with E04, all of
which I verified:

1. **Mixed coordinate systems.** `extract_embeddings.py` writes five independently trained
   networks' hidden vectors into one matrix, and 57,821 of E04's 110,321 quartet contrasts
   subtract one network's embedding from another's. Measured on 8,000 fold-0 rows embedded
   by three checkpoints: median per-coordinate correlation is only 0.48, linear CKA is 0.91
   (so the representations agree up to a rotation, which is exactly what coordinate-wise
   subtraction cannot survive), and the pure checkpoint offset has SD 0.322 against a
   genuine design contrast's 0.396. More than half of E04's features therefore carried a
   nuisance term about 80% the size of the signal.
2. **The outer split did not extend through the representation learner.** 23.6% of locus
   components span more than one official fold (36.2% of rows), so a held-out locus could
   have had relatives in the encoder's training data.
3. **Controls used the wrong representation.** E04's shuffles were run on the 17 engineered
   features while the headline used embeddings, so the quoted comparison changed the
   representation as well as the mechanism. Its PCA was also fitted before outer splitting.

This script removes all three at once. For each official fold k it uses only quartets whose
**all four measurements** lie in fold k, so both designs are embedded by checkpoint k -- one
coordinate system -- and checkpoint k never trained on fold k, so no row in the analysis,
train or test, contributed to the representation that describes it. Locus-grouped CV runs
inside each fold; every transform is fitted inside the training split; and all controls use
the identical embedding features and model capacity.

It also separates the two prediction tasks that section 2.6 correctly says E04 conflated:
the sign of D, and whether the two designs actually change order.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e09_clean_interaction_retest.py
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

N_PCA = 32
LAMBDAS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
N_FOLDS = 5
SIGMA_Z = 0.0378447  # E03's pooled per-measurement SD on the arcsine scale


def single_fold_quartets(man: pd.DataFrame) -> pd.DataFrame:
    """Quartets whose four measurements all lie in one official fold.

    Built here rather than assumed, so the script reproduces from the manifest alone. Every
    (allele, design) in this corpus has all of its measurements in a single fold -- 145,980
    of 146,050 -- so the constraint reduces to the two designs sharing a fold.
    """
    cached = C.cache_path("quartets_single_fold")
    if cached.exists():
        return pd.read_parquet(cached)
    q = pd.read_parquet(C.cache_path("quartets_full"))
    g = man.groupby(["edit_key", "design_key"], observed=True).fold.agg(["min", "max"])
    fold = g["min"].where(g["min"] == g["max"])
    f1 = fold.reindex(list(zip(q.edit_key, q.d1))).to_numpy()
    f2 = fold.reindex(list(zip(q.edit_key, q.d2))).to_numpy()
    keep = np.isfinite(f1) & np.isfinite(f2) & (f1 == f2)
    out = q[keep].assign(fold=f1[keep].astype(int))
    out.to_parquet(cached, index=False)
    return out


def prep(fold: int, q: pd.DataFrame, man: pd.DataFrame, E: np.ndarray, rng):
    """Features and target for one official fold, all inside one coordinate system."""
    qf = q[q.fold == fold].reset_index(drop=True)
    sub = man[man.fold == fold]
    if len(qf) < 500 or sub.empty:
        return None
    sub = sub.assign(locus_group=E4.locus_groups(sub))
    rep = sub.drop_duplicates(subset=["edit_key", "design_key"]).copy()
    rep["pk"] = rep.edit_key + "\x00" + rep.design_key
    pos = pd.Series(np.arange(len(rep)), index=rep.pk)
    i1 = pos.reindex((qf.edit_key + "\x00" + qf.d1).to_numpy()).to_numpy()
    i2 = pos.reindex((qf.edit_key + "\x00" + qf.d2).to_numpy()).to_numpy()
    ok = np.isfinite(i1) & np.isfinite(i2)
    qf, i1, i2 = qf[ok].reset_index(drop=True), i1[ok].astype(int), i2[ok].astype(int)
    rows = rep.row.to_numpy()
    emb_d = E[rows[i1]].astype(np.float32) - E[rows[i2]].astype(np.float32)
    locus_of = sub.drop_duplicates("edit_key").set_index("edit_key").locus_group
    groups = qf.edit_key.map(locus_of).to_numpy()
    ctx = sorted(set(qf.c1) | set(qf.c2))
    cd = np.zeros((len(qf), len(ctx)))
    cd[np.arange(len(qf)), [ctx.index(c) for c in qf.c1]] = 1.0
    cd[np.arange(len(qf)), [ctx.index(c) for c in qf.c2]] -= 1.0
    return qf, emb_d, cd, groups, len(ctx)


def cv_split(groups: np.ndarray, rng) -> np.ndarray:
    uniq = np.unique(groups)
    assign = dict(zip(uniq, rng.integers(0, N_FOLDS, len(uniq))))
    return np.array([assign[g] for g in groups])


def run_cv(emb_d, cd, y, groups, fold_id, rng, mode="interaction"):
    """Locus-grouped CV with the PCA and the scaling fitted inside each training split.

    `mode` selects the feature block, and every mode gets the same PCA dimension and the
    same penalty grid, so a control differs from the real arm only in what it is allowed
    to know.
    """
    fold = cv_split(groups, rng)
    pred = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() < 200:
            continue
        A = emb_d[tr]
        mu = A.mean(0)
        _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
        V = Vt[:N_PCA]
        Xtr = ((A - mu) @ V.T).astype(np.float64)
        sd = Xtr.std(0)
        # A principal direction with negligible variance in the training split carries no
        # information and, divided by its own tiny SD, turns rounding error into features
        # with enormous leverage. Fold 0 is small enough that this actually happened: an
        # earlier run of this script returned R^2 of -4e11 there. Drop such directions
        # rather than flooring the divisor.
        live = sd > 1e-6 * sd.max()
        V, sd = V[live], sd[live]
        Xtr = Xtr[:, live] / sd
        Xte = (((emb_d[te] - mu) @ V.T).astype(np.float64)) / sd
        if mode == "interaction":
            Mtr = E4.interaction_matrix(Xtr, cd[tr])
            Mte = E4.interaction_matrix(Xte, cd[te])
        else:                                    # design contrast only, no context term
            Mtr, Mte = Xtr, Xte
        inner = rng.random(int(tr.sum())) < 0.2
        best, blam = np.inf, LAMBDAS[0]
        for lam in LAMBDAS:
            w = E4.ridge_fit(Mtr[~inner], y[tr][~inner], lam)
            mse = float(np.mean((y[tr][inner] - Mtr[inner] @ w) ** 2))
            if mse < best:
                best, blam = mse, lam
        pred[te] = Mte @ E4.ridge_fit(Mtr, y[tr], blam)
    ok = np.isfinite(pred)
    return pred, ok


def score(pred, y, ok) -> dict:
    mse_null = float(np.mean(y[ok] ** 2))
    return {"n": int(ok.sum()),
            "spearman": C.spearman(pred[ok], y[ok]),
            "pearson": float(np.corrcoef(pred[ok], y[ok])[0, 1]),
            "r2_vs_predict_zero": float(1 - np.mean((y[ok] - pred[ok]) ** 2) / mse_null)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    man = pd.read_parquet(C.require_manifest())
    q = single_fold_quartets(man)
    idx = pd.read_parquet(C.CACHE / "embed_index.parquet")
    man = man.merge(idx[["record_id", "row", "checkpoint"]], on="record_id", validate="1:1")
    E = np.load(C.CACHE / "embeddings.npy")

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "design": {
               "quartets_all_four_measurements_in_one_fold": int(len(q)),
               "of_total_quartets": 110321,
               "note": ("Both designs of every quartet are embedded by the checkpoint that "
                        "held that fold out, so one coordinate system and no exposure."),
           },
           "per_fold": [], "pooled": {}}

    per_arm: dict[str, list] = {}
    for fold in sorted(q.fold.unique()):
        rng = np.random.default_rng([args.seed, int(fold)])
        got = prep(int(fold), q, man, E, rng)
        if got is None:
            continue
        qf, emb_d, cd, groups, n_ctx = got
        y = qf.D_z.to_numpy(dtype=np.float64)

        arms = {}
        p, ok = run_cv(emb_d, cd, y, groups, fold, rng, "interaction")
        arms["embedding_x_context"] = score(p, y, ok)
        pred_real = (p, ok)
        p2, ok2 = run_cv(emb_d, cd, y, groups, fold, rng, "design_only")
        arms["embedding_design_only"] = score(p2, y, ok2)

        # controls on the IDENTICAL representation and capacity
        perm = np.arange(len(qf))
        for _, ii in pd.Series(np.arange(len(qf))).groupby(groups):
            v = ii.to_numpy()
            perm[v] = rng.permutation(v)
        p3, ok3 = run_cv(emb_d, cd[perm], y, groups, fold, rng, "interaction")
        arms["SHUFFLED_context_same_features"] = score(p3, y, ok3)

        cp = pd.Series([f"{a}->{b}" for a, b in zip(qf.c1, qf.c2)]).to_numpy()
        yp = y.copy()
        for _, ii in pd.Series(np.arange(len(qf))).groupby(cp):
            v = ii.to_numpy()
            yp[v] = y[rng.permutation(v)]
        p4, ok4 = run_cv(emb_d, cd, yp, groups, fold, rng, "interaction")
        arms["SHUFFLED_D_same_features"] = score(p4, yp, ok4)

        # ---- the two tasks section 2.6 says E04 conflated ----------------------
        p, ok = pred_real
        d1, d2 = qf.dz1.to_numpy(), qf.dz2.to_numpy()
        se = np.sqrt(2.0) * SIGMA_Z
        supported = (np.abs(d1) > 2 * se) & (np.abs(d2) > 2 * se) & ok
        reverses = (np.sign(d1) * np.sign(d2) < 0)
        tasks = {}
        for frac in (1.0, 0.19, 0.05):
            m = ok.copy()
            if frac < 1.0:
                thr = np.quantile(np.abs(p[ok]), 1 - frac)
                m &= np.abs(p) >= thr
            tasks[f"sign_of_D_top_{int(frac*100)}pct"] = float(
                (np.sign(p[m]) == np.sign(y[m])).mean())
            ms = m & supported
            if ms.sum() >= 30:
                # a reversal is predicted when the correction is large enough to flip the
                # measured order of the pair, which is a different event from a large D
                pred_rev = np.abs(p[ms]) > np.abs(d1[ms] + d2[ms]) / 2
                tasks[f"reversal_rate_in_top_{int(frac*100)}pct"] = float(reverses[ms].mean())
                tasks[f"reversal_balanced_acc_top_{int(frac*100)}pct"] = float(
                    0.5 * ((pred_rev & reverses[ms]).sum() / max(reverses[ms].sum(), 1)
                           + (~pred_rev & ~reverses[ms]).sum() / max((~reverses[ms]).sum(), 1)))
                tasks[f"n_supported_in_top_{int(frac*100)}pct"] = int(ms.sum())

        row = {"fold": int(fold), "quartets": int(len(qf)),
               "locus_components": int(len(np.unique(groups))),
               "contexts": int(n_ctx), "sd_D_z": float(y.std()),
               "arms": arms, "tasks": tasks}
        res["per_fold"].append(row)
        for k, v in arms.items():
            per_arm.setdefault(k, []).append(v["r2_vs_predict_zero"])
        print(f"fold {fold}: " + ", ".join(
            f"{k}={v['r2_vs_predict_zero']:+.4f}" for k, v in arms.items()), flush=True)

    dev = [r for r in res["per_fold"] if r["fold"] >= 1]
    res["pooled"] = {
        "development_folds": [r["fold"] for r in dev],
        **{k: {"mean_r2": float(np.mean([r["arms"][k]["r2_vs_predict_zero"] for r in dev])),
               "min_r2": float(np.min([r["arms"][k]["r2_vs_predict_zero"] for r in dev])),
               "max_r2": float(np.max([r["arms"][k]["r2_vs_predict_zero"] for r in dev])),
               "mean_spearman": float(np.mean([r["arms"][k]["spearman"] for r in dev]))}
           for k in per_arm},
    }

    # ---- positive control that measures recovery of the injected component ----------
    rng = np.random.default_rng(args.seed + 1)
    got = prep(1, q, man, E, rng)
    qf, emb_d, cd, groups, _ = got
    n = len(qf)
    Xr = ((emb_d - emb_d.mean(0)) @ np.linalg.svd(
        emb_d - emb_d.mean(0), full_matrices=False)[2][:N_PCA].T).astype(np.float64)
    Xr /= np.maximum(Xr.std(0, keepdims=True), 1e-9)
    M = E4.interaction_matrix(Xr, cd)
    beta = np.zeros(M.shape[1])
    live = rng.choice(M.shape[1], max(1, M.shape[1] // 8), replace=False)
    beta[live] = rng.normal(size=live.size)
    signal = M @ beta
    signal = (signal - signal.mean()) / signal.std()
    inj = {}
    for frac in (0.02, 0.05, 0.10, 0.25):
        # inject into a NULL target: pure measurement noise at the additive-model scale,
        # so any recovered structure is the injected component and nothing else
        y_null = rng.normal(0.0, 2 * SIGMA_Z, n)
        target = y_null + frac * float(qf.D_z.std()) * signal
        p, ok = run_cv(emb_d, cd, target, groups, 1, rng, "interaction")
        inj[str(frac)] = {
            "injected_sd_as_frac_of_sd_D": frac,
            "r2_vs_predict_zero_on_injected_target": score(p, target, ok)["r2_vs_predict_zero"],
            "correlation_with_the_injected_component_alone": float(
                np.corrcoef(p[ok], signal[ok])[0, 1]),
        }
        print(f"inject {frac}: r_with_signal="
              f"{inj[str(frac)]['correlation_with_the_injected_component_alone']:.3f}",
              flush=True)
    res["positive_control_on_null_target"] = inj

    C.write_outputs("e09_clean_interaction_retest", res, render(res))


def render(r: dict) -> str:
    d, pl = r["design"], r["pooled"]
    L = ["# E09 - E04's claim retested with one coordinate system and no encoder exposure\n",
         f"{d['quartets_all_four_measurements_in_one_fold']:,} of {d['of_total_quartets']:,} "
         "quartets have all four measurements inside a single official fold. Those are the "
         "only ones where both designs are embedded by the same checkpoint **and** that "
         "checkpoint never trained on any row in the analysis. Everything below uses only "
         "those, fold by fold, with locus-grouped CV and every transform fitted inside the "
         "training split.\n",
         "## Held-out prediction of D, per fold\n",
         "| fold | quartets | loci | contexts | embedding x context | design only | "
         "SHUFFLED context | SHUFFLED D |\n|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in r["per_fold"]:
        a = row["arms"]
        L.append(f"| {row['fold']}{' (historical held-out)' if row['fold'] == 0 else ''} | "
                 f"{row['quartets']:,} | {row['locus_components']:,} | {row['contexts']} | "
                 f"{a['embedding_x_context']['r2_vs_predict_zero']:+.4f} | "
                 f"{a['embedding_design_only']['r2_vs_predict_zero']:+.4f} | "
                 f"{a['SHUFFLED_context_same_features']['r2_vs_predict_zero']:+.4f} | "
                 f"{a['SHUFFLED_D_same_features']['r2_vs_predict_zero']:+.4f} |")
    L.append("\nR² against predicting zero. All four arms share the representation, the PCA "
             "dimension, the penalty grid and the split.\n")
    L.append("## Pooled over development folds\n")
    L.append("| arm | mean R² | range | mean Spearman |\n|---|---:|---|---:|")
    for k in ("embedding_x_context", "embedding_design_only",
              "SHUFFLED_context_same_features", "SHUFFLED_D_same_features"):
        v = pl[k]
        L.append(f"| {k} | {v['mean_r2']:+.4f} | [{v['min_r2']:+.4f}, {v['max_r2']:+.4f}] | "
                 f"{v['mean_spearman']:.4f} |")
    L.append("\n## Sign of D versus actual order reversal\n")
    L.append("Section 2.6 of the next-steps plan is right that these are different events: "
             "D can be large with both contrasts the same sign, which is a change of margin "
             "and not a change of choice. Reported separately, on development folds.\n")
    L.append("| fold | sign of D, all | sign of D, top 5% | supported quartets in top 5% | "
             "reversal rate there | reversal balanced accuracy |\n|---|---:|---:|---:|---:|---:|")
    for row in r["per_fold"]:
        t = row["tasks"]
        if row["fold"] == 0:
            continue
        L.append(f"| {row['fold']} | {t.get('sign_of_D_top_100pct', float('nan')):.3f} | "
                 f"{t.get('sign_of_D_top_5pct', float('nan')):.3f} | "
                 f"{t.get('n_supported_in_top_5pct', 0):,} | "
                 f"{t.get('reversal_rate_in_top_5pct', float('nan')):.3f} | "
                 f"{t.get('reversal_balanced_acc_top_5pct', float('nan')):.3f} |")
    L.append("\n## Positive control, injected into a null target\n")
    L.append("E04's control added a synthetic signal to the observed D and scored the sum, "
             "which cannot show that the added component was the part recovered. Here the "
             "signal is injected into pure measurement noise at the additive-model scale, "
             "and the recovery of that component is measured directly.\n")
    L.append("| injected SD (fraction of SD of D) | R² on the injected target | correlation "
             "with the injected component |\n|---|---:|---:|")
    for k, v in r["positive_control_on_null_target"].items():
        L.append(f"| {float(k):.2f} | {v['r2_vs_predict_zero_on_injected_target']:+.4f} | "
                 f"{v['correlation_with_the_injected_component_alone']:.3f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
