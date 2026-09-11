"""E05 - does measuring a small reference panel in a new context improve pegRNA choice?

Research plan priority 1 and section 8A days 5-10: "implement frozen-embedding adaptation
and intercept-only/ridge/fine-tuning controls on newly defined development episodes.
Evaluate several reference budgets."

The model of the plan is

    eta(d, c) = f_theta(d) + a_c + u_theta(d)^T z_c

with f_theta frozen. Here f_theta is the out-of-fold ordinal-S4D backbone: the pooled
pre-FiLM representation and the model's own prediction, both produced by the checkpoint
that held the row's official fold out (see explore_v2/extract_embeddings.py). a_c is a mean
shift and z_c a low-dimensional context vector, both estimated **only** from support
measurements taken inside the target context.

Episode construction. One episode = one target context, one support/query split of that
context's canonical alleles, one measurement budget. Support and query alleles are
disjoint, so nothing about a query allele is ever measured. Query groups are decision
groups (one canonical allele, one full context) with at least two alternative designs, so
the metric is a choice among interchangeable designs and not a comparison of loci.

Hyperparameters (PCA basis, ridge penalty, rank) are fitted or chosen on SOURCE contexts
only, never on the target context, and the PCA basis is fitted on source-context rows.

The decisive comparison is not "does adaptation beat nothing" but "does adaptation beat an
equally informed monotone recalibration". A per-context intercept or any monotone map
cannot reorder candidates inside a decision group at all, so it is the sharpest possible
control for the claim that reference measurements carry *design-dependent* information.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e05_reference_panel.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402

BUDGETS = (0, 12, 24, 48, 96, 192)
RANKS = (2, 4)
LAMBDAS = (10.0, 100.0, 1000.0)
N_PCA = 32
MIN_QUERY_GROUPS = 20
MAX_QUERY_GROUPS = 3000


def arcsine(y):
    return np.arcsin(np.sqrt(np.clip(y, 0.0, 1.0)))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load() -> tuple[pd.DataFrame, np.ndarray]:
    man = pd.read_parquet(C.require_manifest())
    idx = pd.read_parquet(C.CACHE / "embed_index.parquet")
    E = np.load(C.CACHE / "embeddings.npy")
    d = man.merge(idx, on="record_id", validate="1:1")
    d = d[d.fold_x >= 1].copy()          # development rows only; fold 0 stays untouched
    d = d.rename(columns={"fold_x": "fold"}).drop(columns=["fold_y"])
    # replicates are measurements of one design, not two candidates
    d["z"] = arcsine(d.edited.to_numpy())
    d["base_z"] = arcsine(d.oof_pred.to_numpy())
    grp = ["decision_group", "design_key", "edit_key", "context_key", "cond3", "spacer",
           "target_name", "fold"]
    agg = d.groupby(grp, observed=True, as_index=False).agg(
        y=("edited", "mean"), z=("z", "mean"), base=("oof_pred", "mean"),
        base_z=("base_z", "mean"), row=("row", "first"), n_rep=("edited", "size"))
    agg = agg.sort_values("edit_key", kind="stable").reset_index(drop=True)
    agg["cpos"] = np.arange(len(agg))
    return agg, E


def allele_centred(E: np.ndarray, d: pd.DataFrame) -> np.ndarray:
    """Frozen embeddings with each allele's own mean removed.

    Within a decision group only the *differences* between designs can change the ranking,
    and the leading principal directions of the raw embedding are dominated by which locus
    a row belongs to. Removing the allele mean leaves exactly the design-contrast subspace,
    which is where any design-by-context interaction has to live. The allele mean uses
    sequence only -- no outcome enters it -- so it is available for query alleles too.
    """
    X = E[d.row.to_numpy()].astype(np.float32)
    codes = pd.factorize(d.edit_key, sort=False)[0]
    sums = np.zeros((codes.max() + 1, X.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, X)
    cnt = np.bincount(codes, minlength=codes.max() + 1).astype(np.float64)
    return (X - (sums / cnt[:, None])[codes]).astype(np.float32)


def pca_basis(E: np.ndarray, rows: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-n principal directions of the frozen embedding, fitted on source rows only."""
    rng = np.random.default_rng(seed)
    sub = rows if len(rows) <= 60000 else rng.choice(rows, 60000, replace=False)
    X = E[sub].astype(np.float32)
    mu = X.mean(0)
    X = X - mu
    # economy SVD on the sampled matrix; 768 columns so this is cheap
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return mu, Vt[:n].astype(np.float32)


def features(E: np.ndarray, rows: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    X = E[rows].astype(np.float32) - mu
    return X @ V.T


def features_centred(Ec: np.ndarray, cpos: np.ndarray, Vw: np.ndarray) -> np.ndarray:
    return Ec[cpos] @ Vw.T


# --------------------------------------------------------------------------- #
# metrics on decision groups
# --------------------------------------------------------------------------- #
def group_metrics(q: pd.DataFrame, score: np.ndarray) -> dict:
    """Within-decision-group ranking and choice quality.

    Vectorised over groups: episodes reach 20,000 decision groups and seven arms, so a
    Python loop per group would dominate the runtime. Groups whose measured outcomes are
    all equal carry no information about a choice and are dropped from every arm
    identically.
    """
    g = q["_gcode"].to_numpy()
    y = q.y.to_numpy()
    s = np.asarray(score, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ymax = np.maximum.reduceat(y, starts)
    ymin = np.minimum.reduceat(y, starts)
    ysum = np.add.reduceat(y, starts)
    cnt = np.diff(np.r_[starts, len(y)])
    keep = ymax > ymin
    if not keep.any():
        return {"n_groups": 0}
    # the row each arm would pick: first row of each group once sorted by (group, -score)
    order = np.lexsort((-s, g))
    pick_rows = order[np.flatnonzero(np.r_[True, g[order][1:] != g[order][:-1]])]
    picked = y[pick_rows][keep]
    best = ymax[keep]
    rnd = (ysum / cnt)[keep]
    out = {"n_groups": int(keep.sum()), "accuracy": float((picked == best).mean()),
           "selected_efficiency": float(picked.mean()),
           "regret": float((best - picked).mean()),
           "oracle_efficiency": float(best.mean()),
           "random_efficiency": float(rnd.mean())}
    out["regret_reduction_vs_random"] = float(
        1 - out["regret"] / max(best.mean() - rnd.mean(), 1e-12))
    big = np.flatnonzero(keep & (cnt >= 3))
    if big.size:
        rho = []
        for i in big:
            a, b = starts[i], starts[i] + cnt[i]
            rho.append(C.spearman(s[a:b], y[a:b]))
        out["mean_spearman_ge3"] = float(np.nanmean(rho))
    else:
        out["mean_spearman_ge3"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# adaptation arms
# --------------------------------------------------------------------------- #
def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge with an unpenalised intercept, solved in closed form."""
    X1 = np.hstack([np.ones((len(X), 1), dtype=np.float32), X])
    A = X1.T @ X1
    A[1:, 1:] += lam * np.eye(X.shape[1], dtype=np.float32)
    return np.linalg.solve(A, X1.T @ y.astype(np.float32))


def apply_ridge(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1), dtype=np.float32), X]) @ w


def arms(sup: pd.DataFrame, qry: pd.DataFrame, Xs: np.ndarray, Xq: np.ndarray,
         Xsw: np.ndarray, Xqw: np.ndarray,
         lam: float, rank: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Every arm sees exactly the same support labels and the same frozen features."""
    out: dict[str, np.ndarray] = {}
    base_q = qry.base_z.to_numpy(dtype=np.float32)
    out["base"] = base_q
    if len(sup) == 0:
        return out

    y_s = sup.z.to_numpy(dtype=np.float32)
    b_s = sup.base_z.to_numpy(dtype=np.float32)
    r_s = y_s - b_s

    # 1. intercept-only calibration: cannot reorder anything inside a group, by
    #    construction. Included to make that explicit rather than assumed.
    out["intercept"] = base_q + float(r_s.mean())

    # 2. affine recalibration of the frozen score, still monotone in the prediction
    A = np.stack([np.ones_like(b_s), b_s], 1)
    coef = np.linalg.lstsq(A, y_s, rcond=None)[0]
    out["affine"] = coef[0] + coef[1] * base_q

    # 3. low-rank context vector: residual regressed on the first `rank` principal
    #    directions of the frozen representation. This is u(d)^T z_c with u fixed.
    w = fit_ridge(Xs[:, :rank], r_s, lam)
    out[f"lowrank{rank}"] = base_q + apply_ridge(Xq[:, :rank], w)

    # 4. full frozen-feature residual ridge (head-only adaptation)
    w = fit_ridge(Xs, r_s, lam)
    out["residual_ridge"] = base_q + apply_ridge(Xq, w)

    # 5. support-fitted ridge that ignores the pretrained score entirely
    w = fit_ridge(Xs, y_s, lam)
    out["support_only_ridge"] = apply_ridge(Xq, w)

    # 6. shuffled-support control: identical computation and budget, support outcomes
    #    permuted across designs so no design-dependent information survives
    perm = rng.permutation(len(r_s))
    w = fit_ridge(Xs, r_s[perm], lam)
    out["shuffled_support"] = base_q + apply_ridge(Xq, w)

    # 7-9. the same three fits in the allele-centred (design-contrast) basis, which is the
    #      subspace a design-by-context interaction must occupy
    w = fit_ridge(Xsw[:, :rank], r_s, lam)
    out[f"within_lowrank{rank}"] = base_q + apply_ridge(Xqw[:, :rank], w)
    w = fit_ridge(Xsw, r_s, lam)
    out["within_ridge"] = base_q + apply_ridge(Xqw, w)
    w = fit_ridge(Xsw, r_s[perm], lam)
    out["within_ridge_shuffled"] = base_q + apply_ridge(Xqw, w)
    return out


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #
def episode(d: pd.DataFrame, ctx: str, budget: int, rep: int, E: np.ndarray,
            mu: np.ndarray, V: np.ndarray, Ec: np.ndarray, Vw: np.ndarray,
            lam: float, rank: int, seed: int) -> dict | None:
    # `hash(str)` is randomised per process unless PYTHONHASHSEED is fixed, so a recorded
    # integer seed did NOT reproduce the episodes. A stable digest of the context name does.
    ctx_id = int.from_bytes(hashlib.blake2b(ctx.encode(), digest_size=4).digest(), "big")
    # NOTE: `budget` is still part of the seed, so the support/query split is redrawn per
    # budget and the budget curves are not nested. Fixing that means seeding the split on
    # (seed, ctx, rep) only and nesting the support panels; it changes the episode
    # population, so it belongs in a rerun rather than a silent edit here.
    rng = np.random.default_rng((seed, ctx_id, budget, rep))
    dc = d[d.cond3 == ctx]
    alleles = dc.edit_key.unique()
    if len(alleles) < 50:
        return None
    # query alleles must offer a choice; support alleles are everything else
    gsize = dc.groupby("edit_key", observed=True).design_key.nunique()
    choosable = gsize[gsize >= 2].index.to_numpy()
    if len(choosable) < 40:
        return None
    rng.shuffle(choosable)
    n_q = max(len(choosable) // 2, 20)
    q_alleles = set(choosable[:n_q].tolist())
    s_alleles = np.array([a for a in alleles if a not in q_alleles])

    qry = dc[dc.edit_key.isin(q_alleles)]
    # Cap the query set so a single very deep context cannot dominate the runtime; the cap
    # is applied to whole decision groups, at the allele level, with the episode's seed.
    if qry.decision_group.nunique() > MAX_QUERY_GROUPS:
        keep = rng.choice(qry.decision_group.unique(), MAX_QUERY_GROUPS, replace=False)
        qry = qry[qry.decision_group.isin(set(keep.tolist()))]
    qry = qry.sort_values("decision_group", kind="stable")
    qry = qry.assign(_gcode=pd.factorize(qry.decision_group)[0])
    pool = dc[dc.edit_key.isin(set(s_alleles.tolist()))]
    if budget > len(pool):
        return None
    sup = pool.iloc[rng.choice(len(pool), budget, replace=False)] if budget else pool.iloc[:0]

    Xq = features(E, qry.row.to_numpy(), mu, V)
    Xqw = features_centred(Ec, qry.cpos.to_numpy(), Vw)
    if len(sup):
        Xs = features(E, sup.row.to_numpy(), mu, V)
        Xsw = features_centred(Ec, sup.cpos.to_numpy(), Vw)
    else:
        Xs = np.zeros((0, V.shape[0]), np.float32)
        Xsw = np.zeros((0, Vw.shape[0]), np.float32)
    res = {"context": ctx, "budget": budget, "rep": rep,
           "n_query_rows": int(len(qry)), "n_support_alleles": int(len(s_alleles))}
    for name, sc in arms(sup, qry, Xs, Xq, Xsw, Xqw, lam, rank, rng).items():
        m = group_metrics(qry, sc)
        if m["n_groups"] < MIN_QUERY_GROUPS:
            return None
        res[name] = m
    return res


def choose_hyper(d: pd.DataFrame, E: np.ndarray, mu: np.ndarray, V: np.ndarray,
                 Ec: np.ndarray, Vw: np.ndarray,
                 source_contexts: list[str], seed: int) -> tuple[float, int]:
    """Pick (lambda, rank) on source contexts only, at a mid-range budget.

    One choice is made per target context from that target's own 19 source contexts, which
    is why this is called inside the leave-context-out loop rather than once globally.
    """
    best, arg = -np.inf, (LAMBDAS[1], RANKS[0])
    for lam in LAMBDAS:
        for rank in RANKS:
            vals = []
            for ctx in source_contexts:
                e = episode(d, ctx, 48, 0, E, mu, V, Ec, Vw, lam, rank, seed + 7)
                if e is None:
                    continue
                vals.append(e[f"lowrank{rank}"]["accuracy"] - e["base"]["accuracy"])
            if vals and np.mean(vals) > best:
                best, arg = float(np.mean(vals)), (lam, rank)
    return arg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    d, E = load()
    Ec = allele_centred(E, d)
    contexts = sorted(d.cond3.unique())
    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "setup": {"budgets": list(BUDGETS), "reps": args.reps,
                     "n_pca": N_PCA, "lambda_grid": list(LAMBDAS),
                     "rank_grid": list(RANKS),
                     "n_dev_rows": int(len(d)), "contexts": contexts},
           "episodes": [], "hyper": {}}

    for target in contexts:
        src = [c for c in contexts if c != target]
        src_mask = (d.cond3 != target).to_numpy()
        mu, V = pca_basis(E, d.row.to_numpy()[src_mask], N_PCA, args.seed)
        Vw = pca_basis(Ec, d.cpos.to_numpy()[src_mask], N_PCA, args.seed)[1]
        lam, rank = choose_hyper(d[src_mask], E, mu, V, Ec, Vw, src[:3], args.seed)
        res["hyper"][target] = {"lambda": lam, "rank": rank}
        for budget in BUDGETS:
            for rep in range(args.reps if budget else 1):
                e = episode(d, target, budget, rep, E, mu, V, Ec, Vw, lam,
                            rank, args.seed)
                if e is not None:
                    e["lambda"], e["rank"] = lam, rank
                    res["episodes"].append(e)
        print(f"{target}: lam={lam} rank={rank}, episodes so far {len(res['episodes'])}",
              flush=True)

    res["summary"] = summarise(res["episodes"])
    C.write_outputs("e05_reference_panel", res, render(res))


ARM_ORDER = ["base", "intercept", "affine", "lowrank2", "lowrank4", "residual_ridge",
             "support_only_ridge", "shuffled_support",
             "within_lowrank2", "within_lowrank4", "within_ridge",
             "within_ridge_shuffled"]


def summarise(eps: list[dict]) -> dict:
    rows = []
    for e in eps:
        for arm in ARM_ORDER:
            if arm in e:
                rows.append({"context": e["context"], "budget": e["budget"],
                             "rep": e["rep"], "arm": arm, **e[arm]})
    t = pd.DataFrame(rows)
    if t.empty:
        return {}
    by_budget = (t.groupby(["arm", "budget"])
                  [["accuracy", "regret", "selected_efficiency", "mean_spearman_ge3",
                    "regret_reduction_vs_random"]]
                  .mean().reset_index())
    # paired against base within the same episode
    piv = t.pivot_table(index=["context", "budget", "rep"], columns="arm",
                        values=["accuracy", "regret"])
    paired = []
    for arm in ARM_ORDER[1:]:
        if ("accuracy", arm) not in piv.columns:
            continue
        for b in sorted(t.budget.unique()):
            sl = piv.xs(b, level="budget")
            da = (sl[("accuracy", arm)] - sl[("accuracy", "base")]).dropna()
            dr = (sl[("regret", arm)] - sl[("regret", "base")]).dropna()
            if len(da) == 0:
                continue
            paired.append({"arm": arm, "budget": int(b), "n_episodes": int(len(da)),
                           "d_accuracy": float(da.mean()),
                           "d_accuracy_sd_across_contexts": float(da.std()),
                           "d_regret": float(dr.mean()),
                           "contexts_improved": int((da > 0).sum())})
    return {"by_budget": by_budget.to_dict("records"), "paired_vs_base": paired,
            "per_context_at_max_budget": t[t.budget == max(t.budget)].to_dict("records")}


def render(r: dict) -> str:
    s = r.get("summary", {})
    L = ["# E05 - reference-panel adaptation on out-of-fold development episodes\n",
         "Frozen out-of-fold ordinal-S4D backbone. Support and query alleles are disjoint; "
         "every arm receives the same support labels; the PCA basis and (lambda, rank) come "
         "from source contexts only. Fold 0 is not used.\n",
         f"{r['setup']['n_dev_rows']:,} development candidate measurements, "
         f"{len(r['setup']['contexts'])} contexts, budgets {r['setup']['budgets']}, "
         f"{r['setup']['reps']} support draws each.\n"]
    if not s:
        L.append("No episode met the eligibility rules.\n")
        return "\n".join(L)
    bb = pd.DataFrame(s["by_budget"])
    L.append("## Selection quality by measurement budget\n")
    L.append("Mean over episodes of within-decision-group top-1 accuracy. **Read the "
             "paired table below instead for the comparison**: the rank is chosen per "
             "target context, so `lowrank2` and `lowrank4` are averaged over different "
             "context sets and their marginal means are not comparable with each other.\n")
    budgets = sorted(bb.budget.unique())
    L.append("| arm | " + " | ".join(f"B={b}" for b in budgets) + " |")
    L.append("|---" * (len(budgets) + 1) + "|")
    for arm in ARM_ORDER:
        sub = bb[bb.arm == arm].set_index("budget")
        if sub.empty:
            continue
        cells = [f"{sub.accuracy[b]:.3f}" if b in sub.index else "-" for b in budgets]
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("\n## Regret by measurement budget\n")
    L.append("| arm | " + " | ".join(f"B={b}" for b in budgets) + " |")
    L.append("|---" * (len(budgets) + 1) + "|")
    for arm in ARM_ORDER:
        sub = bb[bb.arm == arm].set_index("budget")
        if sub.empty:
            continue
        cells = [f"{sub.regret[b]:.4f}" if b in sub.index else "-" for b in budgets]
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("\n## Paired against no adaptation, within episode\n")
    L.append("| arm | budget | episodes | delta accuracy | SD across episodes | delta regret | episodes improved |\n|---|---:|---:|---:|---:|---:|---:|")
    for p in s["paired_vs_base"]:
        L.append(f"| {p['arm']} | {p['budget']} | {p['n_episodes']} | {p['d_accuracy']:+.4f} | "
                 f"{p['d_accuracy_sd_across_contexts']:.4f} | {p['d_regret']:+.5f} | "
                 f"{p['contexts_improved']}/{p['n_episodes']} |")
    L.append("\nA per-context intercept cannot change the order of candidates inside a "
             "decision group, so its row is exactly zero at every budget. That is an "
             "arithmetic check on the pipeline, not a result: **a calibration success is "
             "not a design-transfer success.** The affine arm is exactly zero too wherever "
             "its fitted slope is positive, and differs from the base at B=12 only because "
             "in one episode of eighty the least-squares slope on twelve support points "
             "came out negative and inverted the ranking. That is worth keeping in view: at "
             "the smallest budgets even a two-parameter recalibration can actively damage a "
             "ranking it was supposed to leave alone.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
