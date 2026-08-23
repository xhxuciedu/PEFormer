"""Repeated-design multi-context dataset and diagnostics (round-6 spec §7, §30, §31).

Round 5's diagnosis: the model scores designs as if s(d,c) ~ g(d) + a(c), a universal
design quality plus a context-level shift. The data require an interaction term
h(d,c) that reorders designs between contexts (true cross-context rank correlation
0.683 vs the model's 0.835). Everything in round 6 is supervised on that interaction,
so it first has to be extracted cleanly.

**Splitting design from context is the load-bearing decision here**, and getting it
wrong silently destroys the signal. Round 5 already showed one failure of exactly this
kind: grouping on (spacer, rtt, pbs, cell, PE, Cas9) looked like it produced 21,994
replicate groups, but those rows differed in `motif`, `epegRNA` and `linker` -- they
were epegRNA versus plain-pegRNA constructs, i.e. *different designs*, and treating
them as the same design would have manufactured spurious "context effects" that are
really construct effects.

So:

  DESIGN  = the physical pegRNA and its target: spacer, RTT, PBS, target site,
            scaffold, 3' motif, linker, epegRNA flag.
  CONTEXT = the experiment run on it: source study, cell line, PE system, Cas9
            variant and PAM, RT variant, PEmax/MLH1dn/NRCH background, timepoint.

`epegRNA`, `motif`, `linker` and `scaffold_name` are assigned to DESIGN because they
change the molecule. `rt_name`, `cas9_type`, `cas9_pam`, `PEmax`, `MLH1dn`, `NRCH` are
assigned to CONTEXT because they change the machinery acting on it.

Outputs
-------
data/processed/round6_repeated_designs.parquet : one row per (design, context)
    observation for every design seen in >= 2 contexts, with global and
    context-relative ranks.
data/processed/round6_context_pairs.parquet    : one row per (design, context_a,
    context_b) with the observed rank shift -- the direct supervision target.
results/round6/context_pair_diagnostics.csv    : per context-pair true vs predicted
    cross-context correlation and excess invariance (§30).
results/round6/context_identifiability.csv     : support per context (§31).
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("round6_interaction")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
DEV = Path("data/processed/round3_dev_assignments.parquet")
LOCKBOX = Path("data/processed/round4_lockbox.parquet")
PRED = Path("results/round3/dev_recalibration")

DESIGN_KEY = ["spacer", "rtt", "pbs", "target_name", "scaffold_name", "motif",
              "linker", "epegRNA"]
CONTEXT_KEY = ["source_study", "cell_type", "pe_type", "cas9_type", "cas9_pam",
               "rt_name", "PEmax", "MLH1dn", "NRCH", "time"]
MIN_CONTEXT_ROWS = 200


def main() -> None:
    d = pd.read_parquet(CORPUS)
    d = d[d.fold != 0].copy()          # held-out is never part of the training-side dataset
    logger.info("training-pool rows: %d", len(d))

    d["design_id"] = d.groupby(DESIGN_KEY, dropna=False).ngroup()
    d["context_id"] = d.groupby(CONTEXT_KEY, dropna=False).ngroup()
    d["context_label"] = (d.cell_type.astype(str) + "/" + d.pe_type.astype(str)
                          + "/" + d.source_study.astype(str))

    # Global rank, and rank *within* each context. The context-relative rank is the
    # quantity the interaction objectives supervise: it removes a(c) by construction.
    d["rank_global"] = rankdata(d.edited.to_numpy()) / len(d)
    d["rank_in_context"] = d.groupby("context_id").edited.transform(
        lambda s: rankdata(s.to_numpy()) / len(s))
    # Dense and average ranks kept for the tie diagnostics the spec asks for (§8).
    d["rank_in_context_dense"] = d.groupby("context_id").edited.transform(
        lambda s: rankdata(s.to_numpy(), method="dense") / max(s.nunique(), 1))

    n_ctx = d.groupby("design_id").context_id.nunique()
    rep_ids = n_ctx[n_ctx >= 2].index
    rep = d[d.design_id.isin(rep_ids)].copy()
    logger.info("designs: %d total, %d seen in >=2 contexts (%d observations)",
                d.design_id.nunique(), len(rep_ids), len(rep))
    logger.info("context groups: %d", d.context_id.nunique())

    # Fold/lockbox provenance so downstream training can enforce disjointness.
    dev = pd.read_parquet(DEV)
    for c in [c for c in dev.columns if c.startswith("round3_dev_fold")]:
        rep = rep.merge(dev[["record_id", c]], on="record_id", how="left")
    lb = set(pd.read_parquet(LOCKBOX).record_id)
    rep["in_lockbox"] = rep.record_id.isin(lb)

    keep = (["record_id", "design_id", "context_id", "context_label", "edited",
             "rank_global", "rank_in_context", "rank_in_context_dense", "fold",
             "in_lockbox"] + DESIGN_KEY + CONTEXT_KEY)
    out = rep[keep]
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/processed/round6_repeated_designs.parquet", index=False)
    logger.info("wrote data/processed/round6_repeated_designs.parquet (%d rows)", len(out))

    # ---- context pairs: the direct supervision target -----------------------------
    logger.info("building context pairs ...")
    pieces = []
    for did, sub in rep.groupby("design_id"):
        sub = sub.drop_duplicates("context_id")
        if len(sub) < 2:
            continue
        recs = sub.to_dict("records")
        for a, b in combinations(recs, 2):
            pieces.append((did, a["context_id"], b["context_id"],
                           a["record_id"], b["record_id"],
                           a["rank_in_context"], b["rank_in_context"],
                           a["rank_in_context"] - b["rank_in_context"],
                           a["edited"], b["edited"], a["fold"], b["fold"],
                           a["context_label"], b["context_label"]))
    pairs = pd.DataFrame(pieces, columns=[
        "design_id", "context_a", "context_b", "record_a", "record_b",
        "rank_a", "rank_b", "delta_rank", "edited_a", "edited_b",
        "fold_a", "fold_b", "label_a", "label_b"])
    pairs.to_parquet("data/processed/round6_context_pairs.parquet", index=False)
    logger.info("wrote data/processed/round6_context_pairs.parquet (%d pairs)", len(pairs))
    logger.info("  |delta_rank| mean %.4f, sd %.4f; %.1f%% exceed 0.1",
                pairs.delta_rank.abs().mean(), pairs.delta_rank.std(),
                100 * (pairs.delta_rank.abs() > 0.1).mean())

    # ---- §31 context identifiability ----------------------------------------------
    ident = d.groupby(["context_id", "context_label"]).agg(
        n_rows=("record_id", "size"),
        n_protospacers=("spacer", "nunique"),
        n_designs=("design_id", "nunique"),
        pct_zero=("edited", lambda s: 100 * float((s == 0).mean())),
        mean_eff=("edited", "mean"),
    ).reset_index()
    rep_per_ctx = rep.groupby("context_id").design_id.nunique().rename("n_repeated_designs")
    ident = ident.merge(rep_per_ctx, on="context_id", how="left").fillna({"n_repeated_designs": 0})
    ident = ident.sort_values("n_rows", ascending=False)
    Path("results/round6").mkdir(parents=True, exist_ok=True)
    ident.to_csv("results/round6/context_identifiability.csv", index=False)
    logger.info("wrote results/round6/context_identifiability.csv (%d contexts)", len(ident))
    small = ident[ident.n_rows < MIN_CONTEXT_ROWS]
    logger.info("  contexts with <%d rows: %d (%.1f%% of rows) -- these need shrinkage, "
                "not independent heads", MIN_CONTEXT_ROWS, len(small),
                100 * small.n_rows.sum() / ident.n_rows.sum())

    # ---- §30 context-pair diagnostics, using the current model's dev predictions ----
    logger.info("computing context-pair diagnostics against the Ordinal-SSM baseline ...")
    pr = pd.concat([pd.read_parquet(PRED / f"predictions_r4p2_ordSSM_oof_round3_dev_fold_{i}.parquet")
                    for i in range(3)]).drop_duplicates("record_id")
    md = d[["record_id", "design_id", "context_label"]]
    j = pr.merge(md, on="record_id")
    rows = []
    for ca, cb in combinations(sorted(j.context_label.unique()), 2):
        A = j[j.context_label == ca].drop_duplicates("design_id").set_index("design_id")
        B = j[j.context_label == cb].drop_duplicates("design_id").set_index("design_id")
        sh = sorted(A.index.intersection(B.index))
        if len(sh) < 200:
            continue
        A, B = A.loc[sh], B.loc[sh]
        t = spearmanr(A.true_efficiency.values, B.true_efficiency.values).statistic
        p = spearmanr(A.predicted_efficiency.values, B.predicted_efficiency.values).statistic
        rows.append({"context_a": ca, "context_b": cb, "n_shared_designs": len(sh),
                     "true_cross_rho": t, "pred_cross_rho": p, "excess_invariance": p - t})
    diag = pd.DataFrame(rows).sort_values("excess_invariance", ascending=False)
    diag.to_csv("results/round6/context_pair_diagnostics.csv", index=False)
    logger.info("wrote results/round6/context_pair_diagnostics.csv (%d pairs)", len(diag))
    if len(diag):
        logger.info("  mean true %.4f | mean pred %.4f | mean excess %+.4f",
                    diag.true_cross_rho.mean(), diag.pred_cross_rho.mean(),
                    diag.excess_invariance.mean())
        logger.info("  worst pairs by excess invariance:")
        for _, r in diag.head(5).iterrows():
            logger.info("    %-28s vs %-28s n=%4d true=%.3f pred=%.3f excess=%+.3f",
                        r.context_a, r.context_b, r.n_shared_designs,
                        r.true_cross_rho, r.pred_cross_rho, r.excess_invariance)


if __name__ == "__main__":
    main()
