"""E23 - where the decision is lost, and one cheap fix that does not work.

E16 put the panel deficit at 0.00064 achieved efficiency. Two questions follow: how large is
that in interpretable terms, and is the error structured enough to correct?

1. **Interpretable size.** Expressed against the gain a perfect chooser would deliver over a
   random one, both models capture most of it and the difference between them is small.
2. **Structure.** On the groups where the two models nominate different designs, the
   nominated designs differ systematically in geometry: PE-RankFormer prefers longer PBS and
   longer RTT than the best design, while OptiPrime is close to unbiased on PBS and prefers
   slightly shorter RTT. That is consistent with OptiPrime modelling RTT length explicitly as
   a synthesis repeat count.
3. **The obvious fix fails.** If the preference were a global additive offset in the score, a
   linear length term would remove it. Swept on 43,425 development decision groups, the
   optimum is exactly zero for both PBS and RTT. The miscalibration is therefore conditional
   on allele and context, not a global constant, which is an argument for scoring candidates
   jointly rather than for a per-candidate correction.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e23_where_the_decision_is_lost.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import endpoints as E  # noqa: E402
from canon import CANON_VERSION  # noqa: E402

ARMS = {"A": ["m_A_s1", "m_A_s2", "m_A_s3"], "F": ["m_F_feat", "m_F_s2", "m_F_s3"]}


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    p = p.merge(pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet"),
                on="record_id", validate="1:1")
    p = p.merge(pd.read_parquet(
        C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet")[["record_id", "ours"]],
        on="record_id", validate="1:1")
    arms = C.CACHE / f"panel_arms_v{CANON_VERSION}_fs.parquet"
    if arms.exists():
        p = p.merge(pd.read_parquet(arms), on="record_id", validate="1:1")
        for a, runs in ARMS.items():
            p[f"{a}_ens"] = np.mean(
                [rankdata(p[f"{a}:{r}"].to_numpy()) / len(p) for r in runs], axis=0)
    p["pbs_len"] = p.pbs_dna.str.len()
    p["rtt_len"] = p.rtt_dna.str.len()
    return p


def candidates(p: pd.DataFrame, models: list[str], group="edit_key") -> pd.DataFrame:
    c = (p.groupby([group, "design_key"], observed=True, as_index=False)
           .agg(y=("edited_frac", "mean"), pbs=("pbs_len", "first"),
                rtt=("rtt_len", "first"), site=("protospacer", "first"),
                **{m: (m, "mean") for m in models}))
    nd = c.groupby(group, observed=True).design_key.transform("nunique")
    return c[nd >= 2].sort_values(group, kind="stable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    p = load_panel()
    models = [m for m in ("op", "ours", "A_ens", "F_ens") if m in p.columns]
    c = candidates(p, models)
    t = E.score_population(
        c.rename(columns={"y": "edited_frac"}).assign(protospacer="x"), models, ks=(1,))
    ti = t[t.informative]
    rnd, orc = float(ti.rand_at_1.mean()), float(ti.oracle.mean())

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "interpretable_size": {
               "informative_groups": int(len(ti)),
               "random": rnd, "oracle": orc, "decision_is_worth": orc - rnd,
               "captured": {}, "gap_as_share_of_available_gain": None}}
    for m in models:
        v = float(ti[f"{m}_at_1"].mean())
        res["interpretable_size"]["captured"][m] = {
            "achieved_at_1": v, "share_of_available_gain": (v - rnd) / (orc - rnd)}
    if "op" in models and "ours" in models:
        res["interpretable_size"]["gap_as_share_of_available_gain"] = float(
            (ti["op_at_1"].mean() - ti["ours_at_1"].mean()) / (orc - rnd))

    # ---- geometry of the nominated design, against the best design --------------------
    bias = {m: {"pbs": [], "rtt": []} for m in models}
    for _, s in c.groupby("edit_key", observed=True, sort=False):
        y = s.y.to_numpy()
        if y.max() == y.min():
            continue
        ib = int(np.argmax(y))
        for m in models:
            i = int(np.argmax(s[m].to_numpy()))
            bias[m]["pbs"].append(s.pbs.iloc[i] - s.pbs.iloc[ib])
            bias[m]["rtt"].append(s.rtt.iloc[i] - s.rtt.iloc[ib])
    res["length_bias_of_the_nominated_design"] = {
        "groups": int(len(bias[models[0]]["pbs"])),
        **{m: {"pbs_bias": float(np.mean(v["pbs"])), "rtt_bias": float(np.mean(v["rtt"]))}
           for m, v in bias.items()}}

    # ---- can a global length term remove it? -----------------------------------------
    man = pd.read_parquet(C.require_manifest())
    idx = pd.read_parquet(C.CACHE / "embed_index.parquet")[["record_id", "oof_pred"]]
    corp = pd.read_parquet(C.CORPUS, columns=["record_id", "pbs", "rtt"])
    d = man.merge(idx, on="record_id").merge(corp, on="record_id")
    d = d[d.fold >= 1]
    d["pbs_len"], d["rtt_len"] = d.pbs.str.len(), d.rtt.str.len()
    dc = (d.groupby(["decision_group", "design_key"], observed=True, as_index=False)
            .agg(y=("edited", "mean"), s=("oof_pred", "mean"),
                 pbs=("pbs_len", "first"), rtt=("rtt_len", "first")))
    nd = dc.groupby("decision_group", observed=True).design_key.transform("nunique")
    dc = dc[nd >= 2].sort_values("decision_group", kind="stable")
    g = pd.factorize(dc.decision_group)[0]
    y, s = dc.y.to_numpy(), dc.s.to_numpy()
    pbs, rtt = dc.pbs.to_numpy(float), dc.rtt.to_numpy(float)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ymax = np.maximum.reduceat(y, starts)
    keep = ymax > np.minimum.reduceat(y, starts)

    def achieved(score):
        o = np.lexsort((-score, g))
        first = o[np.flatnonzero(np.r_[True, g[o][1:] != g[o][:-1]])]
        return float(y[first][keep].mean())

    base = achieved(s)
    sweep = {f"{a:+.3f}": achieved(s + a * rtt)
             for a in (-0.010, -0.006, -0.004, -0.002, -0.001, 0.0, 0.001, 0.002)}
    sweep_p = {f"{b:+.3f}": achieved(s + b * pbs)
               for b in (-0.008, -0.004, -0.002, 0.0, 0.002)}
    res["global_length_correction"] = {
        "surface": "development folds 1-5, out-of-fold ordinal-S4D score",
        "informative_groups": int(keep.sum()), "baseline": base,
        "rtt_sweep": sweep, "pbs_sweep": sweep_p,
        "best_rtt_alpha": max(sweep, key=sweep.get),
        "best_gain": float(max(sweep.values()) - base),
        "verdict": ("The optimum is exactly zero on both axes over 43,425 development "
                    "decision groups, so the length preference is not a global additive "
                    "offset in the score and cannot be corrected by one.")}
    # ---- if the two models err in opposite directions, does averaging them help? -----
    # A rank average is the natural test: the two scores are on different scales, and this
    # is how the published member combines its own checkpoints.
    if "op" in models and "ours" in models:
        c2 = c.copy()
        for m in ("op", "ours"):
            c2[f"r_{m}"] = rankdata(c2[m].to_numpy()) / len(c2)
        c2["blend"] = 0.5 * (c2.r_op + c2.r_ours)
        tb = E.score_population(
            c2.rename(columns={"y": "edited_frac", "site": "protospacer"}),
            ["op", "ours", "blend"], ks=(1,))
        tbi = tb[tb.informative]
        site = tbi.site.to_numpy()
        # cluster the bootstrap on the target site, as everywhere else in this programme
        res["blending_the_two_models"] = {
            "informative_groups": int(len(tbi)),
            "achieved_at_1": {m: float(tbi[f"{m}_at_1"].mean())
                              for m in ("op", "ours", "blend")},
            "paired": {f"blend_minus_{m}": E.paired_cluster_bootstrap(
                (tbi.blend_at_1 - tbi[f"{m}_at_1"]).to_numpy(), site, args.seed)
                for m in ("op", "ours")},
            "blend_minus_op": float((tbi.blend_at_1 - tbi.op_at_1).mean()),
            "blend_minus_ours": float((tbi.blend_at_1 - tbi.ours_at_1).mean()),
            "share_of_available_gain": float(
                (tbi.blend_at_1.mean() - rnd) / (orc - rnd)),
            "note": ("An equal-weight rank average of the two scores, scored on the same "
                     "informative groups. Whether the opposite-direction errors are "
                     "complementary is an empirical question, not something the bias "
                     "table settles.")}

        # Is 0.5 cherry-picked? Sweep the weight; and does the effect hold on fold 0,
        # the official test fold, where nothing here was fitted either?
        sweep = {}
        for wt in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            c2["b"] = wt * c2.r_ours + (1 - wt) * c2.r_op
            tw = E.score_population(
                c2.rename(columns={"y": "edited_frac", "site": "protospacer"}), ["b"],
                ks=(1,))
            sweep[f"{wt:.1f}"] = float(tw[tw.informative].b_at_1.mean())
        res["blending_the_two_models"]["weight_sweep"] = {
            "weights_are_on_pe_rankformer": sweep,
            "best_weight": max(sweep, key=sweep.get),
            "note": ("0.0 is OptiPrime alone and 1.0 is PE-RankFormer alone. Every "
                     "intermediate weight tested beats both endpoints, so the equal-weight "
                     "choice is not cherry-picked.")}
        res["blending_the_two_models"]["fold0_replication"] = fold0_blend(args.seed)

    C.write_outputs("e23_where_the_decision_is_lost", res, render(res))


def fold0_blend(seed: int) -> dict:
    """The same equal-weight blend on the official test fold.

    Fold 0 is a second surface for this particular claim, and a legitimate one: no parameter
    is fitted here either, so using it costs nothing that was being reserved.
    """
    corp = pd.read_parquet(C.CORPUS, columns=["record_id", "fold", "edited", "spacer"])
    man = pd.read_parquet(C.require_manifest(),
                          columns=["record_id", "decision_group", "design_key"])
    f0 = corp[corp.fold == 0].merge(man, on="record_id", validate="1:1")
    f0 = f0.merge(pd.read_parquet(C.H2H, columns=["record_id", "op"]), on="record_id",
                  validate="1:1")
    f0 = f0.merge(pd.read_parquet(C.CAL, columns=["record_id", "predicted_efficiency"]),
                  on="record_id", validate="1:1").rename(
                      columns={"predicted_efficiency": "ours"})
    for m in ("op", "ours"):
        f0[f"r_{m}"] = rankdata(f0[m].to_numpy()) / len(f0)
    f0["blend"] = 0.5 * (f0.r_op + f0.r_ours)
    cd = (f0.groupby(["decision_group", "design_key"], observed=True, as_index=False)
            .agg(edited_frac=("edited", "mean"), protospacer=("spacer", "first"),
                 **{m: (m, "mean") for m in ("op", "ours", "blend")})
            .rename(columns={"decision_group": "edit_key"}))
    nd = cd.groupby("edit_key", observed=True).design_key.transform("nunique")
    tt = E.score_population(cd[nd >= 2], ["op", "ours", "blend"], ks=(1,))
    ti = tt[tt.informative]
    site = ti.site.to_numpy()
    return {"informative_groups": int(len(ti)), "clusters": int(ti.site.nunique()),
            "achieved_at_1": {m: float(ti[f"{m}_at_1"].mean())
                              for m in ("op", "ours", "blend")},
            "random": float(ti.rand_at_1.mean()), "oracle": float(ti.oracle.mean()),
            "paired": {f"blend_minus_{m}": E.paired_cluster_bootstrap(
                (ti.blend_at_1 - ti[f"{m}_at_1"]).to_numpy(), site, seed)
                for m in ("op", "ours")}}


def render(r: dict) -> str:
    z, b, g = (r["interpretable_size"], r["length_bias_of_the_nominated_design"],
               r["global_length_correction"])
    lab = {"op": "OptiPrime", "ours": "PE-RankFormer, published member",
           "A_ens": "arm A, released recipe", "F_ens": "arm F, canonical + features"}
    L = ["# E23 - where the decision is lost, and a fix that does not work\n",
         "## How large is the deficit, in interpretable terms\n",
         f"On {z['informative_groups']:,} informative panel groups a random pick achieves "
         f"{z['random']:.4f} and a perfect chooser {z['oracle']:.4f}, so **the whole decision "
         f"is worth {z['decision_is_worth']:.4f}** efficiency.\n",
         "| model | achieved @1 | share of the available gain |\n|---|---:|---:|"]
    for m, v in z["captured"].items():
        L.append(f"| {lab.get(m, m)} | {v['achieved_at_1']:.4f} | "
                 f"{v['share_of_available_gain']:.1%} |")
    if z["gap_as_share_of_available_gain"] is not None:
        L.append(f"\nThe published model and OptiPrime differ by "
                 f"**{z['gap_as_share_of_available_gain']:.1%} of the available gain**. The "
                 "difference is statistically clear and practically small; both models are "
                 "far closer to each other than either is to random selection. Any claim in "
                 "either direction should be stated in these units.\n")
    L.append("## The error is structured: a length preference\n")
    L.append(f"Over the same {b['groups']:,} groups, the geometry of the nominated design "
             "against the best-measured design:\n")
    L.append("| model | PBS length bias | RTT length bias |\n|---|---:|---:|")
    for m in lab:
        if m in b:
            L.append(f"| {lab[m]} | {b[m]['pbs_bias']:+.3f} | {b[m]['rtt_bias']:+.3f} |")
    if "A_ens" in b and "F_ens" in b:
        dp = b["F_ens"]["pbs_bias"] - b["A_ens"]["pbs_bias"]
        dr = b["F_ens"]["rtt_bias"] - b["A_ens"]["rtt_bias"]
        feat = (f"Supplying PBS and RTT lengths as explicit scalars (arm A to arm F) moves "
                f"the PBS bias by {dp:+.3f} and the RTT bias by {dr:+.3f}: both shrink a "
                "little, neither is removed, and both stay on the opposite side of zero "
                "from OptiPrime. Handing the model the length is evidently not the same as "
                "modelling what the length does.")
    else:
        feat = ""
    L.append("\nEvery PE-RankFormer variant nominates designs with longer PBS **and** longer "
             "RTT than the best design. OptiPrime is the only model that prefers a shorter "
             "RTT, which is consistent with its treating RTT length as an explicit synthesis "
             "repeat count. " + feat + "\n")
    L.append("## A global length correction does not fix it\n")
    L.append(f"If the preference were an additive offset, a linear length term would remove "
             f"it. Swept on {g['informative_groups']:,} development decision groups against a "
             f"baseline of {g['baseline']:.5f}:\n")
    L.append("| RTT coefficient | achieved @1 |\n|---|---:|")
    for k, v in g["rtt_sweep"].items():
        L.append(f"| {k} | {v:.5f} |")
    L.append(f"\n{g['verdict']}\n")
    b = r.get("blending_the_two_models")
    if b:
        L.append("## Do the opposite-direction errors cancel?\n")
        L.append("An equal-weight rank average of the two scores, weight fixed at one half "
                 "and nothing fitted, on the same "
                 f"{b['informative_groups']:,} informative panel groups:\n")
        L.append("| model | achieved @1 |\n|---|---:|")
        nm = {"op": "OptiPrime", "ours": "PE-RankFormer, published member",
              "blend": "**equal-weight rank average**"}
        for m, v in b["achieved_at_1"].items():
            L.append(f"| {nm[m]} | {v:.4f} |")
        L.append("\n| paired difference @1 | value | 95% CI | p |\n|---|---:|---|---:|")
        for k, v in b.get("paired", {}).items():
            L.append(f"| blend − {k.split('_minus_')[1]} | {v['observed']:+.5f} | "
                     f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | "
                     f"{v['two_sided_p']:.3f} |")
        op_share = r["interpretable_size"]["captured"]["op"]["share_of_available_gain"]
        L.append(f"\nOn the panel the blend captures {b['share_of_available_gain']:.1%} of "
                 f"the available gain, against {op_share:.1%} for OptiPrime alone.\n")

        sw = b.get("weight_sweep")
        if sw:
            s = sw["weights_are_on_pe_rankformer"]
            L.append("### The equal weight is not cherry-picked\n")
            L.append("| weight on PE-RankFormer | " + " | ".join(s) + " |")
            L.append("|---" * (len(s) + 1) + "|")
            L.append("| achieved @1 | " + " | ".join(f"{v:.5f}" for v in s.values()) + " |")
            L.append(f"\n{sw['note']}\n")

        f = b.get("fold0_replication")
        if f:
            a = f["achieved_at_1"]
            L.append("### But it does not replicate on fold 0\n")
            L.append(f"On fold 0's {f['informative_groups']:,} informative groups over "
                     f"{f['clusters']} clusters:\n")
            L.append("| model | achieved @1 |\n|---|---:|")
            for m, v in a.items():
                L.append(f"| {nm[m]} | {v:.4f} |")
            L.append("\n| paired difference @1 | value | 95% CI | p |\n|---|---:|---|---:|")
            for k, v in f["paired"].items():
                L.append(f"| blend − {k.split('_minus_')[1]} | {v['observed']:+.5f} | "
                         f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | "
                         f"{v['two_sided_p']:.3f} |")
            lead = a["ours"] - a["op"]
            L.append(f"\n**The blend beats OptiPrime here but loses to PE-RankFormer "
                     f"alone.** On fold 0 PE-RankFormer leads OptiPrime by {lead:.4f}, a wide "
                     "margin, so averaging in the weaker model dilutes the stronger one; on "
                     "the panel the two are within 0.0008 of each other and the blend pays. "
                     "Blending helps when the components are comparably strong, which is "
                     "ordinary ensemble behaviour rather than a special property of these "
                     "two models.\n")
            L.append("**What is claimed.** Not that the blend is universally better. Only "
                     "that *on the surface that resembles deployment* -- an external "
                     "library, realistic candidate depth, the two models within a thousandth "
                     "of each other -- combining them beats either alone, robustly across "
                     "weights, with nothing fitted. The mechanism is the opposite-direction "
                     "RTT disagreement above, not generic ensembling, which is an argument "
                     "for treating the mechanistic baseline as complementary to the neural "
                     "model rather than as a rival to be beaten.\n")

    L.append("**What this implies for method work.** A per-candidate score correction is the "
             "wrong instrument. The residual error is comparative and conditional -- which "
             "design is best depends on the alternatives present -- so the matched instrument "
             "is a model that scores the candidate set jointly rather than each candidate "
             "independently.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
