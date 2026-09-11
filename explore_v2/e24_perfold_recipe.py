"""E24 - the improved recipe, built the way the published model is built.

E22 left one question open. Arm F beats its matched control on the panel (+0.00140,
p = 0.001) yet only draws level with the PUBLISHED member (+0.00003, p = 0.75). The two
things being compared were not built alike: the published `ordSSM` member is five
checkpoints, one per official fold, each trained on a different four-fifths of the
development data; arm F was three seeds trained on identical rows, differing only in
initialisation. Five per-fold checkpoints are a far more diverse ensemble than three
same-data seeds, so the E22 contrast confounds recipe with ensemble construction.

This removes the confound: arm F rebuilt as five per-fold checkpoints (`--val-fold 1..5`,
canonical batching, canonical ranking key, Family C feature branch), combined by the same
plain mean of predicted efficiencies used to form the published member. Only then does
"does the improved recipe beat the published model" have a clean answer.

Both surfaces are evaluated because the manuscript makes claims on both. Fold 0 is the
official test fold and no checkpoint here trains or early-stops on it. The isotonic
calibrator is refitted from scratch on THIS model's development out-of-fold predictions and
frozen before touching either surface, exactly as the published one was.

**Disclosure:** a further development use of the reserved panel, whose single pre-declared
confirmatory comparison was spent in E11. Nothing here is confirmatory.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 .venv/bin/python explore_v2/e24_perfold_recipe.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import featstats  # noqa: E402
import endpoints as E  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from e22_best_arm_on_panel import (attach_family_c_features_for_panel,  # noqa: E402
                                   score)
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import featurize  # noqa: E402

FOLDS = (1, 2, 3, 4, 5)
RUNS = {k: f"m_Ffold_cv{k}" for k in FOLDS}
ARM_F3 = ["m_F_feat", "m_F_s2", "m_F_s3"]


def ckpt_dir(run: str) -> Path:
    dirs = sorted((ROOT / "checkpoints").glob(f"{run}_*"))
    if not dirs:
        raise FileNotFoundError(f"no checkpoint directory for {run}")
    return dirs[-1]


@torch.no_grad()
def score_rows(ckpt: Path, corpus, corpus_feat, idx: np.ndarray, dev: str, bs: int):
    """Score a subset of rows. `score` in E22 always scores the whole corpus, which is
    wasteful when only two folds of 318k rows are wanted."""
    from pe_rankformer.data.dataset import PEDataset, collate
    from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = PERankFormerConfig(**ck["model_config"])
    m = PERankFormer(cfg).to(dev).eval()
    m.load_state_dict(ck["model_state_dict"])
    ds = PEDataset(corpus_feat if cfg.n_features else corpus)
    out = []
    for s in range(0, len(idx), bs):
        b = {k: v.to(dev) for k, v in collate([ds[i] for i in idx[s:s + bs]]).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = m(b)
        out.append(m.efficiency_from_output(o).float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def prep_dev(train_folds):
    """The official featurized corpus with Family C features frozen to `train_folds`.

    The development corpus names its columns `spacer`/`pbs`/`rtt`, not the panel's
    `protospacer`/`pbs_dna`/`rtt_dna`, and its Family C features are already computed in
    `data/processed/family_c_features.parquet` -- so this path reuses that file rather than
    recomputing features through the panel helper.
    """
    from pe_rankformer.data.dataset import load_featurized
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    npz = str(ROOT / "data/processed/featurized_official.npz")
    corpus = load_featurized(npz, vocab)
    feat = pd.read_parquet(featstats.DEV_FEATURES)
    corpus_feat = featstats.attach_frozen(load_featurized(npz, vocab), feat,
                                          *featstats.train_stats(C.CORPUS, train_folds))
    return corpus, corpus_feat


def prep(df: pd.DataFrame, train_folds):
    """featurize a frame the way the trainer does, with FROZEN feature standardisation.

    `train_folds` are the development folds the scoring checkpoint was trained on; its
    feature mean and SD are recovered from them and applied unchanged here, because
    refitting on the evaluation surface would rescale the inputs away from the scale the
    model learned to read (see explore_v2/featstats.py).
    """
    d = df.copy()
    d["edited"] = d.edited_frac
    d["indel"] = 0.0
    d["fold"] = 0
    d["target_name"] = d.protospacer
    d["scaffold_name"] = d.scaffold_name.astype(str)
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = featurize(d, vocab)
    stats = featstats.train_stats(C.CORPUS, train_folds)
    return corpus, attach_family_c_features_for_panel(d, corpus, stats=stats)


def cluster_corr(y, pa, pb, cluster, seed, n_boot=2000, kind="spearman"):
    fn = spearmanr if kind == "spearman" else pearsonr
    obs = float(fn(pa, y).statistic - fn(pb, y).statistic)
    uniq, inv = np.unique(cluster, return_inverse=True)
    buckets = [np.flatnonzero(inv == j) for j in range(len(uniq))]
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([buckets[j] for j in rng.integers(0, len(uniq), len(uniq))])
        v[b] = fn(pa[idx], y[idx]).statistic - fn(pb[idx], y[idx]).statistic
    lo, hi = np.percentile(v, [2.5, 97.5])
    frac = float((v > 0).mean())
    return {"observed": obs, "ci95": [float(lo), float(hi)],
            "two_sided_p": float(max(2 * min(frac, 1 - frac), 1.0 / n_boot)),
            "n_clusters": int(len(uniq))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ panel ---------
    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    panel = panel.merge(pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet"),
                        on="record_id", validate="1:1")
    panel = panel.merge(pd.read_parquet(
        C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet")[["record_id", "ours"]],
        on="record_id", validate="1:1")
    panel = panel.merge(pd.read_parquet(C.CACHE / f"panel_arms_v{CANON_VERSION}_fs.parquet"),
                        on="record_id", validate="1:1")
    # the three-seed ensembles from E22, for context: arm F was the pre-specified choice,
    # but E22 found arms E and H ahead of it on this surface
    THREE_SEED = {"F3_ens": ARM_F3,
                  "A_ens": ["m_A_s1", "m_A_s2", "m_A_s3"],
                  "E_ens": ["m_E_norank", "m_E_s2", "m_E_s3"],
                  "H_ens": ["m_H_s1", "m_H_s2", "m_H_s3"]}
    arm_of = {"F3_ens": "F", "A_ens": "A", "E_ens": "E", "H_ens": "H"}
    for name, runs in THREE_SEED.items():
        cols = [f"{arm_of[name]}:{r}" for r in runs]
        if not all(c in panel.columns for c in cols):
            continue
        panel[name] = np.mean(
            [rankdata(panel[c].to_numpy()) / len(panel) for c in cols], axis=0)

    cache = C.CACHE / f"panel_perfold_F_v{CANON_VERSION}.parquet"
    if cache.exists():
        got = pd.read_parquet(cache)
    else:
        cols = {"record_id": panel.record_id.to_numpy()}
        for k, run in RUNS.items():
            corpus, corpus_feat = prep(panel, [f for f in FOLDS if f != k])
            cols[f"Ffold_cv{k}"] = score(ckpt_dir(run) / "best.pt", corpus, corpus_feat,
                                         dev, args.batch_size)
            print(f"panel: scored {run}", flush=True)
        got = pd.DataFrame(cols)
        got.to_parquet(cache, index=False)
    panel = panel.merge(got, on="record_id", validate="1:1")
    memb = [f"Ffold_cv{k}" for k in FOLDS]
    # combined the same way the published member is: plain mean of predicted efficiencies
    panel["Ffold"] = panel[memb].mean(axis=1)

    # ------------------------------- development OOF, for the frozen calibrator -------
    corp = pd.read_parquet(C.CORPUS)
    man = pd.read_parquet(C.require_manifest(),
                          columns=["record_id", "decision_group", "design_key", "spacer"])
    oof_cache = C.CACHE / f"dev_oof_perfold_F_v{CANON_VERSION}.parquet"
    if oof_cache.exists():
        oof = pd.read_parquet(oof_cache)
    else:
        rows, f0rows = [], []
        for k in FOLDS:
            cf, cff = prep_dev([f for f in FOLDS if f != k])
            fold = np.asarray(cf.fold)
            rid = np.asarray(cf.record_id)
            ik = np.flatnonzero(fold == k)
            i0 = np.flatnonzero(fold == 0)
            ck = ckpt_dir(RUNS[k]) / "best.pt"
            rows.append(pd.DataFrame({"record_id": rid[ik],
                                      "oof": score_rows(ck, cf, cff, ik, dev,
                                                        args.batch_size)}))
            f0rows.append(pd.DataFrame({"record_id": rid[i0],
                                        f"Ffold_cv{k}": score_rows(ck, cf, cff, i0, dev,
                                                                   args.batch_size)}))
            print(f"dev: scored {RUNS[k]} on folds {{{k}, 0}}", flush=True)
        oof = pd.concat(rows, ignore_index=True)
        oof.to_parquet(oof_cache, index=False)
        g0 = f0rows[0]
        for r in f0rows[1:]:
            g0 = g0.merge(r, on="record_id", validate="1:1")
        g0.to_parquet(C.CACHE / f"fold0_perfold_F_v{CANON_VERSION}.parquet", index=False)
    dv = oof.merge(corp[["record_id", "edited"]], on="record_id", validate="1:1")
    iso = IsotonicRegression(out_of_bounds="clip").fit(dv.oof, dv.edited)
    panel["Ffold_cal"] = iso.predict(panel.Ffold.to_numpy())

    res = {"provenance": C.provenance([C.CORPUS, C.H2H], args.seed),
           "canon_version": CANON_VERSION,
           "disclosure": ("A further development use of the reserved panel; nothing here is "
                          "confirmatory."),
           "construction": {
               "members": list(RUNS.values()),
               "combiner": "plain mean of predicted efficiencies, matching the published member",
               "calibrator": ("isotonic, fitted on this model's own development out-of-fold "
                              f"predictions ({len(dv):,} rows over folds 1-5), frozen before "
                              "either surface was scored")},
           "panel": {}, "fold0": {}}

    # ------------------------------------------------- panel: the decision -----------
    models = ["op", "ours", "Ffold"] + [m for m in THREE_SEED if m in panel.columns] + memb
    cand = E.aggregate_candidates(panel, scores=tuple(models))
    nd = cand.groupby("edit_key", observed=True).design_key.transform("nunique")
    t = E.score_population(cand[nd >= 2], models, ks=(1, 3))
    res["panel"]["strata"] = []
    for name, sub in (("all eligible groups", t), ("informative groups", t[t.informative]),
                      ("depth 8+", t[t.depth >= 8]),
                      ("decision worth >= 0.05", t[(t.oracle - t.mean_y) >= 0.05])):
        if len(sub) < 100:
            continue
        s = {"stratum": name, "groups": int(len(sub)), "sites": int(sub.site.nunique()),
             "arms": {}, "paired": {}}
        for m in models + ["rand"]:
            s["arms"][m] = {"achieved_at_1": float(sub[f"{m}_at_1"].mean()),
                            "hit_at_1": float(sub[f"{m}_hit"].mean()),
                            "regret_at_1": float(sub[f"{m}_regret"].mean())}
        s["arms"]["oracle"] = {"achieved_at_1": float(sub.oracle.mean())}
        for a, b in (("Ffold", "ours"), ("Ffold", "op"), ("Ffold", "F3_ens"),
                     ("Ffold", "A_ens"), ("Ffold", "E_ens"), ("Ffold", "H_ens"),
                     ("ours", "op")):
            if f"{a}_at_1" not in sub or f"{b}_at_1" not in sub:
                continue
            s["paired"][f"{a}_minus_{b}"] = E.paired_cluster_bootstrap(
                (sub[f"{a}_at_1"] - sub[f"{b}_at_1"]).to_numpy(),
                sub.site.to_numpy(), args.seed)
        res["panel"]["strata"].append(s)

    # ------------------------------------------------- panel: the correlations -------
    y, cl = panel.edited_frac.to_numpy(), panel.protospacer.to_numpy()
    res["panel"]["correlations"] = {
        m: {"spearman": float(spearmanr(panel[m], y).statistic),
            "pearson": float(pearsonr(panel[m], y).statistic)}
        for m in ["op", "ours", "Ffold", "Ffold_cal"]
        + [x for x in THREE_SEED if x in panel.columns]}
    res["panel"]["margins_over_optiprime"] = {
        "spearman_Ffold": cluster_corr(y, panel.Ffold.to_numpy(), panel.op.to_numpy(), cl,
                                       args.seed, args.n_boot, "spearman"),
        "pearson_Ffold_calibrated": cluster_corr(y, panel.Ffold_cal.to_numpy(),
                                                 panel.op.to_numpy(), cl, args.seed,
                                                 args.n_boot, "pearson"),
        "spearman_Ffold_minus_published": cluster_corr(
            y, panel.Ffold.to_numpy(), panel.ours.to_numpy(), cl, args.seed, args.n_boot,
            "spearman")}

    # ------------------------------------------------------------------ fold 0 -------
    h = pd.read_parquet(C.H2H, columns=["record_id", "y", "op"])
    f0 = corp[corp.fold == 0].copy()
    f0["edited_frac"] = f0.edited
    # fold 0 was scored in the same pass as the out-of-fold predictions above
    g0 = pd.read_parquet(C.CACHE / f"fold0_perfold_F_v{CANON_VERSION}.parquet")
    f0 = f0.merge(g0, on="record_id", validate="1:1").merge(h, on="record_id",
                                                            validate="1:1")
    f0 = f0.merge(pd.read_parquet(C.CAL, columns=["record_id", "predicted_efficiency"]),
                  on="record_id", validate="1:1").rename(
                      columns={"predicted_efficiency": "ours"})
    f0["Ffold"] = f0[memb].mean(axis=1)
    f0["Ffold_cal"] = iso.predict(f0.Ffold.to_numpy())
    f0 = f0.merge(man, on="record_id", validate="1:1", suffixes=("", "_man"))

    y0, cl0 = f0.edited.to_numpy(), f0.spacer.to_numpy()
    res["fold0"] = {
        "rows": int(len(f0)),
        "correlations": {m: {"spearman": float(spearmanr(f0[m], y0).statistic),
                             "pearson": float(pearsonr(f0[m], y0).statistic)}
                         for m in ("op", "ours", "Ffold", "Ffold_cal")},
        "margins_over_optiprime": {
            "spearman_Ffold": cluster_corr(y0, f0.Ffold.to_numpy(), f0.op.to_numpy(), cl0,
                                           args.seed, args.n_boot, "spearman"),
            "pearson_Ffold_calibrated": cluster_corr(y0, f0.Ffold_cal.to_numpy(),
                                                     f0.op.to_numpy(), cl0, args.seed,
                                                     args.n_boot, "pearson")}}
    # One candidate is one DISTINCT DESIGN in one decision group, not one row: repeated
    # measurements of a design are averaged first. `design_key` and `decision_group` both
    # come from the manifest, and the cluster unit is the protospacer (`spacer`).
    m0 = ["op", "ours", "Ffold"]
    cd = (f0.groupby(["decision_group", "design_key"], observed=True, as_index=False)
            .agg(edited_frac=("edited", "mean"), protospacer=("spacer", "first"),
                 **{m: (m, "mean") for m in m0})
            .rename(columns={"decision_group": "edit_key"}))
    ndd = cd.groupby("edit_key", observed=True).design_key.transform("nunique")
    t0 = E.score_population(cd[ndd >= 2], m0, ks=(1,))
    i0 = t0[t0.informative]
    res["fold0"]["decision"] = {
        "informative_groups": int(len(i0)), "sites": int(i0.site.nunique()),
        "arms": {m: float(i0[f"{m}_at_1"].mean()) for m in m0 + ["rand"]},
        "oracle": float(i0.oracle.mean()),
        "paired": {f"{a}_minus_{b}": E.paired_cluster_bootstrap(
            (i0[f"{a}_at_1"] - i0[f"{b}_at_1"]).to_numpy(), i0.site.to_numpy(), args.seed)
            for a, b in (("Ffold", "ours"), ("Ffold", "op"), ("ours", "op"))}}

    C.write_outputs("e24_perfold_recipe", res, render(res))


def render(r: dict) -> str:
    lab = {"op": "OptiPrime", "ours": "PE-RankFormer, published member (5 per-fold)",
           "Ffold": "**arm F, 5 per-fold (this experiment)**",
           "F3_ens": "arm F, 3 same-data seeds (E22)",
           "A_ens": "arm A, 3 seeds (released recipe)", "rand": "random choice",
           "oracle": "perfect chooser", "Ffold_cal": "arm F, 5 per-fold, calibrated"}
    L = ["# E24 - the improved recipe, built the way the published model is built\n",
         "> **Disclosure.** " + r["disclosure"] + "\n",
         f"Five checkpoints, one per official fold, combined by {r['construction']['combiner']}. "
         f"Calibrator: {r['construction']['calibrator']}. Fold 0 is the official test fold and "
         "no member trains or early-stops on it.\n"]
    for s in r["panel"]["strata"]:
        L.append(f"## panel - {s['stratum']}: {s['groups']:,} groups, {s['sites']:,} sites\n")
        L.append("| model | achieved @1 | hit rate @1 | regret @1 |\n|---|---:|---:|---:|")
        for m in ("rand", "op", "ours", "A_ens", "F3_ens", "Ffold"):
            if m in s["arms"]:
                a = s["arms"][m]
                L.append(f"| {lab.get(m, m)} | {a['achieved_at_1']:.4f} | "
                         f"{a['hit_at_1']:.4f} | {a['regret_at_1']:.5f} |")
        L.append(f"| {lab['oracle']} | {s['arms']['oracle']['achieved_at_1']:.4f} | 1.0000 | "
                 "0.00000 |")
        L.append("\n| paired difference @1 | value | 95% CI | p |\n|---|---:|---|---:|")
        for k, v in s["paired"].items():
            a, b = k.split("_minus_")
            L.append(f"| {lab.get(a, a)} − {lab.get(b, b)} | {v['observed']:+.5f} | "
                     f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | {v['two_sided_p']:.3f} |")
        L.append("")
    for surf, key in (("reserved panel", "panel"), ("held-out fold 0", "fold0")):
        c = r[key]["correlations"]
        L.append(f"## correlations - {surf}\n")
        L.append("| model | Spearman | Pearson |\n|---|---:|---:|")
        for m in ("op", "ours", "Ffold", "Ffold_cal"):
            L.append(f"| {lab.get(m, m)} | {c[m]['spearman']:.4f} | {c[m]['pearson']:.4f} |")
        L.append("\n| margin | value | 95% CI | p |\n|---|---:|---|---:|")
        for k, v in r[key]["margins_over_optiprime"].items():
            L.append(f"| {k.replace('_', ' ')} | {v['observed']:+.4f} | "
                     f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['two_sided_p']:.4f} |")
        L.append("")
    d = r["fold0"]["decision"]
    L.append(f"## fold 0 - the decision: {d['informative_groups']:,} informative groups, "
             f"{d['sites']:,} clusters\n")
    L.append("| model | achieved @1 |\n|---|---:|")
    for m, v in d["arms"].items():
        L.append(f"| {lab.get(m, m)} | {v:.5f} |")
    L.append(f"| {lab['oracle']} | {d['oracle']:.5f} |")
    L.append("\n| paired difference @1 | value | 95% CI | p |\n|---|---:|---|---:|")
    for k, v in d["paired"].items():
        a, b = k.split("_minus_")
        L.append(f"| {lab.get(a, a)} − {lab.get(b, b)} | {v['observed']:+.5f} | "
                 f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | {v['two_sided_p']:.3f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
