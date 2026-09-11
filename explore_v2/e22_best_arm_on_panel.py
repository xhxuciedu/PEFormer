"""E22 - does the best training recipe close the decision gap to OptiPrime?

E19 established, over three paired seeds on fold 0, that canonical batching helps
(E - A = +0.00099, same sign in all three), that the feature branch adds a further
consistent gain (F - E = +0.00064, 3/3), and that the best arm is F -- canonical batching,
canonical ranking key and the Family C feature branch -- at +0.00164 over the released
recipe, replicated in every seed.

E11/E16 established that the released model loses the fixed-allele decision to OptiPrime on
the reserved panel by 0.00064. The magnitudes are comparable, so the question is direct: on
the panel, does arm F reach parity?

Arms A and F are scored at all three seeds, individually and rank-averaged as a matched
three-model ensemble, so the comparison against OptiPrime is not confounded by ensemble size.

**Disclosure:** a further development use of the reserved panel, whose one pre-declared
confirmatory comparison was spent in E11. Nothing here is confirmatory; a sealed surface is
still required.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 .venv/bin/python explore_v2/e22_best_arm_on_panel.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import endpoints as E  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, featurize  # noqa: E402
from pe_rankformer.data.family_c_features import attach_family_c_features  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

ARMS = {"A": ["m_A_s1", "m_A_s2", "m_A_s3"],      # released recipe
        "D": ["m_D_s1", "m_D_s2", "m_D_s3"],      # canonical batching + ranking, no features
        "E": ["m_E_norank", "m_E_s2", "m_E_s3"],  # canonical batching only, no features
        "F": ["m_F_feat", "m_F_s2", "m_F_s3"],    # canonical both + feature branch
        "H": ["m_H_s1", "m_H_s2", "m_H_s3"]}      # canonical batching + features, no ranking
# Arms D and E carry no feature branch, so they are the control for the extrapolation
# question the frozen-standardisation fix raised: does the feature branch cost accuracy on a
# library whose geometry lies outside the range it was fitted on?


@torch.no_grad()
def score(ckpt: Path, corpus, corpus_feat, device: str, bs: int) -> np.ndarray:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = PERankFormerConfig(**ck["model_config"])
    m = PERankFormer(cfg).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    ds = PEDataset(corpus_feat if cfg.n_features else corpus)
    idx = np.arange(len(corpus.record_id))
    out = []
    for s in range(0, len(idx), bs):
        b = {k: v.to(device) for k, v in collate([ds[i] for i in idx[s:s + bs]]).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = m(b)
        out.append(m.efficiency_from_output(o).float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    panel = panel.merge(pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet"),
                        on="record_id", validate="1:1")
    panel = panel.merge(pd.read_parquet(
        C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet")[["record_id", "ours"]],
        on="record_id", validate="1:1")

    df = panel.copy()
    df["edited"] = df.edited_frac
    df["indel"] = 0.0
    df["fold"] = 0
    df["target_name"] = df.protospacer
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = featurize(df, vocab)
    # arm F trained with val_fold 1, so its training rows are folds 2-5
    import featstats
    corpus_feat = attach_family_c_features_for_panel(
        df, corpus, stats=featstats.train_stats(C.CORPUS, [2, 3, 4, 5]))

    cache = C.CACHE / f"panel_arms_v{CANON_VERSION}_fs.parquet"
    got = (pd.read_parquet(cache) if cache.exists()
           else pd.DataFrame({"record_id": panel.record_id.to_numpy()}))
    added = False
    for arm, runs in ARMS.items():
        for run in runs:
            if f"{arm}:{run}" in got.columns:
                continue
            dirs = sorted((ROOT / "checkpoints").glob(f"{run}_*"))
            if not dirs:
                print(f"SKIP {arm} {run}: no checkpoint", flush=True)
                continue
            got[f"{arm}:{run}"] = score(dirs[-1] / "best.pt", corpus, corpus_feat, device,
                                        args.batch_size)
            added = True
            print(f"scored {arm} {run}", flush=True)
    if added:
        got.to_parquet(cache, index=False)
    panel = panel.merge(got, on="record_id", validate="1:1")

    # matched three-seed rank-average per arm, so ensemble size cannot explain a difference
    models = ["op", "ours"]
    for arm, runs in ARMS.items():
        runs = [r for r in runs if f"{arm}:{r}" in panel.columns]
        if not runs:
            continue
        per = [rankdata(panel[f"{arm}:{r}"].to_numpy()) / len(panel) for r in runs]
        panel[f"{arm}_ens"] = np.mean(per, axis=0)
        models.append(f"{arm}_ens")
        for r in runs:
            models.append(f"{arm}:{r}")

    cand = E.aggregate_candidates(panel, scores=tuple(models))
    nd = cand.groupby("edit_key", observed=True).design_key.transform("nunique")
    cand = cand[nd >= 2]
    t = E.score_population(cand, models, ks=(1, 3))

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "disclosure": ("A further development use of the reserved panel; nothing here is "
                          "confirmatory."),
           "arms": {a: r for a, r in ARMS.items()}, "strata": []}
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
        pairs = [(f"{a}_ens", "op") for a in ARMS] + \
                [(f"{a}_ens", "ours") for a in ARMS] + \
                [("F_ens", "A_ens"), ("F_ens", "E_ens"), ("H_ens", "E_ens"),
                 ("E_ens", "A_ens"), ("ours", "op")]
        for a, b in pairs:
            if f"{a}_at_1" not in sub or f"{b}_at_1" not in sub:
                continue
            s["paired"][f"{a}_minus_{b}"] = E.paired_cluster_bootstrap(
                (sub[f"{a}_at_1"] - sub[f"{b}_at_1"]).to_numpy(),
                sub.site.to_numpy(), args.seed)
        s["pooled_spearman"] = {m: C.spearman(panel[m].to_numpy(),
                                              panel.edited_frac.to_numpy())
                                for m in ["op", "ours"] + [f"{a}_ens" for a in ARMS]
                                if m in panel.columns}
        res["strata"].append(s)
    C.write_outputs("e22_best_arm_on_panel", res, render(res))


def attach_family_c_features_for_panel(df: pd.DataFrame, corpus, stats=None):
    """Compute the 17 Family C features for the panel and attach them.

    The RuleSet3 column is the only one that needs an external model. Those scores already
    exist for all 30,300 panel protospacers: they were computed in the isolated rs3
    environment to let OptiPrime run, and are read back out of its own disk cache rather
    than recomputed, so the two predictors see the identical values.
    """
    import hashlib
    import pickle
    sys.path.insert(0, str(ROOT / "scripts/data"))
    import compute_family_c_features as cf

    d = df.copy()
    d["scaffold_name"] = d.scaffold_name.astype(str)
    parts = [d[["record_id"]].reset_index(drop=True), cf.compute_length_gc(d),
             cf.compute_edit_geometry(d), cf.compute_tm(d),
             cf.compute_mfe(d, scaffold_col="scaffold_name")]
    feat = pd.concat(parts, axis=1)

    cache_root = (ROOT / "data/interim/reserved_panel_kim_large/_disk_cache/RuleSet3Score")
    scores = {}
    for shard in cache_root.glob("*_DATA.pkl"):
        scores.update(pickle.load(shard.open("rb")))
    rna = d.spacer.str.replace("T", "U", regex=False).str.replace("^G", "g", regex=True)
    h = rna.apply(lambda x: hashlib.sha256(x.encode("ascii")).hexdigest()[:10])
    feat["ruleset3_score"] = h.map(scores).to_numpy()
    miss = float(feat.ruleset3_score.isna().mean())
    print(f"panel features: ruleset3 missing for {miss:.2%} of rows", flush=True)

    tmp = C.CACHE / "panel_family_c_features.parquet"
    feat.to_parquet(tmp, index=False)
    if stats is not None:
        # Standardise by the CHECKPOINT'S OWN training statistics. Refitting them on the
        # evaluation surface rescales the inputs away from the scale the model was trained
        # to read; see explore_v2/featstats.py.
        import featstats
        return featstats.attach_frozen(corpus, feat, *stats)
    return attach_family_c_features(corpus, str(tmp),
                                    np.arange(len(corpus.record_id)))


def render(r: dict) -> str:
    lab = {"op": "OptiPrime", "ours": "PE-RankFormer, published member",
           "A_ens": "arm A, 3-seed ensemble (released recipe)",
           "F_ens": "arm F, 3-seed ensemble (canonical batching + features)",
           "rand": "random choice"}
    L = ["# E22 - the best training recipe, on the panel\n",
         "> **Disclosure.** " + r["disclosure"] + "\n",
         "Arms A and F are each rank-averaged over their three seeds, so ensemble size is "
         "matched between them and cannot explain a difference.\n"]
    for st in r["strata"]:
        L.append(f"## {st['stratum']} — {st['groups']:,} groups, {st['sites']:,} sites\n")
        L.append("| model | achieved @1 | hit rate @1 | regret @1 |\n|---|---:|---:|---:|")
        for m in ("rand", "op", "ours", "A_ens", "F_ens"):
            a = st["arms"][m]
            L.append(f"| {lab[m]} | {a['achieved_at_1']:.4f} | {a['hit_at_1']:.4f} | "
                     f"{a['regret_at_1']:.5f} |")
        L.append("\n| paired difference @1 | value | 95% CI | p |\n|---|---:|---|---:|")
        for k, v in st["paired"].items():
            a, b = k.split("_minus_")
            L.append(f"| {lab.get(a, a)} − {lab.get(b, b)} | {v['observed']:+.5f} | "
                     f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | {v['two_sided_p']:.3g} |")
        L.append("")
    ps = r["strata"][0]["pooled_spearman"]
    L.append("Pooled Spearman on the panel: "
             + ", ".join(f"{lab.get(k,k)} {v:.4f}" for k, v in ps.items()) + ".\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
