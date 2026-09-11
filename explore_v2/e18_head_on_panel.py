"""E18 - is the ordinal head the right head for the *decision*?

The manuscript's titular claim is that a metric-matched **ordinal** outcome head is one of
the two components responsible for its advantage (+0.0092 pooled Spearman). That was
established on the pooled metric at the development evaluation's stated resolution of
approximately 0.005. The decision differences that separate methods on the reserved panel
are an order of magnitude smaller than that -- ours minus OptiPrime is 0.00064 with a
confidence half-width of 0.00026 -- so the pooled apparatus cannot resolve them, and neither
can fold 0: there the simplex-headed member beats the ordinal-headed one by +0.00075 with a
confidence interval of [-0.00156, +0.00284] and p = 0.50 over 546 informative groups.

The panel has 30,475 groups over 22,931 sites, roughly 56 times fold 0's decision population.
This scores the ordinal-S4D and simplex-S4D members there, out of fold by construction
(neither trained on this library), so the head comparison is made at a resolution that can
actually answer it.

**Disclosure:** a further development use of the reserved panel, whose one pre-declared
confirmatory comparison was spent in E11. Component decisions taken here need a sealed
surface before they become confirmatory claims.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_v2/e18_head_on_panel.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import endpoints as E  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, featurize  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

# The paper's members, each five checkpoints (one per official fold), averaged in rank space
MEMBERS = {"ordSSM": "r4p2_ordSSM", "ssm": "r4p2_ssm", "ordC": "r4p2_ordC"}
FEATURE_BRANCH = {"ordC"}   # the feature-branch member needs its features at eval time


@torch.no_grad()
def score_member(prefix: str, corpus, device: str, bs: int) -> np.ndarray:
    """Rank-average the member's five per-fold checkpoints, as the paper's rule specifies."""
    from scipy.stats import rankdata
    cks = sorted((ROOT / "checkpoints").glob(f"{prefix}_cv*/best.pt"))
    assert len(cks) == 5, f"{prefix}: expected 5 checkpoints, found {len(cks)}"
    ds = PEDataset(corpus)
    idx = np.arange(len(corpus.record_id))
    ranks = np.zeros(len(idx))
    for ck_path in cks:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        m = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device).eval()
        m.load_state_dict(ck["model_state_dict"])
        preds = []
        for s in range(0, len(idx), bs):
            b = {k: v.to(device) for k, v in collate([ds[i] for i in idx[s:s + bs]]).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o = m(b)
            preds.append(m.efficiency_from_output(o).float().cpu().numpy())
        ranks += rankdata(np.concatenate(preds)) / len(idx)
        del m
        torch.cuda.empty_cache()
        print(f"  {ck_path.parent.name}", flush=True)
    return ranks / len(cks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--members", nargs="+", default=["ordSSM", "ssm"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    panel = panel.merge(pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet"),
                        on="record_id", validate="1:1")
    cache = C.CACHE / f"panel_members_v{CANON_VERSION}.parquet"
    if cache.exists():
        got = pd.read_parquet(cache)
    else:
        df = panel.copy()
        df["edited"] = df.edited_frac
        df["indel"] = 0.0
        df["fold"] = 0
        df["target_name"] = df.protospacer
        vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
        corpus = featurize(df, vocab)
        cols = {"record_id": panel.record_id.to_numpy()}
        for name in args.members:
            if name in FEATURE_BRANCH:
                print(f"skipping {name}: needs the feature pipeline at eval time", flush=True)
                continue
            print(f"scoring member {name}", flush=True)
            cols[name] = score_member(MEMBERS[name], corpus, device, args.batch_size)
        got = pd.DataFrame(cols)
        got.to_parquet(cache, index=False)
    panel = panel.merge(got, on="record_id", validate="1:1")
    models = [m for m in args.members if m in panel.columns] + ["op"]

    cand = E.aggregate_candidates(panel, scores=tuple(models))
    nd = cand.groupby("edit_key", observed=True).design_key.transform("nunique")
    cand = cand[nd >= 2]
    t = E.score_population(cand, models, ks=(1, 3))

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "disclosure": ("A further development use of the reserved panel. Component "
                          "decisions taken here need a sealed surface to become "
                          "confirmatory."),
           "resolution_note": {
               "fold0_informative_groups": 546,
               "fold0_ssm_minus_ordSSM": 0.00075,
               "fold0_ci": [-0.00156, 0.00284], "fold0_p": 0.504,
               "paper_development_resolution_pooled_rho": 0.005,
               "panel_groups": int(len(t))},
           "strata": []}
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
        pairs = [("ssm", "ordSSM"), ("ordSSM", "op"), ("ssm", "op")]
        for a, b in pairs:
            if a in models and b in models:
                s["paired"][f"{a}_minus_{b}"] = E.paired_cluster_bootstrap(
                    (sub[f"{a}_at_1"] - sub[f"{b}_at_1"]).to_numpy(),
                    sub.site.to_numpy(), args.seed)
        s["pooled_spearman"] = {m: C.spearman(panel[m].to_numpy(),
                                              panel.edited_frac.to_numpy())
                                for m in models}
        res["strata"].append(s)
    C.write_outputs("e18_head_on_panel", res, render(res))


def render(r: dict) -> str:
    rn = r["resolution_note"]
    L = ["# E18 - the ordinal head versus the simplex head, on the decision\n",
         "> **Disclosure.** " + r["disclosure"] + "\n",
         "## Why fold 0 cannot answer this\n",
         f"The manuscript attributes +0.0092 pooled Spearman to the ordinal head, established "
         f"at a development resolution of about {rn['paper_development_resolution_pooled_rho']}. "
         "The decision differences that separate methods are an order of magnitude smaller. "
         f"On fold 0's {rn['fold0_informative_groups']} informative decision groups the "
         f"simplex-headed member beats the ordinal-headed one by "
         f"{rn['fold0_ssm_minus_ordSSM']:+.5f}, CI [{rn['fold0_ci'][0]:+.5f}, "
         f"{rn['fold0_ci'][1]:+.5f}], p = {rn['fold0_p']:.3g} — no power at all. The panel "
         f"offers {rn['panel_groups']:,} groups.\n",
         "## On the panel\n"]
    for st in r["strata"]:
        L.append(f"### {st['stratum']} — {st['groups']:,} groups, {st['sites']:,} sites\n")
        L.append("| member | achieved @1 | hit rate @1 | regret @1 |\n|---|---:|---:|---:|")
        for m, a in st["arms"].items():
            L.append(f"| {m} | {a['achieved_at_1']:.4f} | {a['hit_at_1']:.4f} | "
                     f"{a['regret_at_1']:.5f} |")
        L.append("\n| paired difference @1 | value | 95% CI | p |\n|---|---:|---|---:|")
        for k, v in st["paired"].items():
            L.append(f"| {k.replace('_minus_', ' − ')} | {v['observed']:+.5f} | "
                     f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] | {v['two_sided_p']:.3g} |")
        L.append("")
    ps = r["strata"][0]["pooled_spearman"]
    L.append("Pooled Spearman on the panel, for contrast with the decision above: "
             + ", ".join(f"{k} {v:.4f}" for k, v in ps.items()) + ".\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
