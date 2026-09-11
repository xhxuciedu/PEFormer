"""E16 - recompute the panel endpoints with the corrected accounting, and disclose the changes.

`NEXT_STEPS_AFTER_E15.md` section 2 identified four defects in E11/E15's inline endpoint
code. All four were reproduced exactly (13 groups with more rows than designs; 2 depth-2
groups given a top-three result; a random hit rate of 0.429 on all-zero groups where 1.0 is
correct; p = 0.0005 for two identical arms). `endpoints.py` fixes them and
`test_endpoints.py` pins the invariants with 13 tests.

This script re-derives every panel endpoint through that module for all four predictors and
prints the corrected value beside the published one, so the effect of the correction is
visible rather than quietly absorbed.

Budget curves are reported on one **fixed** eligible population per k -- groups with at
least k distinct candidates -- each with its own oracle and random baseline, rather than
quoting an all-group oracle beside a @3 computed on a deeper subset.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_v2/e16_corrected_endpoints.py
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

MODELS = ["ours", "op", "e13_ctrl", "e13_canon"]
LAB = {"ours": "PE-RankFormer, published member", "op": "OptiPrime",
       "e13_ctrl": "retrained, current ranking key",
       "e13_canon": "retrained, canonical decision-group key",
       "rand": "random choice"}
KS = (1, 3, 5)

# what E11/E15 reported, for a side-by-side
PUBLISHED = {"groups": 30475, "oracle": 0.0461,
             "ours_at_1": 0.0363, "op_at_1": 0.0369, "rand_at_1": 0.0217,
             "ours_hit": 0.6599, "op_hit": 0.6647, "rand_hit": 0.3354,
             "ours_minus_op_at_1": -0.00062, "canon_minus_ctrl_at_1": 0.00178,
             "canon_minus_op_at_1": -0.00117}


@torch.no_grad()
def score_ckpt(stem: str, corpus, device: str, bs: int) -> np.ndarray:
    ck_dir = sorted((ROOT / "checkpoints").glob(f"{stem}_*/best.pt"))[-1]
    ck = torch.load(ck_dir, map_location=device, weights_only=False)
    m = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    ds = PEDataset(corpus)
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


def load_panel(device: str, bs: int) -> pd.DataFrame:
    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    panel = panel.merge(pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet"),
                        on="record_id", validate="1:1")
    panel = panel.merge(pd.read_parquet(
        C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet")[["record_id", "ours"]],
        on="record_id", validate="1:1")
    cache = C.CACHE / f"panel_predictions_e13_v{CANON_VERSION}.parquet"
    if cache.exists():
        panel = panel.merge(pd.read_parquet(cache), on="record_id", validate="1:1")
    else:
        df = panel.copy()
        df["edited"] = df.edited_frac
        df["indel"] = 0.0
        df["fold"] = 0
        df["target_name"] = df.protospacer
        vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
        corpus = featurize(df, vocab)
        cols = {"record_id": panel.record_id.to_numpy()}
        for stem in ("e13_ctrl", "e13_canon"):
            cols[stem] = score_ckpt(stem, corpus, device, bs)
            print(f"scored {stem}", flush=True)
        got = pd.DataFrame(cols)
        got.to_parquet(cache, index=False)
        panel = panel.merge(got, on="record_id", validate="1:1")
    return panel


def summarise(t: pd.DataFrame, label: str, seed: int) -> dict:
    """Endpoints on a fixed population per k, plus every paired difference that matters."""
    out = {"stratum": label, "groups": int(len(t)), "sites": int(t.site.nunique()),
           "all_zero_groups": int(t.all_zero.sum()),
           "groups_with_tied_maximum": int((t.n_maximisers > 1).sum()),
           "oracle_all_groups": float(t.oracle.mean()),
           "populations": {}, "arms": {}, "paired": {}}
    for k in KS:
        sub = t[t.depth >= k]
        if len(sub) == 0:
            continue
        out["populations"][str(k)] = {"groups": int(len(sub)),
                                      "oracle": float(sub.oracle.mean()),
                                      "random": float(sub[f"rand_at_{k}"].mean())}
    for m in MODELS + ["rand"]:
        a = {}
        for k in KS:
            sub = t[t.depth >= k]
            if len(sub):
                a[f"achieved_at_{k}"] = float(sub[f"{m}_at_{k}"].mean())
        a["hit_at_1"] = float(t[f"{m}_hit"].mean())
        a["regret_at_1"] = float(t[f"{m}_regret"].mean())
        out["arms"][m] = a
    for a, b in (("ours", "op"), ("e13_canon", "op"), ("e13_ctrl", "op"),
                 ("e13_canon", "e13_ctrl"), ("ours", "ours")):
        d = E.paired_cluster_bootstrap((t[f"{a}_at_1"] - t[f"{b}_at_1"]).to_numpy(),
                                       t.site.to_numpy(), seed)
        out["paired"][f"{a}_minus_{b}_at_1"] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = load_panel(device, args.batch_size)
    rows_before = len(panel)
    cand = E.aggregate_candidates(panel, scores=tuple(MODELS))
    # only groups that still pose a choice after duplicate designs are collapsed
    depth = cand.groupby("edit_key", observed=True).design_key.transform("nunique")
    cand = cand[depth >= 2]
    t = E.score_population(cand, MODELS, ks=KS)
    t.to_csv(C.OUT / "e16_group_table.csv", index=False)

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "accounting": {
               "panel_rows": int(rows_before),
               "candidates_after_aggregating_duplicate_designs": int(len(cand)),
               "rows_collapsed": int(rows_before - len(cand)),
               "groups_retained_with_2plus_candidates": int(len(t)),
               "groups_dropped_by_recount": int(PUBLISHED["groups"] - len(t)),
           },
           "published_for_comparison": PUBLISHED,
           "strata": [summarise(t, "all eligible groups (depth >=2)", args.seed),
                      summarise(t[t.informative], "informative groups", args.seed),
                      summarise(t[t.depth >= 8], "depth 8+", args.seed),
                      summarise(t[(t.oracle - t.mean_y) >= 0.05],
                                "decision worth >= 0.05", args.seed)]}
    C.write_outputs("e16_corrected_endpoints", res, render(res))


def render(r: dict) -> str:
    a, pub = r["accounting"], r["published_for_comparison"]
    prim = r["strata"][0]
    L = ["# E16 - the panel endpoints, recomputed with corrected accounting\n",
         "Four defects in E11/E15's inline endpoint code were reproduced exactly and fixed in "
         "`endpoints.py`, whose invariants are pinned by 13 tests in `test_endpoints.py`. "
         "This is the corrected recomputation, printed beside the published values.\n",
         "## What the recount changed\n",
         f"- panel rows: {a['panel_rows']:,}\n"
         f"- distinct (group, design) candidates after collapsing duplicate designs: "
         f"{a['candidates_after_aggregating_duplicate_designs']:,} "
         f"({a['rows_collapsed']:,} rows collapsed)\n"
         f"- decision groups retained at depth >=2: {a['groups_retained_with_2plus_candidates']:,} "
         f"(published {pub['groups']:,}; {a['groups_dropped_by_recount']:+,})\n"
         f"- groups with a tied measured maximum: {prim['groups_with_tied_maximum']:,}, of which "
         f"{prim['all_zero_groups']:,} are all-zero\n",
         "## Primary stratum, corrected\n",
         "Each budget is scored on the groups that actually have that many distinct "
         "candidates, with that population's own oracle and random baseline:\n",
         "| budget k | groups | oracle | random |\n|---|---:|---:|---:|"]
    for k, v in prim["populations"].items():
        L.append(f"| {k} | {v['groups']:,} | {v['oracle']:.4f} | {v['random']:.4f} |")
    L.append("\n| predictor | achieved @1 | @3 | @5 | hit rate @1 | regret @1 |\n|---|---:|---:|---:|---:|---:|")
    for m in ("rand", "op", "ours", "e13_ctrl", "e13_canon"):
        v = prim["arms"][m]
        L.append(f"| {LAB[m]} | {v.get('achieved_at_1', float('nan')):.4f} | "
                 f"{v.get('achieved_at_3', float('nan')):.4f} | "
                 f"{v.get('achieved_at_5', float('nan')):.4f} | "
                 f"{v['hit_at_1']:.4f} | {v['regret_at_1']:.5f} |")
    L.append("\n### Published versus corrected, the numbers that carry claims\n")
    L.append("| quantity | published | corrected |\n|---|---:|---:|")
    pa = prim["arms"]
    for key, lab, val in (("ours_at_1", "PE-RankFormer achieved @1", pa["ours"]["achieved_at_1"]),
                          ("op_at_1", "OptiPrime achieved @1", pa["op"]["achieved_at_1"]),
                          ("rand_at_1", "random achieved @1", pa["rand"]["achieved_at_1"]),
                          ("ours_hit", "PE-RankFormer hit rate", pa["ours"]["hit_at_1"]),
                          ("op_hit", "OptiPrime hit rate", pa["op"]["hit_at_1"]),
                          ("rand_hit", "random hit rate", pa["rand"]["hit_at_1"])):
        L.append(f"| {lab} | {pub[key]:.4f} | {val:.4f} |")
    for key, lab in (("ours_minus_op_at_1", "ours − OptiPrime @1"),
                     ("canon_minus_ctrl_at_1", "canonical − control @1"),
                     ("canon_minus_op_at_1", "canonical − OptiPrime @1")):
        pk = {"ours_minus_op_at_1": "ours_minus_op_at_1",
              "canon_minus_ctrl_at_1": "e13_canon_minus_e13_ctrl_at_1",
              "canon_minus_op_at_1": "e13_canon_minus_op_at_1"}[key]
        d = prim["paired"][pk]
        L.append(f"| {lab} | {pub[key]:+.5f} | {d['observed']:+.5f} "
                 f"(CI [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}], p {d['two_sided_p']:.3g}) |")
    same = prim["paired"]["ours_minus_ours_at_1"]
    L.append(f"\nSelf-comparison control: a predictor against itself now returns "
             f"observed {same['observed']:.1f}, p = {same['two_sided_p']:.3g}, "
             f"degenerate flag {same['degenerate_zero_difference']} — where the published "
             "code returned p = 0.0005.\n")
    for st in r["strata"][1:]:
        L.append(f"\n## {st['stratum']}\n")
        L.append(f"{st['groups']:,} groups, {st['sites']:,} sites.\n")
        L.append("| predictor | achieved @1 | hit rate @1 | regret @1 |\n|---|---:|---:|---:|")
        for m in ("rand", "op", "ours", "e13_ctrl", "e13_canon"):
            v = st["arms"][m]
            L.append(f"| {LAB[m]} | {v.get('achieved_at_1', float('nan')):.4f} | "
                     f"{v['hit_at_1']:.4f} | {v['regret_at_1']:.5f} |")
        L.append("\n| paired difference @1 | value | 95% CI | p |\n|---|---:|---|---:|")
        for k in ("ours_minus_op_at_1", "e13_canon_minus_op_at_1",
                  "e13_canon_minus_e13_ctrl_at_1"):
            d = st["paired"][k]
            L.append(f"| {k.replace('_at_1', '').replace('_minus_', ' − ')} | "
                     f"{d['observed']:+.5f} | [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}] | "
                     f"{d['two_sided_p']:.3g} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
