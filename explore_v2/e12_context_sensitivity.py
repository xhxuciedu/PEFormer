"""E12 - does the panel result depend on the experimental context we assigned it?

The reserved panel's rows carry no context metadata of their own, so E10 assigned the
context verbatim from the corpus's `Kim_HEK293T_LibSmall_PE2max_*` rows: HEK293T, PE2max,
conventional pegRNA, SpNGG, GC_F+E scaffold. Both predictors receive that assignment
identically, so the *comparison* is fair whatever the truth is. What the comparison cannot
settle by itself is whether the assignment changes the **decision**: if a different
plausible context reordered candidates inside a decision group, the panel's endpoints would
be an artefact of a metadata guess.

This scores the panel under the assigned context and under three alternatives a reasonable
person might have chosen, and measures how much the within-group ranking moves. Only
PE-RankFormer is re-scored: it is the model whose context input we control, and it is the
one whose result would be at risk.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 .venv/bin/python explore_v2/e12_context_sensitivity.py
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
import e11_score_reserved_panel as E11  # noqa: E402
from canon import CANON_VERSION  # noqa: E402

VARIANTS = {
    "assigned (Kim HEK293T PE2max, conventional pegRNA)": {},
    "epegRNA motif instead of none": {"motif": "tevoPreQ1", "epegRNA": 1},
    "PE4max instead of PE2max": {"pe_type": "PE4", "MLH1dn": 1},
    "K562 instead of HEK293T": {"cell_type": "K562"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    scores = {}
    for name, override in VARIANTS.items():
        p = panel.copy()
        for k, v in override.items():
            p[k] = v
        cached = (C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet"
                  if not override else None)
        if cached is not None and cached.exists():
            scores[name] = pd.read_parquet(cached).set_index("record_id").ours
            print(f"{name}: reused cache", flush=True)
        else:
            r = E11.score_ours(p, device, args.batch_size)
            scores[name] = r.set_index("record_id").ours
            print(f"{name}: scored", flush=True)

    base_name = next(iter(VARIANTS))
    panel = panel.set_index("record_id")
    y = panel.edited_frac
    g = pd.factorize(panel.edit_key)[0]
    order = np.argsort(g, kind="stable")
    gs = g[order]
    ys = y.to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    cnt = np.diff(np.r_[starts, len(gs)])
    ymax = np.maximum.reduceat(ys, starts)
    keep = ymax > np.minimum.reduceat(ys, starts)

    def picks(s: np.ndarray) -> np.ndarray:
        o = np.lexsort((-s, gs))
        return o[np.flatnonzero(np.r_[True, gs[o][1:] != gs[o][:-1]])]

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION, "variants": []}
    base_s = scores[base_name].reindex(panel.index).to_numpy()[order]
    base_pick = picks(base_s)
    for name in VARIANTS:
        s = scores[name].reindex(panel.index).to_numpy()[order]
        pk = picks(s)
        row = {"context": name,
               "pooled_spearman_vs_observed": C.spearman(s, ys),
               "rank_correlation_with_assigned_context": C.spearman(s, base_s),
               "same_top_pick_as_assigned": float((pk == base_pick).mean()),
               "achieved_efficiency_at_1": float(ys[pk][keep].mean()),
               "best_design_hit_rate": float((ys[pk][keep] == ymax[keep]).mean()),
               "regret_at_1": float((ymax[keep] - ys[pk][keep]).mean())}
        res["variants"].append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
    res["n_groups_scored"] = int(keep.sum())
    C.write_outputs("e12_context_sensitivity", res, render(res))


def render(r: dict) -> str:
    L = ["# E12 - sensitivity of the panel result to the assigned context\n",
         f"{r['n_groups_scored']:,} informative decision groups. PE-RankFormer re-scored "
         "under the assigned context and three alternatives.\n",
         "| context supplied to the model | pooled rho vs observed | rank agreement with "
         "the assigned context | same top pick | achieved @1 | hit rate | regret @1 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for v in r["variants"]:
        L.append(f"| {v['context']} | {v['pooled_spearman_vs_observed']:.4f} | "
                 f"{v['rank_correlation_with_assigned_context']:.4f} | "
                 f"{v['same_top_pick_as_assigned']:.4f} | "
                 f"{v['achieved_efficiency_at_1']:.4f} | "
                 f"{v['best_design_hit_rate']:.4f} | {v['regret_at_1']:.5f} |")
    L.append("\nIf the decision endpoints barely move across these alternatives, the panel "
             "result does not rest on the metadata assignment. If they move a lot, the "
             "assignment is load-bearing and has to be stated as a limitation of the "
             "absolute numbers -- though not of the head-to-head comparison, since both "
             "predictors receive the identical context.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
