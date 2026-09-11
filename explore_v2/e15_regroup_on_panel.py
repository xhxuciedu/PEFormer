"""E15 - does the regrouping repair close the gap on the panel where it was found?

E11 found that OptiPrime chooses better pegRNAs than PE-RankFormer on the reserved panel
(-0.00062 achieved efficiency, p = 0.0005, widening with candidate depth). E13 diagnosed a
mechanism in our own training objective -- the pairwise ranking loss groups rows by the raw
window pair and so sees 3.9% of the available within-allele comparisons -- and E14 showed
that repairing the grouping improves achieved efficiency on fold 0's fixed-allele decision
by +0.00258 (p = 0.01). This scores the repaired model on the panel itself.

**Disclosure: this is the panel's second use.** Its one pre-declared comparison was spent in
E11. What keeps this from being post-hoc fitting is the order of events, which the file
system records:

  13:22:59  E13's hypothesis, corpus, config and both training runs committed
  14:43:22  E11's panel head-to-head written -- the first time the deficit was known
  15:20:28  training's last checkpoint write (epoch selection on fold 1 only)

No configuration, hyperparameter or decision was altered after 13:22:59, and epoch selection
used the validation fold, never the panel. The regrouping hypothesis came from reading
`ranking_group_key` against E01's canonicalisation, not from the panel result. Even so, this
is reported as a **secondary, disclosed** comparison, not as a second confirmation, and the
next confirmatory claim needs a surface that is still sealed.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_v2/e15_regroup_on_panel.py
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
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, featurize  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

ARMS = {"e13_ctrl": "control (current ranking key)",
        "e13_canon": "canonical decision-group key"}


@torch.no_grad()
def score(ckpt: Path, corpus, device: str, bs: int) -> np.ndarray:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    ds = PEDataset(corpus)
    idx = np.arange(len(corpus.record_id))
    out = []
    for s in range(0, len(idx), bs):
        b = {k: v.to(device) for k, v in
             collate([ds[i] for i in idx[s:s + bs]]).items()}
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
    op = pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet")
    panel = panel.merge(op, on="record_id", validate="1:1")

    df = panel.copy()
    df["edited"] = df.edited_frac
    df["indel"] = 0.0
    df["fold"] = 0
    df["target_name"] = df.protospacer
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = featurize(df, vocab)

    cks = {}
    for stem in ARMS:
        found = sorted((ROOT / "checkpoints").glob(f"{stem}_*/best.pt"))
        if not found:
            raise SystemExit(f"missing checkpoint for {stem}")
        cks[stem] = found[-1]
        panel[stem] = score(found[-1], corpus, device, args.batch_size)
        print(f"{ARMS[stem]}: scored from {found[-1].parent.name}", flush=True)

    # the frozen published member, for continuity with E11
    cached = C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet"
    panel = panel.merge(pd.read_parquet(cached)[["record_id", "ours"]],
                        on="record_id", validate="1:1")

    models = ["ours", "op", "e13_ctrl", "e13_canon"]
    t = E11.build_table(panel, models)
    t.to_csv(C.OUT / "e15_panel_group_table.csv", index=False)

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "disclosure": ("Second use of the reserved panel. Training committed 13:22:59; "
                          "E11's panel head-to-head written 14:43:22; no configuration "
                          "changed after 13:22:59; epoch selection used fold 1 only."),
           "checkpoints": {k: str(v.relative_to(ROOT)) for k, v in cks.items()},
           "pooled_spearman": {m: C.spearman(panel[m].to_numpy(),
                                             panel.edited_frac.to_numpy())
                               for m in models},
           "strata": []}
    for name, sub in (("all eligible groups (depth >=2)", t),
                      ("depth 5-7", t[(t.depth >= 5) & (t.depth <= 7)]),
                      ("depth 8+", t[t.depth >= 8]),
                      ("decision worth >= 0.05", t[(t.oracle - t.mean_y) >= 0.05])):
        if len(sub) < 100:
            continue
        s = E11.summarise(sub, models, args.seed, name)
        # every pair that matters, against OptiPrime and against the control arm
        s["paired"] = {}
        for a, b in (("e13_canon", "op"), ("e13_ctrl", "op"), ("e13_canon", "e13_ctrl"),
                     ("ours", "op")):
            s["paired"][f"{a}_minus_{b}"] = {
                "achieved_efficiency_at_1": E11.site_bootstrap(
                    sub, f"{a}_best_of_1", f"{b}_best_of_1", args.seed),
                "regret_at_1": E11.site_bootstrap(
                    sub, f"{a}_regret_1", f"{b}_regret_1", args.seed)}
        res["strata"].append(s)
    C.write_outputs("e15_regroup_on_panel", res, render(res))


def render(r: dict) -> str:
    lab = {"ours": "PE-RankFormer, published member", "op": "OptiPrime",
           "e13_ctrl": "retrained, current ranking key",
           "e13_canon": "retrained, canonical decision-group key"}
    L = ["# E15 - the regrouping repair, on the panel where the deficit was found\n",
         "> **Disclosure.** " + r["disclosure"] + " This is a secondary comparison, not a "
         "second confirmation; the next confirmatory claim needs a sealed surface.\n",
         "## Pooled Spearman on the panel\n", "| model | pooled rho |\n|---|---:|"]
    for m, v in r["pooled_spearman"].items():
        L.append(f"| {lab[m]} | {v:.4f} |")
    for st in r["strata"]:
        L.append(f"\n## {st['stratum']}\n")
        L.append(f"{st['groups']:,} groups over {st['sites']:,} target sites; oracle "
                 f"{st['mean_oracle']:.4f}.\n")
        L.append("| model | achieved @1 | @3 | best-design hit rate | regret @1 |\n|---|---:|---:|---:|---:|")
        for m in ("rand", "op", "ours", "e13_ctrl", "e13_canon"):
            if m not in st["arms"]:
                continue
            a = st["arms"][m]
            L.append(f"| {lab.get(m, 'random choice')} | "
                     f"{a['achieved_efficiency_at_1']:.4f} | "
                     f"{a.get('achieved_efficiency_at_3', float('nan')):.4f} | "
                     f"{a['best_design_hit_rate']:.4f} | {a['regret_at_1']:.5f} |")
        L.append("\n| paired difference, achieved efficiency @1 | value | 95% CI | p |\n|---|---:|---|---:|")
        for k, v in st["paired"].items():
            e = v["achieved_efficiency_at_1"]
            a, b = k.split("_minus_")
            L.append(f"| {lab[a]} − {lab[b]} | {e['observed']:+.5f} | "
                     f"[{e['ci95'][0]:+.5f}, {e['ci95'][1]:+.5f}] | {e['two_sided_p']:.3g} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
