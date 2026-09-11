"""E28 - what target adaptation costs on the source corpus (plan §4, §8).

Adapting a model to one library can quietly destroy it on everything else, and the plan is
explicit that preservation must be measured rather than assumed. Each adapted model is
scored on development fold 0 -- the official held-out test fold of the source corpus, which
no adaptation run touched -- and compared with the checkpoint it started from.

Only the pooled-only arms appear here. The geometry arms take an input the source corpus
would need recomputed through a different pipeline, and the geometry-only control has no
sequence model to retain anything.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=n .venv/bin/python explore_v2/e28_source_retention.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import adapt  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from e26_adaptation_pilot import ARMS, achieved_at_1, load_base  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402

RUNS = C.CACHE / "adapt_runs"


@torch.no_grad()
def score_fold0(model, corpus, rows, device, bs=1024, geom_dim=0):
    ds = PEDataset(corpus)
    out = []
    for s in range(0, len(rows), bs):
        idx = rows[s:s + bs]
        batch = {k: v.to(device) for k, v in collate([ds[int(i)] for i in idx]).items()}
        g = (torch.zeros(len(idx), geom_dim, device=device) if geom_dim else None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, _ = model(batch, g)
            p = model.base.efficiency_from_output(o)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = load_featurized(str(ROOT / "data/processed/featurized_official.npz"), vocab)
    fold = np.asarray(corpus.fold)
    rows = np.flatnonzero(fold == 0)
    rid = np.asarray(corpus.record_id)[rows]
    man = pd.read_parquet(C.require_manifest(),
                          columns=["record_id", "decision_group", "design_key", "spacer",
                                   "edited"])
    ref = pd.DataFrame({"record_id": rid}).merge(man, on="record_id", validate="1:1")
    y = ref.edited.to_numpy()

    def decision(score):
        d = ref.assign(s=score).groupby(["decision_group", "design_key"], observed=True,
                                        as_index=False).agg(y=("edited", "mean"),
                                                            s=("s", "mean"))
        nd = d.groupby("decision_group", observed=True).design_key.transform("nunique")
        d = d[nd >= 2]
        gid = pd.factorize(d.decision_group)[0]
        return achieved_at_1(d.s.to_numpy(), d.y.to_numpy(), gid, informative_only=True)

    res = {"provenance": C.provenance([C.CORPUS], args.seed), "canon_version": CANON_VERSION,
           "surface": {"fold": 0, "rows": int(len(rows)),
                       "note": ("the source corpus's official held-out fold; no adaptation "
                                "run trained or selected on it")},
           "models": {}}

    base0 = load_base(device)
    start = adapt.Adapted(base0, "P", geom_dim=0).to(device).eval()
    p0 = score_fold0(start, corpus, rows, device)
    res["models"]["start_checkpoint"] = {
        "spearman": C.spearman(p0, y), "decision_at_1": decision(p0)}
    del start, base0
    torch.cuda.empty_cache()

    for mp in sorted(RUNS.glob("*.json")):
        if "smoke" in mp.stem:
            continue
        meta = json.loads(mp.read_text())
        cfg = ARMS[meta["arm"]]
        if cfg["geom"] or cfg["mode"] == "G":
            continue
        base = load_base(device)
        m = adapt.Adapted(base, cfg["mode"], geom_dim=0).to(device)
        m.load_state_dict(torch.load(mp.with_suffix(".pt"), map_location=device,
                                     weights_only=False)["state_dict"])
        m.eval()
        p = score_fold0(m, corpus, rows, device)
        # the label budget variants are separate arms for aggregation: "ordinary
        # fine-tuning" means the full budget, not an average over learning-curve runs
        tag = meta["args"].get("tag") or ""
        res["models"][meta["name"]] = {
            "arm": meta["arm"] + (f"|{tag}" if tag else ""),
            "spearman": C.spearman(p, y), "decision_at_1": decision(p)}
        del m, base
        torch.cuda.empty_cache()
        print(f"scored {meta['name']}", flush=True)

    s0 = res["models"]["start_checkpoint"]
    for k, v in res["models"].items():
        if k == "start_checkpoint":
            continue
        v["spearman_change"] = v["spearman"] - s0["spearman"]
        v["decision_change"] = v["decision_at_1"] - s0["decision_at_1"]
    by_arm: dict[str, list] = {}
    for k, v in res["models"].items():
        if k != "start_checkpoint":
            by_arm.setdefault(v["arm"], []).append(v["spearman_change"])
    res["mean_spearman_change_by_arm"] = {a: float(np.mean(x)) for a, x in by_arm.items()}
    C.write_outputs("e28_source_retention", res, render(res))


def render(r: dict) -> str:
    s0 = r["models"]["start_checkpoint"]
    L = ["# E28 - what target adaptation costs on the source corpus\n",
         f"Development fold 0, {r['surface']['rows']:,} rows -- "
         f"{r['surface']['note']}.\n",
         f"The starting checkpoint scores **{s0['spearman']:.4f}** pooled Spearman and "
         f"**{s0['decision_at_1']:.5f}** achieved efficiency @1 there.\n",
         "| adapted model | arm | Spearman | change | decision @1 | change |",
         "|---|---|---:|---:|---:|---:|"]
    for k, v in r["models"].items():
        if k == "start_checkpoint":
            continue
        L.append(f"| {k} | {v['arm']} | {v['spearman']:.4f} | {v['spearman_change']:+.4f} | "
                 f"{v['decision_at_1']:.5f} | {v['decision_change']:+.5f} |")
    L.append("\n## Mean source Spearman change by arm\n")
    L.append("| arm | change |\n|---|---:|")
    for a, v in sorted(r["mean_spearman_change_by_arm"].items(), key=lambda kv: kv[1]):
        L.append(f"| {a} | {v:+.4f} |")
    L.append("\nA negative change is forgetting. The plan asks for this tradeoff to be "
             "visible rather than assumed away, and for replay to be tested against "
             "target-only adaptation only if the cost is consequential.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
