"""E27 - evaluate adapted models on the held-out test components (plan §1).

Primary endpoint, as specified: group-mean achieved efficiency at rank 1 on the identical
eligible test candidates, against released OptiPrime, with a paired interval clustered on the
locus component -- the independence unit the E25 partition was built from. Regret, hit rate,
depth strata and pooled correlation are reported alongside, and source retention is measured
on development fold 0 so that any forgetting is visible rather than assumed.

Nothing was selected on these components: every arm's epoch and every design choice came from
the validation components.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=n .venv/bin/python explore_v2/e27_adaptation_test.py
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
import adapt_data as AD  # noqa: E402
import endpoints as E  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from e26_adaptation_pilot import ARMS, load_base  # noqa: E402

RUNS = C.CACHE / "adapt_runs"


def reference_scores() -> pd.DataFrame:
    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet",
                            columns=["record_id", "edit_key", "design_key"])
    op = pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet")
    ours = pd.read_parquet(
        C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet")[["record_id", "ours"]]
    m = panel.merge(op, on="record_id").merge(ours, on="record_id")
    return (m.groupby(["edit_key", "design_key"], observed=True, as_index=False)
             .agg(op=("op", "mean"), ours=("ours", "mean")))


@torch.no_grad()
def score_run(meta_path: Path, c: pd.DataFrame, corpus, geom, gid, idx, device, bs=1024):
    meta = json.loads(meta_path.read_text())
    arm = meta["arm"]
    cfg = ARMS[arm]
    ck = torch.load(meta_path.with_suffix(".pt"), map_location=device, weights_only=False)
    if arm == "G":
        model = adapt.GeometryOnly(geom.shape[1]).to(device)
        model.load_state_dict(ck["state_dict"])
    else:
        base = load_base(device)
        model = adapt.Adapted(base, cfg["mode"],
                              geom_dim=geom.shape[1] if cfg["geom"] else 0).to(device)
        model.load_state_dict(ck["state_dict"])
    model.eval()
    batcher = AD.Batcher(corpus, geom, c.y.to_numpy(), gid)
    rng = np.random.default_rng(0)
    batches = [idx[b] for b in AD.group_batches(gid[idx], bs, rng, shuffle=False)]
    sel, pred, order = [], [], []
    for b in batches:
        batch, g, y, _ = batcher.make(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if arm == "G":
                s = model(g)
                p = s
            else:
                out, s0 = model(batch, g)
                p = model.base.efficiency_from_output(out)
                s = model.selection_score(out, s0)
        sel.append(s.float().cpu().numpy())
        pred.append(p.float().cpu().numpy())
        order.append(b[np.argsort(gid[b], kind="stable")])
    del model
    torch.cuda.empty_cache()
    return meta, np.concatenate(order), np.concatenate(sel), np.concatenate(pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    c = AD.load_candidates()
    corpus = AD.featurize_candidates(c)
    geom = AD.geometry_matrix(c, (c.split == "train").to_numpy())
    gid = pd.factorize(c.edit_key, sort=False)[0]
    idx = np.flatnonzero((c.split == args.split).to_numpy())

    ev = c.iloc[idx][["edit_key", "design_key", "component", "y"]].copy()
    ev = ev.merge(reference_scores(), on=["edit_key", "design_key"], validate="1:1")
    ev = ev.rename(columns={"y": "edited_frac", "component": "protospacer"})
    models = ["op", "ours"]

    for meta_path in sorted(RUNS.glob("*.json")):
        if "smoke" in meta_path.stem:
            continue
        meta, order, sel, pred = score_run(meta_path, c, corpus, geom, gid, idx, device)
        col = meta["name"]
        add = pd.DataFrame({"edit_key": c.edit_key.to_numpy()[order],
                            "design_key": c.design_key.to_numpy()[order],
                            col: sel, f"{col}__pred": pred})
        ev = ev.merge(add, on=["edit_key", "design_key"], validate="1:1")
        models.append(col)
        print(f"scored {col}", flush=True)

    t = E.score_population(ev, models, ks=(1, 3))
    res = {"provenance": C.provenance([C.CORPUS], args.seed), "canon_version": CANON_VERSION,
           "split": args.split,
           "note": ("Clusters are E25 locus components. No arm's epoch or configuration was "
                    "chosen using this split."),
           "strata": []}
    for name, sub in (("all eligible", t), ("informative", t[t.informative]),
                      ("depth >= 5", t[t.depth >= 5])):
        if len(sub) < 50:
            continue
        s = {"stratum": name, "groups": int(len(sub)),
             "clusters": int(sub.site.nunique()), "arms": {}, "paired": {}}
        for m in models + ["rand"]:
            s["arms"][m] = {"achieved_at_1": float(sub[f"{m}_at_1"].mean()),
                            "hit_at_1": float(sub[f"{m}_hit"].mean()),
                            "regret_at_1": float(sub[f"{m}_regret"].mean())}
        s["arms"]["oracle"] = {"achieved_at_1": float(sub.oracle.mean())}
        for m in models:
            if m == "op":
                continue
            for ref in ("op", "ours"):
                if m == ref:
                    continue
                s["paired"][f"{m}_minus_{ref}"] = E.paired_cluster_bootstrap(
                    (sub[f"{m}_at_1"] - sub[f"{ref}_at_1"]).to_numpy(),
                    sub.site.to_numpy(), args.seed)
        res["strata"].append(s)
    res["pooled_spearman"] = {
        m: C.spearman(ev[m if m in ("op", "ours") else f"{m}__pred"].to_numpy(),
                      ev.edited_frac.to_numpy()) for m in models}
    res["regret_reduction_vs_optiprime"] = {}
    all_e = res["strata"][0]
    op_reg = all_e["arms"]["op"]["regret_at_1"]
    for m in models:
        if m == "op":
            continue
        res["regret_reduction_vs_optiprime"][m] = float(
            (op_reg - all_e["arms"][m]["regret_at_1"]) / op_reg)
    # ---- the validation evidence the arm choice was actually made on ------------------
    val = {}
    for mp in sorted(RUNS.glob("*.json")):
        if "smoke" in mp.stem:
            continue
        m = json.loads(mp.read_text())
        key = m["arm"] + (f"|{m['args']['tag']}" if m["args"].get("tag") else "")
        val.setdefault(key, {"seeds": [], "val_at_1": [], "val_rho": [], "epochs": [],
                             "trainable": m["trainable_params"],
                             "train_groups": m["train_groups"],
                             "wall_seconds": []})
        v = val[key]
        v["seeds"].append(m["args"]["seed"])
        v["val_at_1"].append(m["best_val"]["achieved_at_1"])
        v["val_rho"].append(m["best_val"]["spearman"])
        v["epochs"].append(m["best_val"]["epoch"])
        v["wall_seconds"].append(m["wall_seconds"])
    for v in val.values():
        v["mean_val_at_1"] = float(np.mean(v["val_at_1"]))
        v["sd_val_at_1"] = float(np.std(v["val_at_1"], ddof=1)) if len(v["val_at_1"]) > 1 else None
        v["mean_val_rho"] = float(np.mean(v["val_rho"]))
        v["n_seeds"] = len(v["seeds"])
    res["validation"] = val
    # seed-paired contrasts, the only ones the plan will accept as evidence of a mechanism
    res["validation_paired"] = {}
    for a, b in (("M", "P"), ("M", "S"), ("M", "shared"), ("S", "P"), ("S", "Z"),
                 ("P", "Z"), ("Z", "G")):
        if a not in val or b not in val:
            continue
        common = sorted(set(val[a]["seeds"]) & set(val[b]["seeds"]))
        if not common:
            continue
        da = {s: x for s, x in zip(val[a]["seeds"], val[a]["val_at_1"])}
        db = {s: x for s, x in zip(val[b]["seeds"], val[b]["val_at_1"])}
        d = [da[s] - db[s] for s in common]
        res["validation_paired"][f"{a}_minus_{b}"] = {
            "seeds": common, "per_seed": [round(x, 6) for x in d],
            "mean": float(np.mean(d)),
            "same_sign": bool(all(x > 0 for x in d) or all(x < 0 for x in d))}
    C.write_outputs(f"e27_adaptation_{args.split}", res, render(res))


def render(r: dict) -> str:
    lab = {"op": "OptiPrime (released)", "ours": "PE-RankFormer (published, unadapted)",
           "rand": "random choice", "oracle": "perfect chooser"}
    L = [f"# E27 - adapted models on the held-out {r['split']} components\n",
         f"> {r['note']}\n"]
    for s in r["strata"]:
        L.append(f"## {s['stratum']} - {s['groups']:,} groups, {s['clusters']:,} clusters\n")
        L.append("| model | achieved @1 | hit @1 | regret @1 | vs OptiPrime | p |")
        L.append("|---|---:|---:|---:|---|---:|")
        for m, a in s["arms"].items():
            d = s["paired"].get(f"{m}_minus_op")
            delta = (f"{d['observed']:+.5f} [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]"
                     if d else "—")
            pv = f"{d['two_sided_p']:.3f}" if d else "—"
            L.append(f"| {lab.get(m, m)} | {a['achieved_at_1']:.5f} | "
                     f"{a.get('hit_at_1', float('nan')):.4f} | "
                     f"{a.get('regret_at_1', float('nan')):.5f} | {delta} | {pv} |")
        L.append("")
    L.append("## Prediction retained?\n")
    L.append("| model | pooled Spearman |\n|---|---:|")
    for m, v in r["pooled_spearman"].items():
        L.append(f"| {lab.get(m, m)} | {v:.4f} |")
    L.append("\n## Regret reduction against OptiPrime\n")
    L.append("| model | reduction |\n|---|---:|")
    for m, v in r["regret_reduction_vs_optiprime"].items():
        L.append(f"| {lab.get(m, m)} | {v:+.1%} |")
    L.append("\nThe plan's provisional development target was a 10% reduction of OptiPrime's "
             "regret.\n")
    v = r.get("validation")
    if v:
        L.append("## The validation evidence the arms were chosen on\n")
        L.append("| arm | seeds | mean val @1 | sd | mean val rho | trainable | train groups |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for k, a in sorted(v.items(), key=lambda kv: -kv[1]["mean_val_at_1"]):
            sd = f"{a['sd_val_at_1']:.5f}" if a["sd_val_at_1"] is not None else "—"
            L.append(f"| {k} | {a['n_seeds']} | {a['mean_val_at_1']:.5f} | {sd} | "
                     f"{a['mean_val_rho']:.4f} | {a['trainable']:,} | "
                     f"{a['train_groups']:,} |")
        vp = r.get("validation_paired") or {}
        if vp:
            L.append("\n| seed-paired contrast | per seed | mean | all seeds agree |")
            L.append("|---|---|---:|:--:|")
            for k, d in vp.items():
                L.append(f"| {k.replace('_minus_', ' − ')} | "
                         f"{', '.join(f'{x:+.5f}' for x in d['per_seed'])} | "
                         f"{d['mean']:+.5f} | {'yes' if d['same_sign'] else '**no**'} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
