"""Three-seed average OUTCOMES, not prediction ensembling; paired locus intervals."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from common import OUT, CACHE, provenance, write_json
from summarize import group_table, contrast, NAMES
from e27_adaptation_test import reference_scores

ARMS = ("A","D","E","F")
SEEDS = (20260910,20260911,20260912)
SURFACES = ("target_outer_val","source_val","source_audit")

def main():
    torch.set_num_threads(2)
    index = pd.read_parquet(CACHE/"index.parquet")
    data = torch.load(CACHE/"prepared.pt",map_location="cpu",weights_only=False)
    q0 = data["q0"].numpy()
    baseline = {}
    for surface in SURFACES:
        f = index.loc[index.surface==surface,["row","group_id","design_key","component","y"]].rename(columns={"group_id":"edit_key"}).copy()
        f["selection"] = q0[f.row.to_numpy()]
        baseline[surface] = group_table(f,source=surface.startswith("source"))
        if surface=="target_outer_val":
            op = f.drop(columns="selection").merge(reference_scores(),on=["edit_key","design_key"],validate="1:1")
            baseline["optiprime"] = group_table(op.assign(selection=op.op))
    records, tables, paths = [], {}, []
    for budget in (200,1000):
        for arm in ARMS:
            row = {"budget":budget,"arm":arm,"seeds":[],"surfaces":{},"contrasts":{}}
            seed_tables = {s:[] for s in SURFACES}
            for seed in SEEDS:
                folder=OUT/"runs"/f"{arm}_b{budget}_sub20260910_s{seed}"
                path=folder/"results.json"
                r=json.loads(path.read_text())
                paths.append(path)
                row["seeds"].append({"seed":seed,"step":r["best_step"],
                    "inner_at1":max(h["inner"]["selection"]["achieved_at_1"] for h in r["history"]),
                    **{s:r["surfaces"][s]["selection"]["achieved_at_1"] for s in SURFACES}})
                for surface in SURFACES:
                    p=folder/f"{surface}.parquet"
                    paths.append(p)
                    seed_tables[surface].append(group_table(pd.read_parquet(p),source=surface.startswith("source")))
            for surface in SURFACES:
                tt=seed_tables[surface]
                if any(not t.index.equals(tt[0].index) for t in tt):
                    raise ValueError("Seed population mismatch")
                mean=tt[0].copy()
                mean["achieved"]=np.mean([t.achieved.to_numpy() for t in tt],axis=0)
                mean["regret"]=mean.oracle-mean.achieved
                tables[(budget,arm,surface)]=mean
                row["surfaces"][surface]={"achieved":float(mean.achieved.mean()),"regret":float(mean.regret.mean()),
                    "seed_sd":float(np.std([t.achieved.mean() for t in tt],ddof=1))}
                row["contrasts"][surface+"_minus_start"]=contrast(mean,baseline[surface])
                if surface=="target_outer_val":
                    row["contrasts"]["target_minus_optiprime"]=contrast(mean,baseline["optiprime"])
            records.append(row)
    direct=[]
    for budget in (200,1000):
        for a,b in (("E","A"),("F","A"),("E","D"),("F","E")):
            for surface in SURFACES:
                d=contrast(tables[(budget,a,surface)],tables[(budget,b,surface)])
                direct.append({"budget":budget,"left":a,"right":b,"surface":surface,**d})
    output={"note":"Same target subset, three optimizer seeds; exploratory development intervals condition on selected models. Means are outcomes, not an ensemble.",
            "runs":records,"direct_contrasts":direct,"inputs_sha256":provenance(paths+[OUT/"summarize_replication.py"])}
    write_json(OUT/"replication_results.json",output)
    lines=["# V3 replication: three optimizer seeds", "",output["note"],"",
        "| Total nominal groups | Arm | Target @1 mean (seed SD) | Delta vs OptiPrime [95% component CI] | Source @1 mean | Source change [95% component CI] |",
        "|---:|---|---:|---|---:|---|"]
    for r in records:
        t=r['surfaces']['target_outer_val']; s=r['surfaces']['source_audit']
        o=r['contrasts']['target_minus_optiprime']; c=r['contrasts']['source_audit_minus_start']
        lines.append(f"| {r['budget']} | {r['arm']}: {NAMES[r['arm']]} | {t['achieved']:.6f} ({t['seed_sd']:.6f}) | {o['observed']:+.6f} [{o['ci95'][0]:+.6f}, {o['ci95'][1]:+.6f}] | {s['achieved']:.6f} | {c['observed']:+.6f} [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] |")
    lines += ["", "## Direct mean-seed contrasts", "", "| Budget | Contrast | Surface | Difference | 95% interval |", "|---:|---|---|---:|---|"]
    for d in direct:
        lines.append(f"| {d['budget']} | {d['left']} minus {d['right']} | {d['surface']} | {d['observed']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
    lines += ["", "## Individual seeds", "", "| Budget | Arm | Seed | Selected step | Inner @1 | Outer target @1 | Source audit @1 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in records:
        for s in r['seeds']:
            lines.append(f"| {r['budget']} | {r['arm']} | {s['seed']} | {s['step']} | {s['inner_at1']:.6f} | {s['target_outer_val']:.6f} | {s['source_audit']:.6f} |")
    lines += ["", "No target-test or acquired independent-panel predictions were made. A single target subset",
        "does not measure acquisition/subsampling uncertainty. No multiplicity correction is applied;",
        "these are development results, not confirmatory superiority tests."]
    (OUT/"REPLICATION_RESULTS.md").write_text("\n".join(lines)+"\n")
    print("Replicated results written",flush=True)

if __name__ == "__main__":
    main()
