"""Generate exploratory result tables/paired component intervals from completed fits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ROOT, OUT, CACHE, provenance, write_json
from e27_adaptation_test import reference_scores
from e29_summarize_audit import paired_interval

NAMES = {"A":"Native ordinal", "B":"Direct Huber", "C":"Fresh pairwise", "D":"Shared Huber + pairwise",
         "E":"Anchored pairwise", "F":"Anchored + preservation", "G":"Anchored utility + preservation",
         "C_frozen":"Frozen fresh pairwise", "F_replay":"Anchored + source-label replay"}

def group_table(f, source=False):
    if f.duplicated(["edit_key","design_key"]).any():
        raise ValueError("Expected unique candidates")
    t = f.groupby("edit_key").agg(site=("component","first"),depth=("y","size"),
                                  oracle=("y","max"),minimum=("y","min"))
    top = f.sort_values(["edit_key","selection","design_key"],ascending=[True,False,True]).drop_duplicates("edit_key")
    t["achieved"] = top.set_index("edit_key").y
    t["regret"] = t.oracle-t.achieved
    if source:
        t = t.loc[(t.depth>=2)&(t.oracle>t.minimum)]
    return t.sort_index()

def contrast(a,b):
    if not a.index.equals(b.index):
        raise ValueError("Population mismatch")
    return paired_interval(a.achieved-b.achieved,a.site)

def main():
    torch.set_num_threads(2)
    results = sorted((OUT/"runs").glob("*/results.json"))
    if len(results) < 14:
        raise RuntimeError(f"Screen still incomplete: {len(results)} results")
    index = pd.read_parquet(CACHE/"index.parquet")
    data = torch.load(CACHE/"prepared.pt",map_location="cpu",weights_only=False)
    q0 = data["q0"].numpy()
    base, basemetrics = {}, {}
    for surface in ("target_outer_val","source_val","source_audit"):
        f = index.loc[index.surface==surface,["row","group_id","design_key","component","y"]].rename(columns={"group_id":"edit_key"}).copy()
        f["selection"] = q0[f.row.to_numpy()]
        base[surface] = group_table(f,source=surface.startswith("source"))
        basemetrics[surface] = {"achieved":float(base[surface].achieved.mean()),"groups":len(base[surface])}
        if surface == "target_outer_val":
            op = f.drop(columns="selection").merge(reference_scores(),on=["edit_key","design_key"],validate="1:1")
            base["optiprime"] = group_table(op.assign(selection=op.op))
            basemetrics["optiprime"] = {"achieved":float(base["optiprime"].achieved.mean()),"groups":len(base["optiprime"])}
    records, tables = [], {}
    for path in results:
        r = json.loads(path.read_text())
        prov = json.loads((path.parent/"provenance.json").read_text())
        row = {k:r[k] for k in ("arm","budget","seed","subset_seed","best_step","training_seconds","total_seconds","replay_rows_processed")}
        row["tag"] = path.parent.name
        row["label_budget"] = prov["label_budget"]
        row["surfaces"], row["contrasts"] = {}, {}
        for surface in ("target_outer_val","source_val","source_audit"):
            t = group_table(pd.read_parquet(path.parent/f"{surface}.parquet"),source=surface.startswith("source"))
            tables[(row["tag"],surface)] = t
            row["surfaces"][surface] = {"achieved":float(t.achieved.mean()),"regret":float(t.regret.mean()),"groups":len(t)}
            recorded = r["surfaces"][surface]["selection"]["achieved_at_1"]
            if not np.isclose(recorded,t.achieved.mean(),atol=1e-12,rtol=0):
                raise ValueError("Independent endpoint reimplementation disagrees")
            row["contrasts"][surface+"_minus_start"] = contrast(t,base[surface])
            if surface == "target_outer_val":
                row["contrasts"]["target_minus_optiprime"] = contrast(t,base["optiprime"])
        records.append(row)
    direct = []
    for b in (200,1000):
        selected = [r for r in records if r["budget"]==b and r["seed"]==20260910]
        byarm = {r["arm"]:r for r in selected}
        for left,right in (("E","A"),("F","A"),("F","E"),("G","F"),("D","C"),("F","C_frozen"),("F","F_replay")):
            if left not in byarm or right not in byarm:
                continue
            for surface in ("target_outer_val","source_val","source_audit"):
                direct.append({"budget":b,"left":left,"right":right,"surface":surface,
                    **contrast(tables[(byarm[left]["tag"],surface)],tables[(byarm[right]["tag"],surface)])})
    source_mix = index.groupby(["surface","source_study"]).agg(candidates=("gid","size"),groups=("gid","nunique")).reset_index().to_dict("records")
    output = {"status":"Exploratory development screen, not confirmation",
              "baselines":basemetrics,"runs":records,"direct_contrasts":direct,
              "source_mix":source_mix,"total_process_hours":sum(r["total_seconds"] for r in records)/3600,
              "input_sha256":provenance(results+[OUT/"summarize.py",CACHE/"prepared.json"])}
    write_json(OUT/"screen_results.json",output)
    lines = ["# V3 initial screen: computed results", "", "All checkpoint choices use budget-internal validation;",
        "the 5,561-group outer validation is an exposed retrospective evaluation, not a new test.",
        "One optimizer seed and one subset seed per arm/budget; intervals are exploratory and unadjusted.",
        "", f"Frozen start target @1: {basemetrics['target_outer_val']['achieved']:.6f}; released OptiPrime: {basemetrics['optiprime']['achieved']:.6f}.",
        f"Frozen start source-audit @1: {basemetrics['source_audit']['achieved']:.6f} ({basemetrics['source_audit']['groups']} informative groups).", "",
        "| Total nominal groups | Arm | Selected step | Target @1 | Delta vs OptiPrime [95% component CI] | Source @1 | Delta vs source start |", "|---:|---|---:|---:|---|---:|---:|"]
    for r in sorted(records,key=lambda r:(r['budget'],r['arm'],r['seed'])):
        c = r['contrasts']['target_minus_optiprime']
        lo,hi = c['ci95']
        d = r['contrasts']['source_audit_minus_start']['observed']
        lines.append(f"| {r['budget']} | {r['arm']}: {NAMES[r['arm']]} | {r['best_step']} | {r['surfaces']['target_outer_val']['achieved']:.6f} | {c['observed']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {r['surfaces']['source_audit']['achieved']:.6f} | {d:+.6f} |")
    lines += ["", "Source @1 is the actual deployed selector, not its original prediction head.",
        "Efficiencies are fractions; multiply differences by 100 for percentage points.", "",
        "## Direct exploratory contrasts", "", "| Budget | Contrast | Surface | Difference | 95% component interval |", "|---:|---|---|---:|---|"]
    for d in direct:
        lo,hi=d['ci95']
        lines.append(f"| {d['budget']} | {d['left']} minus {d['right']} | {d['surface']} | {d['observed']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "## Resource and supervision accounting", "",
        f"{len(records)} completed fits, {sum(r['total_seconds'] for r in records)/3600:.3f} summed process-hours (including evaluation/I/O; not pure GPU kernel-hours).",
        "Each fit has 100 target updates; preservation adds replay work recorded per run.",
        "Nominal 200 actually labels 202 groups / 788 candidates (168 train + 34 validation groups).",
        "Nominal 1,000 actually labels 1,001 groups / 3,848 candidates (839 train + 162 validation groups).",
        "Source replay uses existing source supervision/our own teacher, not additional target labels.", "",
        "## Source composition", "", "| Surface | Study | Candidates | Groups |", "|---|---|---:|---:|"]
    for r in source_mix:
        lines.append(f"| {r['surface']} | {r['source_study']} | {r['candidates']} | {r['groups']} |")
    lines += ["", "See `screen_results.json` for all per-arm paired intervals, full precision and hashes.",
              "These results do not establish a universal low-label threshold or reject all replay/residual methods."]
    (OUT/"SCREEN_RESULTS.md").write_text("\n".join(lines)+"\n")
    # Compact Pareto display: both axes describe the same deployed selector.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    for ax,b in zip(axes,(200,1000)):
        for r in records:
            if r['budget']!=b:
                continue
            x=r['contrasts']['source_audit_minus_start']['observed']*100
            y=r['contrasts']['target_outer_val_minus_start']['observed']*100
            ax.scatter(x,y,s=40)
            ax.annotate(r['arm'],(x,y),xytext=(4,4),textcoords="offset points")
        ax.axvline(0,color="gray",lw=.8)
        ax.axhline(0,color="gray",lw=.8)
        ax.set_title(f"Nominal total budget {b}")
        ax.set_xlabel("Source selection change (percentage points)")
    axes[0].set_ylabel("Target selection change (percentage points)")
    fig.suptitle("V3 development screen: target utility versus deployed-source retention")
    fig.tight_layout()
    fig.savefig(OUT/"screen_pareto.svg")
    print(f"Summarized {len(records)} completed fits",flush=True)

if __name__ == "__main__":
    main()
