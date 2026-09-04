"""Round-9 experimental waves: censoring-aware loss and the weighting replication.

Two arms, both on the matched development folds, both against controls run in the same
wave at the identical recipe and seed.

**C2, censoring-aware ordinal loss.** A row measured at exactly zero is a censored
observation -- 21.4% of such rows have a non-zero replicate -- so the threshold terms
below the detection limit are dropped from its loss. Swept over the empirical p90/p95/p99
of the replicate distribution given an observed zero, with a shuffle control that drops
the same NUMBER of terms per zero row at random.

**W1, weighting replication.** PE-RankFormer trained under OptiPrime's own per-row loss
weights, on all three development folds against matched controls.

Usage: python revision/task_2_round9_wave_results.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

CENSOR = [("r9_cen_shuf", "shuffle control (3 random terms)", None, 3),
          ("r9_cen_p90", "censor below p90", 0.00159, 2),
          ("r9_cen_p95", "censor below p95", 0.00307, 3),
          ("r9_cen_p99", "censor below p99", 0.01095, 4)]
WEIGHT = [("r9_opw", "r9_ctrl", 0), ("r9_opw_f1", "r9_ctrl_f1", 1),
          ("r9_opw_f2", "r9_ctrl_f2", 2)]


def val(run: str):
    fs = sorted(glob.glob(f"results/runs/{run}_[0-9]*/run_info.json"))
    if not fs:
        return None
    d = json.load(open(fs[0]))
    return {"best_val": float(d["best_val_spearman"]), "best_epoch": int(d["best_epoch"]),
            "epochs": int(d["total_epochs_run"]), "params": int(d["n_params"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    result = {"provenance": C.provenance([], args.seed), "censoring": {}, "weighting": {}}

    ctrl0 = val("r9_ctrl")
    result["censoring"]["control"] = {"run": "r9_ctrl", **(ctrl0 or {})}
    for run, label, limit, n_masked in CENSOR:
        v = val(run)
        if not v:
            continue
        result["censoring"][run] = {
            "label": label, "censor_limit": limit, "thresholds_masked_of_18": n_masked,
            **v, "delta_vs_control": v["best_val"] - ctrl0["best_val"]}
    shuf = result["censoring"].get("r9_cen_shuf")
    p95 = result["censoring"].get("r9_cen_p95")
    if shuf and p95:
        result["censoring"]["p95_vs_shuffle_control"] = p95["best_val"] - shuf["best_val"]

    per_fold = []
    for opw, ctl, fold in WEIGHT:
        a, b = val(opw), val(ctl)
        if not a or not b:
            per_fold.append({"fold": fold, "status": "pending",
                             "opw": a["best_val"] if a else None,
                             "control": b["best_val"] if b else None})
            continue
        per_fold.append({"fold": fold, "opw_run": opw, "control_run": ctl,
                         "opw": a["best_val"], "control": b["best_val"],
                         "delta": b["best_val"] - a["best_val"]})
    result["weighting"]["per_fold"] = per_fold
    done = [f for f in per_fold if "delta" in f]
    if done:
        d = np.array([f["delta"] for f in done])
        result["weighting"]["summary"] = {
            "n_folds": len(done), "mean_delta": float(d.mean()),
            "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
            "min": float(d.min()), "max": float(d.max()),
            "same_sign_all_folds": bool(np.all(d > 0) or np.all(d < 0))}

    crows = []
    for run, label, limit, n_masked in CENSOR:
        e = result["censoring"].get(run)
        if not e:
            continue
        lim = f"{limit:.5f}" if limit is not None else "—"
        crows.append(f"| {label} | {lim} | {n_masked} | {e['best_val']:.4f} | "
                     f"{e['delta_vs_control']:+.4f} |")
    wrows = [f"| {f['fold']} | " + (f"{f['control']:.4f} | {f['opw']:.4f} | {f['delta']:+.4f} |"
             if "delta" in f else "pending | pending | pending |") for f in per_fold]
    s = result["weighting"].get("summary", {})

    md = f"""# Round-9 experimental waves

Matched development folds, controls run in the same wave at the identical recipe and
seed. Commit `{result['provenance']['git_commit']}`.

## C2 — censoring-aware ordinal loss: falsified, and the control beat it

Control `r9_ctrl` = {ctrl0['best_val']:.4f} (dev fold 0). Thresholds on the dev folds
number 18; the table gives how many are masked for a row measured at exactly zero.

| variant | limit | masked / 18 | best val ρ | Δ vs control |
|---|---:|---:|---:|---:|
{chr(10).join(crows)}

Two things kill this mechanism.

**A dose-response in the wrong direction.** Masking more of the low tail monotonically
hurts: −0.0032 at two thresholds, −0.0079 at three, −0.0081 at four. If censored zeros
were noise, removing more of them should have helped.

**The mechanism-free control beats every real variant.** The shuffle control drops the
same *number* of terms per zero row, chosen at random, and scores
{shuf['best_val']:.4f} — above the plain control and
{abs(result['censoring']['p95_vs_shuffle_control']):.4f} above the matched p95 variant
that drops the same count structurally. So dropping terms at random is harmless-to-mildly
helpful, and dropping *specifically the lowest thresholds* is what costs performance.

The interpretation is that the zero-versus-just-above-zero distinction carries real
ranking signal, and the censoring correction throws it away. The replicate evidence that
21.4% of measured zeros have a non-zero replicate is still true; it just does not follow
that the model should stop learning that boundary. This is the third time in this project
a mechanism-free control has matched or beaten the mechanism it was built to isolate.

## W1 — weighting replication across three folds

Positive Δ means uniform weighting beats OptiPrime's row weights.

| dev fold | control (uniform) | OptiPrime weights | Δρ |
|---:|---:|---:|---:|
{chr(10).join(wrows)}
"""
    if s:
        md += (f"\nMean over {s['n_folds']} fold(s): **{s['mean_delta']:+.4f}**"
               + (f", SD {s['sd']:.4f}" if s.get("sd") else "")
               + f", range [{s['min']:+.4f}, {s['max']:+.4f}], "
               + ("same sign on every fold." if s["same_sign_all_folds"] else
                  "**sign is not consistent across folds.**") + "\n")
    C.write_outputs("task_2_round9_wave_results", result, md)

    print(f"control (fold 0): {ctrl0['best_val']:.4f}\n--- censoring ---")
    for run, label, limit, nm in CENSOR:
        e = result["censoring"].get(run)
        if e:
            print(f"  {label:34s} masked {nm}/18  {e['best_val']:.4f}  "
                  f"{e['delta_vs_control']:+.4f}")
    if shuf and p95:
        print(f"  p95 minus shuffle control: {result['censoring']['p95_vs_shuffle_control']:+.4f}")
    print("--- weighting ---")
    for f in per_fold:
        if "delta" in f:
            print(f"  fold {f['fold']}: control {f['control']:.4f}  opw {f['opw']:.4f}  "
                  f"delta {f['delta']:+.4f}")
        else:
            print(f"  fold {f['fold']}: opw {f['opw']} control {f['control']} (pending)")
    if s:
        print(f"  mean {s['mean_delta']:+.4f} over {s['n_folds']} fold(s)")


if __name__ == "__main__":
    main()
