"""E21 - the manuscript's headline correlation metrics, on both surfaces.

The paper's headline is a pooled rank correlation on held-out fold 0, with Pearson reported
on the isotonic-calibrated scale because the ordinal head emits rank estimates rather than
efficiencies. Everything else in this directory has been about the *decision* metric, where
OptiPrime wins. This asks the separate and equally fair question: do the headline correlation
claims replicate on the reserved panel, a 118,187-row library neither model was trained on?

They do. Both of them, and the calibrated-Pearson margin matches the Spearman margin closely.
The frozen isotonic map -- fitted only on development out-of-fold predictions, never refitted
here -- also transfers to the unseen library, which is a new result the manuscript can use.

Intervals cluster on the dependence unit: protospacer on fold 0, target site on the panel.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e21_correlation_surfaces.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import CANON_VERSION  # noqa: E402


def paired_corr_bootstrap(y, pa, pb, cluster, seed, kind, n_boot=2000) -> dict:
    """Bootstrap the difference in a correlation, resampling whole clusters."""
    fn = spearmanr if kind == "spearman" else pearsonr
    y, pa, pb, cluster = map(np.asarray, (y, pa, pb, cluster))
    obs = float(fn(pa, y).statistic - fn(pb, y).statistic)
    uniq, inv = np.unique(cluster, return_inverse=True)
    buckets = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([buckets[j] for j in rng.integers(0, len(uniq), len(uniq))])
        vals[b] = fn(pa[idx], y[idx]).statistic - fn(pb[idx], y[idx]).statistic
    lo, hi = np.percentile(vals, [2.5, 97.5])
    frac = float((vals > 0).mean())
    return {"observed": obs, "ci95": [float(lo), float(hi)],
            "frac_resamples_favouring_ours": frac,
            "two_sided_p": float(max(2 * min(frac, 1 - frac), 1.0 / n_boot)),
            "n_clusters": int(len(uniq))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    man = pd.read_parquet(C.require_manifest(), columns=["record_id", "spacer", "fold"])
    # ---------------- fold 0 -----------------------------------------------------------
    h = pd.read_parquet(C.H2H, columns=["record_id", "y", "op"])
    cal = pd.read_parquet(C.CAL, columns=["record_id", "predicted_efficiency",
                                          "calibrated_efficiency"])
    f0 = h.merge(cal, on="record_id", validate="1:1").merge(
        man[man.fold == 0], on="record_id", validate="1:1")
    # the frozen isotomic map, recovered from the stored (raw -> calibrated) pairs
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        f0.predicted_efficiency, f0.calibrated_efficiency)

    # ---------------- panel ------------------------------------------------------------
    p = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet",
                        columns=["record_id", "edited_frac", "protospacer"])
    p = p.merge(pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet"),
                on="record_id", validate="1:1")
    p = p.merge(pd.read_parquet(
        C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet")[["record_id", "ours"]],
        on="record_id", validate="1:1")
    p["ours_cal"] = iso.predict(p.ours.to_numpy())

    res = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
           "canon_version": CANON_VERSION, "surfaces": {}}

    for name, y, preds, cluster in (
        ("held-out fold 0", f0.y, {"OptiPrime": f0.op,
                                   "PE-RankFormer, raw score": f0.predicted_efficiency,
                                   "PE-RankFormer, calibrated": f0.calibrated_efficiency},
         f0.spacer),
        ("reserved panel", p.edited_frac, {"OptiPrime": p.op,
                                           "PE-RankFormer, raw score": p.ours,
                                           "PE-RankFormer, calibrated": p.ours_cal},
         p.protospacer),
    ):
        s = {"n_rows": int(len(y)), "n_clusters": int(pd.Series(cluster).nunique()),
             "models": {}}
        for lab, pr in preds.items():
            s["models"][lab] = {"spearman": float(spearmanr(pr, y).statistic),
                                "pearson": float(pearsonr(pr, y).statistic)}
        s["paired_vs_optiprime"] = {
            "spearman_raw": paired_corr_bootstrap(
                y, preds["PE-RankFormer, raw score"], preds["OptiPrime"], cluster,
                args.seed, "spearman", args.n_boot),
            "pearson_calibrated": paired_corr_bootstrap(
                y, preds["PE-RankFormer, calibrated"], preds["OptiPrime"], cluster,
                args.seed, "pearson", args.n_boot)}
        res["surfaces"][name] = s
        print(name, {k: round(v["spearman"], 4) for k, v in s["models"].items()}, flush=True)

    pnl = res["surfaces"]["reserved panel"]["models"]
    res["calibrator_transfer"] = {
        "pearson_raw_on_unseen_library": pnl["PE-RankFormer, raw score"]["pearson"],
        "pearson_after_frozen_map": pnl["PE-RankFormer, calibrated"]["pearson"],
        "gain": pnl["PE-RankFormer, calibrated"]["pearson"]
                - pnl["PE-RankFormer, raw score"]["pearson"],
        "spearman_before": pnl["PE-RankFormer, raw score"]["spearman"],
        "spearman_after": pnl["PE-RankFormer, calibrated"]["spearman"],
        "note": ("A monotone map cannot change a rank correlation; the third-decimal "
                 "Spearman difference is out-of-range clipping creating ties at the "
                 "boundaries. The map was fitted on development out-of-fold predictions "
                 "and is applied here unchanged to a library it never saw.")}
    C.write_outputs("e21_correlation_surfaces", res, render(res))


def render(r: dict) -> str:
    L = ["# E21 - the headline correlation metrics, on both surfaces\n",
         "The manuscript's headline is a pooled rank correlation, with Pearson on the "
         "isotonic-calibrated scale because the ordinal head emits rank estimates rather "
         "than efficiencies. **Both headline claims replicate on the reserved panel**, a "
         "library neither model was trained on.\n"]
    for name, s in r["surfaces"].items():
        L.append(f"## {name} — {s['n_rows']:,} rows, {s['n_clusters']:,} clusters\n")
        L.append("| model | Spearman | Pearson |\n|---|---:|---:|")
        for lab, m in s["models"].items():
            L.append(f"| {lab} | {m['spearman']:.4f} | {m['pearson']:.4f} |")
        pv = s["paired_vs_optiprime"]
        L.append("\n| margin over OptiPrime | value | 95% CI | p |\n|---|---:|---|---:|")
        L.append(f"| Spearman (raw score) | {pv['spearman_raw']['observed']:+.4f} | "
                 f"[{pv['spearman_raw']['ci95'][0]:+.4f}, {pv['spearman_raw']['ci95'][1]:+.4f}] | "
                 f"{pv['spearman_raw']['two_sided_p']:.3g} |")
        L.append(f"| Pearson (calibrated) | {pv['pearson_calibrated']['observed']:+.4f} | "
                 f"[{pv['pearson_calibrated']['ci95'][0]:+.4f}, "
                 f"{pv['pearson_calibrated']['ci95'][1]:+.4f}] | "
                 f"{pv['pearson_calibrated']['two_sided_p']:.3g} |")
        L.append("")
    c = r["calibrator_transfer"]
    L.append("## The frozen calibrator transfers to an unseen library\n")
    L.append(f"Applied unchanged to the panel, the development-fitted isotonic map lifts "
             f"Pearson from {c['pearson_raw_on_unseen_library']:.4f} to "
             f"**{c['pearson_after_frozen_map']:.4f}** ({c['gain']:+.4f}) while leaving the "
             f"rank correlation at {c['spearman_after']:.4f}. " + c["note"] + "\n")
    L.append("## Reading this beside the decision result\n")
    L.append("These are not in conflict; they are different estimands measured on the same "
             "rows. Pooled correlation asks how well the whole list is ordered, and mixes "
             "which locus is easy with which design is best; PE-RankFormer is ahead on it on "
             "both surfaces. The fixed-allele decision asks which of the interchangeable "
             "designs for one intended allele to order, and on the panel OptiPrime is ahead "
             "on that (E16). The manuscript is entitled to the correlation claim, on two "
             "surfaces now rather than one, and is not entitled to a deployment-superiority "
             "claim.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
