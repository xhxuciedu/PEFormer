"""Verify that every number asserted in the manuscript prose matches its artifact.

The result tables are generated from JSON by `revision/make_paper_tables.py`, so they
cannot drift. The prose is written by hand and can, and during this revision it did
twice: a row count and a weighted mean were transcribed from memory and were wrong.
This script closes that gap by checking each numeric claim in the prose against the file
that produced it.

Each check names the value as it appears in the manuscript, the artifact it comes from,
and the path into that artifact. A check fails if the manuscript no longer contains the
artifact's value, formatted as the manuscript formats it -- so editing either side
without the other is caught.

Usage:  python reproduce/verify_manuscript.py [--tex PATH]
Exit code 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "reports/paper/pe_rankformer_paper.tex"


def get(obj, path: str):
    """Walk a dotted path into nested dicts/lists, e.g. 'a.b.0.c'."""
    for key in path.split("."):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


# (description, artifact, json path, format, transform)
CHECKS: list[tuple[str, str, str, str, object]] = [
    # ---- headline comparison ----
    ("pooled Spearman, ours", "results/round4/final_bootstrap.json",
     "observed.round4", ".4f", None),
    ("pooled Spearman, OptiPrime", "results/round4/final_bootstrap.json",
     "observed.optiprime", ".4f", None),
    ("pooled margin", "results/round4/final_bootstrap.json",
     "round4_vs_optiprime.observed_difference", "+.4f", None),
    ("pooled margin CI low", "results/round4/final_bootstrap.json",
     "round4_vs_optiprime.ci95.0", "+.4f", None),
    ("pooled margin CI high", "results/round4/final_bootstrap.json",
     "round4_vs_optiprime.ci95.1", "+.4f", None),
    ("absolute CI low", "results/round4/final_bootstrap.json",
     "round4_absolute.ci95.0", ".4f", None),
    ("absolute CI high", "results/round4/final_bootstrap.json",
     "round4_absolute.ci95.1", ".4f", None),
    ("Kim held-out, ours", "results/round4/final_bootstrap.json",
     "by_study.deepprime.round4", ".4f", None),
    ("Liu held-out, ours", "results/round4/final_bootstrap.json",
     "by_study.hsu2026.round4", ".4f", None),
    # ---- partition-wise significance ----
    ("Liu margin", "results/round9/partition_bootstrap.json", "liu.delta", "+.4f", None),
    ("Liu margin CI low", "results/round9/partition_bootstrap.json",
     "liu.ci95.0", "+.4f", None),
    ("Kim margin", "results/round9/partition_bootstrap.json", "kim.delta", "+.4f", None),
    # ---- per-target (task 1.1) ----
    ("within-target, ours", "revision/task_1_1_per_target.json",
     "per_target.ours.mean", ".4f", None),
    ("within-target, OptiPrime", "revision/task_1_1_per_target.json",
     "per_target.optiprime.mean", ".4f", None),
    ("within-target margin", "revision/task_1_1_per_target.json",
     "per_target.bootstrap_mean_delta_unweighted.observed", "+.4f", None),
    ("targets scored", "revision/task_1_1_per_target.json",
     "per_target.n_targets_scored", "d", None),
    ("targets favouring ours (%)", "revision/task_1_1_per_target.json",
     "per_target.delta.frac_targets_favouring_ours", ".1f", lambda v: 100 * v),
    ("within-condition margin", "revision/task_1_1_per_target.json",
     "per_condition.delta_n_weighted", "+.4f", None),
    # ---- utility (task 1.2) ----
    ("precision@1, ours (%)", "revision/task_1_2_utility.json",
     "metrics.p_at_1.ours", ".1f", lambda v: 100 * v),
    ("precision@1, OptiPrime (%)", "revision/task_1_2_utility.json",
     "metrics.p_at_1.optiprime", ".1f", lambda v: 100 * v),
    ("top-pick efficiency delta", "revision/task_1_2_utility.json",
     "metrics.top1_eff.ours_vs_optiprime.observed", "+.3f", None),
    # ---- tie-robust (task 1.3) ----
    ("Kendall tau-b margin", "revision/task_1_3_tie_robust.json",
     "metrics.all.delta.kendall_tau_b", "+.4f", None),
    ("AUROC margin", "revision/task_1_3_tie_robust.json",
     "metrics.all.delta.auroc_edits_at_all", "+.4f", None),
    ("Spearman|y>0 margin", "revision/task_1_3_tie_robust.json",
     "metrics.all.delta.spearman_editing_only", "+.4f", None),
    # ---- leakage (task 1.4) ----
    ("twinned held-out rows", "revision/task_1_4_leakage_free.json",
     "counts.rows_with_a_training_twin", "d", None),
    ("leakage-free Spearman", "revision/task_1_4_leakage_free.json",
     "subsets.leakage_free.ours", ".4f", None),
    ("leakage-free margin", "revision/task_1_4_leakage_free.json",
     "subsets.leakage_free.delta", "+.4f", None),
    # ---- calibration floors (task 1.5) ----
    ("trivial-floor MAE", "revision/task_1_5_calibration_baselines.json",
     "absolute_error.constant at training median.mae", ".4f", None),
    ("calibrated MAE", "revision/task_1_5_calibration_baselines.json",
     "absolute_error.PE-RankFormer + isotonic.mae", ".4f", None),
    ("OptiPrime MAE", "revision/task_1_5_calibration_baselines.json",
     "absolute_error.OptiPrime.mae", ".4f", None),
    ("top-1% predicted", "revision/task_1_5_calibration_baselines.json",
     "top_1pct.mean_predicted", ".3f", None),
    ("top-1% observed", "revision/task_1_5_calibration_baselines.json",
     "top_1pct.mean_observed", ".3f", None),
    # ---- ceiling (task 1.6) ----
    ("held-out headroom gap", "revision/task_1_6_ceiling_uncertainty.json",
     "surfaces.heldout.gap_point", "+.4f", None),
    ("headroom CI low", "revision/task_1_6_ceiling_uncertainty.json",
     "surfaces.heldout.gap_ci95.0", "+.4f", None),
    ("headroom CI high", "revision/task_1_6_ceiling_uncertainty.json",
     "surfaces.heldout.gap_ci95.1", "+.4f", None),
    ("replicate groups", "revision/task_1_6_ceiling_uncertainty.json",
     "n_replicate_groups", "d", None),
    ("dev 3-fold headroom gap", "revision/task_1_6_ceiling_uncertainty.json",
     "surfaces", "+.4f",
     lambda sf: sum(sf[f"dev_fold_{i}"]["gap_point"] for i in range(3)) / 3),
    # ---- weighting (tasks 1.8, 2) ----
    ("weighting delta, fold 0", "revision/task_2_round9_wave_results.json",
     "weighting.per_fold.0.delta", "+.4f", None),
    ("weighting delta, mean of 3", "revision/task_2_round9_wave_results.json",
     "weighting.summary.mean_delta", "+.4f", None),
    ("weighting delta on Kim", "revision/task_1_8_weighting_by_partition.json",
     "partitions.deepprime.delta", "+.4f", None),
    ("weighting delta on Liu", "revision/task_1_8_weighting_by_partition.json",
     "partitions.hsu2026.delta", "+.4f", None),
    # ---- censoring arm ----
    ("censoring p95 delta", "revision/task_2_round9_wave_results.json",
     "censoring.r9_cen_p95.delta_vs_control", "+.4f", None),
    ("censoring vs shuffle control", "revision/task_2_round9_wave_results.json",
     "censoring.p95_vs_shuffle_control", "+.4f", None),
    # ---- feature baseline ----
    ("GBM pooled Spearman", "results/round9/gbm_baseline.json",
     "partitions.all.gbm", ".4f", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=TEX)
    args = ap.parse_args()
    tex = args.tex.read_text()
    # strip comments; the manuscript uses {,} inside numbers, so normalise those too
    body = "\n".join(l for l in tex.split("\n") if not l.strip().startswith("%"))
    body_norm = body.replace("{,}", ",")

    cache: dict[str, dict] = {}
    ok = bad = 0
    print(f"verifying manuscript prose against artifacts: {args.tex.relative_to(ROOT)}\n")
    print(f"{'claim':34s} {'value':>10s}  artifact")
    for desc, rel, path, fmt, tf in CHECKS:
        if rel not in cache:
            cache[rel] = load(rel)
        try:
            v = get(cache[rel], path)
        except (KeyError, IndexError, TypeError) as e:
            print(f"  MISSING  {desc:32s} -- {rel}:{path} ({e})")
            bad += 1
            continue
        if tf:
            v = tf(v)
        txt = f"{int(v):,}" if fmt == "d" else f"{v:{fmt}}"
        # a signed value may appear with or without an explicit +
        variants = {txt, txt.lstrip("+"), txt.replace(",", "{,}")}
        found = any(t in body_norm or t in body for t in variants)
        status = "ok " if found else "FAIL"
        if found:
            ok += 1
        else:
            bad += 1
        print(f"{status} {desc:30s} {txt:>10s}  {rel.split('/')[-1]}")
    print(f"\n{ok} of {ok + bad} checks passed")
    if bad:
        print("A FAIL means the manuscript no longer states that artifact's value. Either\n"
              "the prose drifted from the analysis, or the analysis was re-run and the\n"
              "prose needs updating. Regenerate tables with revision/make_paper_tables.py.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
