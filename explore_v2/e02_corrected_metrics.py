"""E02 - the same frozen predictions, scored on the decision a user actually faces.

Research plan section 8A, days 1-3: "Recalculate selection metrics descriptively and
report candidate coverage."

Four groupings of the same 20,509 held-out rows, from the loosest to the one that fixes
everything a user has already decided:

  1. `spacer`                     - the manuscript's grouping.
  2. `spacer x source/cell/editor` - condition held fixed, allele still free.
  3. `canonical allele x context`  - **the decision group**: allele and all nine context
                                     fields fixed, only the pegRNA molecule varies.
  4. pairwise within (3)           - the only depth fold 0 supports, scored as a choice.

Everything is descriptive: predictions are frozen files, no fitting of any kind. Intervals
are protospacer-clustered percentile bootstraps.

Usage: .venv/bin/python explore_v2/e02_corrected_metrics.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402

MODELS = ("ours", "ordssm", "op")
LABEL = {"ours": "PE-RankFormer (final ensemble)", "ordssm": "ordinal-S4D member",
         "op": "OptiPrime"}


def load() -> pd.DataFrame:
    man = pd.read_parquet(C.require_manifest())
    man = man[man.fold == 0]
    pred = C.load_heldout_predictions()
    d = man.merge(pred, on="record_id", validate="1:1")
    assert len(d) == 20509, len(d)
    # A replicate is not a candidate. Average the 5 metadata-identical replicate groups
    # that survive into fold 0 before any group is ranked.
    n_before = len(d)
    agg = {c: "first" for c in d.columns if c not in ("y", "ours", "ordssm", "op",
                                                      "record_id", "edited")}
    d = (d.groupby("replicate_key", observed=True)
           .agg(y=("y", "mean"), ours=("ours", "mean"), ordssm=("ordssm", "mean"),
                op=("op", "mean"), n_rep=("y", "size"),
                **{k: (k, v) for k, v in agg.items()})
           .reset_index(drop=True))
    d.attrs["rows_before_replicate_merge"] = n_before
    return d


def mean_spearman(d: pd.DataFrame, key: str, model: str, min_n: int) -> float:
    """Mean per-group Spearman over groups with >= min_n rows and variable outcome.

    Same eligibility rule as revision/task_1_1 so the numbers are comparable.
    """
    vals = []
    for _, s in d.groupby(key, observed=True):
        if len(s) < min_n:
            continue
        y = s.y.to_numpy()
        if np.ptp(y) == 0:
            continue
        vals.append(C.spearman(s[model].to_numpy(), y))
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def pair_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per two-candidate decision group.

    Fold 0's decision groups top out at four designs, and only 85 reach three, so the
    honest estimand is the binary choice. For each group we record which design is truly
    better, by how much, and what each predictor would have picked.
    """
    recs = []
    for g, s in d.groupby("decision_group", observed=True):
        if s.design_key.nunique() != 2 or len(s) != 2:
            continue
        y = s.y.to_numpy()
        r = {"decision_group": g, "spacer": s.spacer.iloc[0],
             "edit_key": s.edit_key.iloc[0], "cond3": s.cond3.iloc[0],
             "edit_type": s.edit_type.iloc[0],
             "y_hi": float(y.max()), "y_lo": float(y.min()),
             "gap": float(y.max() - y.min()), "tie": bool(y[0] == y[1]),
             "both_zero": bool(y.max() == 0)}
        for m in MODELS:
            p = s[m].to_numpy()
            pick = int(np.argmax(p))
            r[f"{m}_correct"] = float(y[pick] == y.max())
            r[f"{m}_picked"] = float(y[pick])
            r[f"{m}_regret"] = float(y.max() - y[pick])
        r["rand_picked"] = float(y.mean())
        r["rand_regret"] = float(y.max() - y.mean())
        recs.append(r)
    return pd.DataFrame(recs)


def paired_boot(t: pd.DataFrame, col_a: str, col_b: str, seed: int, n_boot: int) -> dict:
    """Bootstrap the paired difference col_a - col_b, resampling protospacers."""
    def stat(s: pd.DataFrame) -> float:
        d = s[col_a].to_numpy() - s[col_b].to_numpy()
        d = d[np.isfinite(d)]
        return float(d.mean()) if d.size else np.nan
    return C.cluster_bootstrap(t, stat, seed=seed, key="spacer", n_boot=n_boot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    d = load()
    res: dict = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
                 "rows_before_replicate_merge": d.attrs["rows_before_replicate_merge"],
                 "rows_after_replicate_merge": int(len(d))}

    # ------------------------------------------------- grouping ladder
    ladder = []
    d = d.assign(spacer_cond=d.spacer + "@" + d.cond3)
    for key, min_n, name in [("spacer", 5, "spacer (manuscript)"),
                             ("spacer_cond", 5, "spacer x source/cell/editor"),
                             ("decision_group", 3, "allele x context, >=3 designs"),
                             ("decision_group", 2, "allele x context, >=2 designs")]:
        g = d.groupby(key, observed=True)
        elig = sum(1 for _, s in g if len(s) >= min_n and np.ptp(s.y.to_numpy()) > 0)
        row = {"grouping": name, "key": key, "min_designs": min_n, "eligible_groups": elig}
        for m in MODELS:
            row[m] = mean_spearman(d, key, m, min_n)
        row["delta_ours_op"] = row["ours"] - row["op"]
        row["delta_ordssm_op"] = row["ordssm"] - row["op"]
        ladder.append(row)
    res["grouping_ladder"] = ladder

    # what a spacer group actually contains
    gs = d.groupby("spacer", observed=True).agg(n=("y", "size"), n_allele=("edit_key", "nunique"),
                                                n_cond=("cond3", "nunique"),
                                                n_design=("design_key", "nunique"))
    gs5 = gs[gs.n >= 5]
    res["spacer_group_composition"] = {
        "groups_ge5": int(len(gs5)),
        "median_rows": float(gs5.n.median()),
        "median_alleles": float(gs5.n_allele.median()),
        "median_conditions": float(gs5.n_cond.median()),
        "frac_multi_condition": float((gs5.n_cond > 1).mean()),
        "frac_multi_allele": float((gs5.n_allele > 1).mean()),
        "median_rows_per_allele_condition": float((gs5.n / (gs5.n_allele * gs5.n_cond)).median()),
    }

    # ----------------------------------------------------- pairwise decision
    t = pair_table(d)
    res["pairwise"] = {"groups": int(len(t)),
                       "spacers": int(t.spacer.nunique()),
                       "alleles": int(t.edit_key.nunique()),
                       "ties": int(t.tie.sum()),
                       "both_zero": int(t.both_zero.sum())}
    scorable = t[~t.tie].reset_index(drop=True)
    res["pairwise"]["scorable_groups"] = int(len(scorable))
    res["pairwise"]["gap_quantiles"] = {q: float(scorable.gap.quantile(q))
                                        for q in (0.25, 0.5, 0.75, 0.9)}
    res["pairwise"]["mean_gap"] = float(scorable.gap.mean())
    res["pairwise"]["mean_best"] = float(scorable.y_hi.mean())

    acc = {}
    for m in MODELS:
        acc[m] = {"accuracy": float(scorable[f"{m}_correct"].mean()),
                  "selected_efficiency": float(scorable[f"{m}_picked"].mean()),
                  "regret": float(scorable[f"{m}_regret"].mean())}
    acc["random"] = {"accuracy": 0.5,
                     "selected_efficiency": float(scorable.rand_picked.mean()),
                     "regret": float(scorable.rand_regret.mean())}
    acc["oracle"] = {"accuracy": 1.0,
                     "selected_efficiency": float(scorable.y_hi.mean()), "regret": 0.0}
    res["pairwise"]["metrics"] = acc
    res["pairwise"]["paired_tests"] = {
        f"{a}_minus_{b}_{metric}": paired_boot(scorable, f"{a}_{metric}",
                                               f"{b}_{metric}", args.seed, args.n_boot)
        for metric in ("correct", "picked", "regret")
        for a, b in (("ours", "op"), ("ordssm", "op"))}

    # gap-stratified: does the model get the decisions that matter?
    strat = []
    for lo, hi in [(0.0, 0.01), (0.01, 0.05), (0.05, 0.20), (0.20, 1.01)]:
        s = scorable[(scorable.gap >= lo) & (scorable.gap < hi)]
        if len(s) < 20:
            continue
        strat.append({"gap_range": [lo, hi], "n": int(len(s)),
                      "mean_gap": float(s.gap.mean()),
                      **{f"{m}_acc": float(s[f"{m}_correct"].mean()) for m in MODELS},
                      **{f"{m}_regret": float(s[f"{m}_regret"].mean()) for m in MODELS}})
    res["pairwise"]["by_gap"] = strat

    # per condition
    percond = []
    for c, s in scorable.groupby("cond3", observed=True):
        if len(s) < 25:
            continue
        percond.append({"cond3": c, "n": int(len(s)), "mean_gap": float(s.gap.mean()),
                        **{f"{m}_acc": float(s[f"{m}_correct"].mean()) for m in MODELS}})
    res["pairwise"]["by_condition"] = sorted(percond, key=lambda r: -r["n"])

    # ------------------------------------------- leakage-stratified pairwise
    man = pd.read_parquet(C.require_manifest(),
                          columns=["fold", "decision_group"])
    shared = set(man.loc[man.fold != 0, "decision_group"]) & set(
        man.loc[man.fold == 0, "decision_group"])
    scorable = scorable.assign(shared=scorable.decision_group.isin(shared))
    res["pairwise"]["by_train_overlap"] = [
        {"decision_group_seen_in_training": bool(k), "n": int(len(s)),
         **{f"{m}_acc": float(s[f"{m}_correct"].mean()) for m in MODELS},
         **{f"{m}_regret": float(s[f"{m}_regret"].mean()) for m in MODELS}}
        for k, s in scorable.groupby("shared")]

    t.to_csv(C.OUT / "e02_pairwise_table.csv", index=False)
    C.write_outputs("e02_corrected_metrics", res, render(res))


def render(r: dict) -> str:
    L = ["# E02 - the frozen predictions scored on the user's decision\n",
         "Descriptive re-scoring of frozen predictions. No fitting. Intervals are "
         "protospacer-clustered percentile bootstraps.\n",
         f"The five metadata-identical replicate groups inside fold 0 are averaged first, "
         f"taking {r['rows_before_replicate_merge']:,} rows to {r['rows_after_replicate_merge']:,} "
         "candidate measurements: a repeat measurement is not an alternative design.\n",
         "## 1. The grouping ladder\n",
         "| grouping | eligible groups | OptiPrime | ordinal-S4D | final ensemble | delta (ens - OP) |",
         "|---|---:|---:|---:|---:|---:|"]
    for row in r["grouping_ladder"]:
        L.append(f"| {row['grouping']} (>= {row['min_designs']}) | {row['eligible_groups']:,} | "
                 f"{row['op']:.4f} | {row['ordssm']:.4f} | {row['ours']:.4f} | "
                 f"{row['delta_ours_op']:+.4f} |")
    sc = r["spacer_group_composition"]
    L.append(f"\nA protospacer group of >=5 rows contains a median of {sc['median_alleles']:.0f} "
             f"intended alleles across {sc['median_conditions']:.0f} conditions, i.e. a median of "
             f"{sc['median_rows_per_allele_condition']:.1f} rows per allele-and-condition cell. "
             f"{sc['frac_multi_condition']:.0%} of those groups mix conditions and "
             f"{sc['frac_multi_allele']:.0%} mix alleles. Ranking inside such a group is mostly "
             "the question *which edit, in which cell line, is easier* - not *which pegRNA should "
             "I order*.\n")
    p = r["pairwise"]
    L.append("## 2. The decision fold 0 can actually score: pick one of two designs\n")
    L.append(f"{p['groups']:,} decision groups on fold 0 hold exactly two alternative designs for "
             f"one canonical allele in one fully specified context, spanning {p['spacers']:,} "
             f"protospacers. {p['ties']:,} are exact ties in the measured outcome "
             f"({p['both_zero']:,} of them both-zero) and cannot be scored, leaving "
             f"**{p['scorable_groups']:,} scorable binary choices**.\n")
    L.append("| predictor | accuracy | efficiency of the pick | regret |\n|---|---:|---:|---:|")
    for m in ("random", "op", "ordssm", "ours", "oracle"):
        v = p["metrics"][m]
        _ = v
        name = {"random": "random choice", "oracle": "oracle (best of the two)"}.get(m) or LABEL[m]
        L.append(f"| {name} | {v['accuracy']:.3f} | {v['selected_efficiency']:.4f} | {v['regret']:.4f} |")
    pt = p["paired_tests"]
    L.append("\n| paired difference | value | 95% CI | p |\n|---|---:|---|---:|")
    for k, v in pt.items():
        L.append(f"| {k.replace('_', ' ')} | {v['observed']:+.4f} | "
                 f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['two_sided_p']:.3g} |")
    L.append(f"\nThe whole decision is worth {p['mean_gap']:.4f} efficiency on average "
             f"(median gap {p['gap_quantiles'][0.5]:.4f}); the better of the two designs delivers "
             f"{p['mean_best']:.4f}. That is the ceiling any pegRNA-selection method can win here, "
             "and it bounds how much a ranking improvement can be worth at the bench.\n")
    L.append("### By how much the two designs actually differ\n")
    L.append("| measured gap | groups | mean gap | OP acc | S4D acc | ens acc | ens regret |\n|---|---:|---:|---:|---:|---:|---:|")
    for s in p["by_gap"]:
        L.append(f"| [{s['gap_range'][0]:.2f}, {s['gap_range'][1]:.2f}) | {s['n']:,} | "
                 f"{s['mean_gap']:.4f} | {s['op_acc']:.3f} | {s['ordssm_acc']:.3f} | "
                 f"{s['ours_acc']:.3f} | {s['ours_regret']:.4f} |")
    L.append("\n### By condition\n")
    L.append("| condition | groups | mean gap | OP acc | S4D acc | ens acc |\n|---|---:|---:|---:|---:|---:|")
    for s in p["by_condition"]:
        L.append(f"| {s['cond3']} | {s['n']:,} | {s['mean_gap']:.4f} | {s['op_acc']:.3f} | "
                 f"{s['ordssm_acc']:.3f} | {s['ours_acc']:.3f} |")
    L.append("\n### By whether training saw the same decision group\n")
    L.append("| decision group also in training | groups | OP acc | ens acc | ens regret |\n|---|---:|---:|---:|---:|")
    for s in p["by_train_overlap"]:
        L.append(f"| {s['decision_group_seen_in_training']} | {s['n']:,} | {s['op_acc']:.3f} | "
                 f"{s['ours_acc']:.3f} | {s['ours_regret']:.4f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
