"""E01 - canonical decision manifest and candidate-availability audit.

Research plan section 8A, days 1-3: "reconstruct decision keys; audit which rows are
alternative designs, distinct intended edits, and true replicates".

The question this answers is prior to any modelling. The manuscript's deployment metrics
(within-target Spearman, precision@1, top-1 efficiency, regret) are computed over groups
of rows sharing a `spacer`. A user choosing a pegRNA has already fixed the allele they
want and the cell/editor they are working in, so the group they face is
(intended allele) x (experimental context). This script counts how many such groups the
frozen corpus actually contains, at what candidate depth, and on which fold.

Nothing is trained, tuned or selected. Fold 0 is described, not fitted.

Usage: .venv/bin/python explore_v2/e01_decision_manifest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import CONTEXT_FIELDS, DESIGN_FIELDS  # noqa: E402


def group_table(d: pd.DataFrame, key: str) -> pd.DataFrame:
    return d.groupby(key, observed=True).agg(
        n=("record_id", "size"),
        n_design=("design_key", "nunique"),
        n_spacer=("spacer", "nunique"),
        n_edit=("edit_key", "nunique"),
        n_cond=("context_key", "nunique"),
    )


def depth_profile(g: pd.DataFrame, col: str = "n_design") -> dict:
    return {f"groups_ge_{k}": int((g[col] >= k).sum()) for k in (2, 3, 4, 5, 8, 12)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    df = C.load_corpus()
    res: dict = {"provenance": C.provenance([C.CORPUS], args.seed),
                 "flank": 12,
                 "design_fields": DESIGN_FIELDS, "context_fields": CONTEXT_FIELDS}

    # ---------------------------------------------------------------- keys
    res["keys"] = {
        "rows": int(len(df)),
        "raw_wt_edited_pairs": int((df.full_unedited + "|" + df.full_edited).nunique()),
        "target_name": int(df.target_name.nunique()),
        "spacer": int(df.spacer.nunique()),
        "canonical_edits": int(df.edit_key.nunique()),
        "design_keys": int(df.design_key.nunique()),
        "context_keys": int(df.context_key.nunique()),
        "decision_groups": int(df.decision_group.nunique()),
        "flank_full_frac": float(df.flank_full.mean()),
        "edit_type_top": df.edit_type.value_counts().head(12).to_dict(),
    }
    # how much of the raw-pair multiplicity was a window convention rather than an edit
    per_target = df.groupby("target_name", observed=True).agg(
        n_pair=("full_unedited", lambda s: s.nunique()),
        n_edit=("edit_key", "nunique"))
    res["keys"]["targets_with_multiple_windows_one_edit"] = int(
        ((per_target.n_pair > 1) & (per_target.n_edit == 1)).sum())
    res["keys"]["edits_reached_by_multiple_spacers"] = int(
        (df.groupby("edit_key", observed=True).spacer.nunique() > 1).sum())

    # ------------------------------------------------- candidate availability
    res["availability"] = {}
    for name, d in [("corpus", df), ("fold0_heldout", df[df.fold == 0]),
                    ("fold1_val", df[df.fold == 1]),
                    ("folds2_5_train", df[df.fold >= 2])]:
        g = group_table(d, "decision_group")
        prof = depth_profile(g)
        rows_ge = {f"rows_in_groups_ge_{k}": int(g.n[g.n_design >= k].sum())
                   for k in (2, 3, 5)}
        # the manuscript's grouping, for contrast, at its own >=5 threshold
        gs = group_table(d, "spacer")
        res["availability"][name] = {
            "rows": int(len(d)), "decision_groups": int(len(g)),
            **prof, **rows_ge,
            "spacer_groups_ge_5": int((gs.n >= 5).sum()),
            "spacer_groups_ge_5_median_n_edit": float(gs.n_edit[gs.n >= 5].median()),
            "spacer_groups_ge_5_median_n_cond": float(gs.n_cond[gs.n >= 5].median()),
            "spacer_groups_ge_5_frac_multi_cond": float((gs.n_cond[gs.n >= 5] > 1).mean()),
            "spacer_groups_ge_5_frac_multi_edit": float((gs.n_edit[gs.n >= 5] > 1).mean()),
        }

    # ------------------------------------------------------------ replicates
    rk = df.groupby("replicate_key", observed=True).agg(
        n=("record_id", "size"), n_file=("source_file", "nunique"),
        sd=("edited", "std"), gmean=("edited", "mean"))
    rep = rk[rk.n > 1]
    res["replicates"] = {
        "groups": int(len(rep)), "rows": int(rep.n.sum()),
        "size_profile": rep.n.value_counts().sort_index().to_dict(),
        "groups_spanning_two_source_files": int((rep.n_file > 1).sum()),
        "median_abs_within_group_sd": float(rep.sd.median()),
        "mean_within_group_sd": float(rep.sd.mean()),
        "mean_within_group_mean": float(rep.gmean.mean()),
        "note": ("A replicate here is metadata-identical: same canonical allele, same "
                 "design molecule, same nine context fields. Two rows can still be the "
                 "same library measured twice or two genuinely independent transfections; "
                 "the corpus does not record which, so these are an upper bound on "
                 "identified replication and a lower bound on measurement noise."),
    }

    # --------------------------------------------------- fold purity / leakage
    df = df.assign(_f0=(df.fold == 0))
    leak = {}
    for key in ["target_name", "spacer", "edit_key", "decision_group"]:
        g = df.groupby(key, observed=True)._f0.agg(["mean", "size"])
        touching = g[g["mean"] > 0]
        mixed = touching[touching["mean"] < 1]
        leak[key] = {"keys_touching_fold0": int(len(touching)),
                     "keys_split_across_folds": int(len(mixed)),
                     "frac_split": float(len(mixed) / max(1, len(touching)))}
    # rows, not keys: how much of fold 0 sits in a decision group that training also saw
    dgm = df.groupby("decision_group", observed=True)._f0.transform("mean")
    f0 = df[df.fold == 0]
    res["fold_structure"] = {
        "by_key": leak,
        "fold0_rows_in_decision_group_shared_with_other_folds":
            int(((dgm < 1) & df._f0).sum()),
        "fold0_rows_total": int(len(f0)),
        "note": ("OptiPrime's official folds are not locus-disjoint under the canonical "
                 "keys: 80.7% of fold-0 protospacers and 20.3% of fold-0 decision groups "
                 "also appear in training folds. The manuscript's leakage check counted "
                 "exact design-and-condition twins (196 rows); alternative designs for "
                 "the same intended allele at the same site are a separate and much "
                 "larger channel."),
    }

    # ------------------------------------ the three tasks of the plan, counted
    # Task 1 (choose a pegRNA) needs >=2 designs for one allele in one context.
    # Task 3 (compare intended edits) is what a spacer group mostly contains.
    d0 = df[df.fold == 0]
    g0 = group_table(d0, "decision_group")
    res["task_separation_fold0"] = {
        "task1_choose_pegRNA_groups": int((g0.n_design >= 2).sum()),
        "task1_rows": int(g0.n[g0.n_design >= 2].sum()),
        "task1_max_candidates": int(g0.n_design.max()),
        "task3_spacer_groups_ge5": int((group_table(d0, "spacer").n >= 5).sum()),
    }

    # ---------------------------------------------------------------- manifest
    C.CACHE.mkdir(parents=True, exist_ok=True)
    cols = ["record_id", "fold", "edit_key", "edit_type", "ref", "alt", "flank_full",
            "design_key", "context_key", "cond3", "decision_group", "replicate_key",
            "target_name", "spacer", "source_file", "edited", "weight"]
    df[cols].to_parquet(C.manifest_path(), index=False)
    res["manifest"] = {"path": str(C.manifest_path().relative_to(C.ROOT)),
                       "canon_version": __import__("canon").CANON_VERSION,
                       "rows": int(len(df)), "columns": cols}

    md = render(res)
    C.write_outputs("e01_decision_manifest", res, md)


def render(r: dict) -> str:
    k, a, f = r["keys"], r["availability"], r["fold_structure"]
    L = []
    L.append("# E01 - canonical decision manifest and candidate availability\n")
    L.append("Descriptive audit of the frozen corpus. Nothing trained, tuned or selected.\n")
    L.append("## Canonicalisation\n")
    L.append("| quantity | count |\n|---|---:|")
    for key, lab in [("rows", "rows"), ("raw_wt_edited_pairs", "distinct raw (WT, edited) string pairs"),
                     ("target_name", "stored target-site names"), ("spacer", "distinct protospacers"),
                     ("canonical_edits", "**canonical intended alleles**"),
                     ("design_keys", "distinct pegRNA design molecules"),
                     ("context_keys", "distinct experimental contexts"),
                     ("decision_groups", "**decision groups (allele x context)**")]:
        L.append(f"| {lab} | {k[key]:,} |")
    L.append(f"| rows with a full 12 bp flank on both sides | {k['flank_full_frac']:.1%} |")
    L.append(f"| target sites whose several WT windows are one allele | {k['targets_with_multiple_windows_one_edit']:,} |")
    L.append(f"| alleles reachable by more than one protospacer | {k['edits_reached_by_multiple_spacers']:,} |")
    L.append("\nThe raw string pair is a design artefact: collapsing the window convention "
             "takes 130,921 distinct WT windows down to "
             f"{k['canonical_edits']:,} intended alleles, and merges designs that reach the "
             "same allele from either strand.\n")
    L.append("## Candidate depth per decision group\n")
    L.append("| surface | rows | decision groups | >=2 designs | >=3 | >=5 | rows in >=2 |\n|---|---:|---:|---:|---:|---:|---:|")
    for name, v in a.items():
        L.append(f"| {name} | {v['rows']:,} | {v['decision_groups']:,} | {v['groups_ge_2']:,} | "
                 f"{v['groups_ge_3']:,} | {v['groups_ge_5']:,} | {v['rows_in_groups_ge_2']:,} |")
    h = a["fold0_heldout"]
    L.append(f"\n**The held-out surface cannot support the manuscript's decision metric at its "
             f"own candidate depth.** Fold 0 contains {h['groups_ge_2']:,} groups with two or more "
             f"alternative designs for one allele in one context, {h['groups_ge_3']:,} with three "
             f"or more, and **{h['groups_ge_5']:,} with five or more**. The manuscript's "
             f"{h['spacer_groups_ge_5']:,} protospacer groups of >=5 rows are a different object: "
             f"{h['spacer_groups_ge_5_frac_multi_cond']:.1%} of them span more than one "
             f"source/cell/editor condition and {h['spacer_groups_ge_5_frac_multi_edit']:.1%} contain "
             f"more than one intended allele (median "
             f"{h['spacer_groups_ge_5_median_n_edit']:.0f} alleles per group).\n")
    L.append("## True replicates\n")
    rep = r["replicates"]
    L.append(f"- metadata-identical replicate groups: **{rep['groups']:,}** covering {rep['rows']:,} rows\n"
             f"- size profile: {rep['size_profile']}\n"
             f"- groups spanning two source files: {rep['groups_spanning_two_source_files']:,}\n"
             f"- median within-group SD of measured efficiency: {rep['median_abs_within_group_sd']:.4f} "
             f"(mean of group means {rep['mean_within_group_mean']:.4f})\n")
    L.append(rep["note"] + "\n")
    L.append("## Fold structure under the canonical keys\n")
    L.append("| key | keys touching fold 0 | also present in other folds | share |\n|---|---:|---:|---:|")
    for key, v in f["by_key"].items():
        L.append(f"| {key} | {v['keys_touching_fold0']:,} | {v['keys_split_across_folds']:,} | {v['frac_split']:.1%} |")
    L.append(f"\n{f['fold0_rows_in_decision_group_shared_with_other_folds']:,} of "
             f"{f['fold0_rows_total']:,} fold-0 rows ("
             f"{f['fold0_rows_in_decision_group_shared_with_other_folds']/f['fold0_rows_total']:.1%}) "
             "sit in a decision group that the training folds also contain.\n")
    L.append(f["note"] + "\n")
    L.append("## Consequence for the plan\n")
    L.append("1. A fixed-allele selection benchmark at >=5 candidates does not exist on fold 0. "
             "It exists on the development folds (fold 1 alone: "
             f"{a['fold1_val']['groups_ge_5']:,} groups) and in the corpus as a whole "
             f"({a['corpus']['groups_ge_5']:,} groups).\n")
    L.append("2. On fold 0 the identifiable decision is the **pairwise** one: two alternative "
             "designs for the same allele in the same context. That is a real estimand and E02 "
             "reports it.\n")
    L.append("3. Locus-disjoint splits must be rebuilt on the canonical edit key, not the "
             "protospacer and not `target_name`, before any transfer claim.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
