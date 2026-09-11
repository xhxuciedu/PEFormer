"""Nested component budgets, and audit of the historical group-level sampler.

All subset construction uses identifiers only, never measured outcomes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _v2common as C
from e30_replicate_pairwise import sha256


def nested_components(frame, budgets, seed):
    """Whole-component random prefixes, reaching (possibly exceeding) each budget.

    `None` denotes all training groups. Exact caps and complete components cannot
    generally both be satisfied. Overshoot is explicit, not silently truncated.
    """
    train = frame.loc[frame.split == "train"]
    if train.empty or train.groupby("edit_key").component.nunique().max() != 1:
        raise ValueError("Need nonempty training data and one component per group")
    sizes = train.groupby("component", sort=True).edit_key.nunique()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    cumulative = np.cumsum(sizes.loc[order].to_numpy())
    result = {}
    for budget in budgets:
        if budget is not None and budget < 1:
            raise ValueError("Budgets must be positive")
        n = len(order) if budget is None else min(len(order), int(np.searchsorted(cumulative, budget)) + 1)
        result[budget] = order[:n].copy()
    return result


def main():
    path = C.CACHE / "adaptation_partition_v2.parquet"
    frame = pd.read_parquet(path)
    for key in ("component", "edit_key", "protospacer"):
        if key in frame and frame.groupby(key).split.nunique().max() != 1:
            raise ValueError(f"Cross-split overlap in {key}")
    val = frame.loc[frame.split == "val"]
    train = frame.loc[frame.split == "train"]
    rows, summary = [], []
    for seed in (20260910, 20260911, 20260912):
        subsets = nested_components(frame, [200, 1000, 5000, None], seed)
        # Component roles stay fixed across nested total-label budgets. This is
        # an inner validation set drawn exclusively from E25 training components.
        all_components = np.sort(train.component.unique())
        draws = np.random.default_rng(seed + 1000).random(len(all_components))
        inner_val = set(all_components[draws < 0.2])
        for budget, chosen in subsets.items():
            sub = train.loc[train.component.isin(chosen)]
            for component in chosen:
                rows.append({"seed": seed, "nominal_groups": budget or -1,
                             "component": component,
                             "total_budget_role": "inner_val" if component in inner_val else "train"})
            iv = sub.loc[sub.component.isin(inner_val)]
            tr = sub.loc[~sub.component.isin(inner_val)]
            if iv.empty or tr.empty:
                raise ValueError("Budget cannot support both inner train and validation")
            summary.append({"seed": seed, "nominal_groups": budget,
                            "actual_selected_groups": int(sub.edit_key.nunique()),
                            "selected_candidates": len(sub), "components": len(chosen),
                            "training_budget_protocol": {
                                "training_groups": int(sub.edit_key.nunique()),
                                "validation_groups": int(val.edit_key.nunique()),
                                "total_labelled_groups": int(sub.edit_key.nunique() + val.edit_key.nunique()),
                                "total_candidate_measurements": len(sub) + len(val)},
                            "total_budget_protocol": {
                                "training_groups": int(tr.edit_key.nunique()),
                                "inner_validation_groups": int(iv.edit_key.nunique()),
                                "total_labelled_groups": int(sub.edit_key.nunique()),
                                "total_candidate_measurements": len(sub),
                                "fixed_E25_validation_used_for_selection": False}})
    # Reproduce the old sampling order exactly (AD sorts split/edit/design).
    train = train.sort_values(["edit_key", "design_key"])
    groups = train.edit_key.unique()
    historic, previous = [], None
    for budget in (200, 1000, 5000):
        keep = set(np.random.default_rng(20260910).choice(groups, min(budget, len(groups)), replace=False))
        sub = train.loc[train.edit_key.isin(keep)]
        full_sizes = train.groupby("component").edit_key.nunique()
        selected_sizes = sub.groupby("component").edit_key.nunique()
        historic.append({"training_groups": budget, "training_candidates": len(sub),
                         "partial_components": int((selected_sizes < full_sizes.loc[selected_sizes.index]).sum()),
                         "previous_subset_nested": None if previous is None else previous <= keep,
                         "previous_groups_retained": None if previous is None else len(previous & keep),
                         "labelled_validation_groups": int(val.edit_key.nunique())})
        previous = keep
    dest = C.CACHE / "adapt_followup_budget_components.parquet"
    pd.DataFrame(rows).to_parquet(dest, index=False)
    result = {"partition_sha256": sha256(path), "manifest_sha256": sha256(dest),
              "construction": "Outcome-free random component prefixes; explicit overshoot",
              "historical_sampler": historic, "budgets": summary}
    lines = ["# E31: corrected label-budget protocol", "",
             "Protocol construction only: no new learning-curve training has been run.", "",
             "Whole-component prefixes are nested; actual budgets may exceed nominal values.",
             "Total-budget inner validation uses only selected E25 training components.", "",
             "| Seed | Nominal groups | Actual total groups | Candidates | Inner train | Inner val |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        inner = row["total_budget_protocol"]
        lines.append(f"| {row['seed']} | {row['nominal_groups'] or 'all'} | {row['actual_selected_groups']} | "
                     f"{row['selected_candidates']} | {inner['training_groups']} | {inner['inner_validation_groups']} |")
    lines += ["", "Historical small-budget runs also used 5,561 labelled validation groups.",
              "Their sampler retained whole decision groups, not necessarily whole components.",
              "See JSON for overlap and partial-component counts. No new test results are used."]
    C.write_outputs("e31_label_budgets", result, "\n".join(lines))


if __name__ == "__main__":
    main()
