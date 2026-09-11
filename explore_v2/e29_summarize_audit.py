"""Summarize cached, head-specific predictions without running or tuning models."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _v2common as C
import endpoints as E
from e29_head_audit import CACHE
from e27_adaptation_test import reference_scores


def paired_interval(diff, cluster, seed=20260910, n_boot=2000):
    """Same whole-cluster bootstrap as E, accelerated via cluster sums/counts."""
    diff, cluster = np.asarray(diff, dtype=float), np.asarray(cluster)
    ok = np.isfinite(diff)
    diff, cluster = diff[ok], cluster[ok]
    if not len(diff):
        raise ValueError("Empty contrast")
    unique, code = np.unique(cluster, return_inverse=True)
    sums = np.bincount(code, weights=diff)
    counts = np.bincount(code)
    rng = np.random.default_rng(seed)
    values = []
    for offset in range(0, n_boot, 100):
        draw = rng.integers(0, len(unique), size=(min(100, n_boot - offset), len(unique)))
        values.extend(sums[draw].sum(axis=1) / counts[draw].sum(axis=1))
    values = np.asarray(values)
    # Mirror the existing endpoint treatment of zero differences and finite
    # bootstrap resolution; these are conditional, unadjusted audit intervals.
    degenerate = bool(np.ptp(values) == 0 and np.abs(values).max() == 0)
    above, below = float((values > 0).mean()), float((values < 0).mean())
    p = 1.0 if degenerate else min(1.0, 2 * min(above, below)
                                  if above and below else 2 / n_boot)
    return {"observed": float(diff.mean()), "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
            "two_sided_p": float(p), "groups": len(diff), "clusters": len(unique),
            "n_boot": n_boot, "degenerate": degenerate}


def metrics(frame, source=False):
    if frame[["prediction", "selection", "y"]].isna().any().any():
        raise ValueError("Missing prediction or outcome")
    cand = frame.groupby(["edit_key", "design_key"], sort=True, as_index=False).agg(
        y=("y", "mean"), prediction=("prediction", "mean"),
        selection=("selection", "mean"), component=("component", "first"))
    grouped = cand.groupby("edit_key", sort=False, observed=True)
    table = grouped.agg(site=("component", "first"), depth=("design_key", "size"),
                        oracle=("y", "max"), minimum=("y", "min")).reset_index()
    table["informative"] = table.oracle > table.minimum
    for head in ("prediction", "selection"):
        # idxmax keeps the first maximum, matching the stable candidate-order
        # tie rule in E.score_population. Avoid its unused @k/random work here.
        winners = cand.loc[grouped[head].idxmax(), ["edit_key", "y"]].rename(
            columns={"y": f"{head}_at_1"})
        table = table.merge(winners, on="edit_key", validate="1:1")
        table[f"{head}_regret"] = table.oracle - table[f"{head}_at_1"]
    if source:
        table = table.loc[(table.depth >= 2) & table.informative].copy()
    result = {"rows": len(frame), "groups": len(table),
              "eligibility": "depth>=2 and informative" if source else "all E25 eligible groups"}
    for head in ("prediction", "selection"):
        result[head] = {"achieved_at_1": float(table[f"{head}_at_1"].mean()),
                        "spearman": C.spearman(frame[head].to_numpy(), frame.y.to_numpy()),
                        "regret_at_1": float(table[f"{head}_regret"].mean())}
    return result, table


def contrast(tables, left, right, head="selection"):
    a, b = tables[left], tables[right]
    joined = a[["edit_key", "site", f"{head}_at_1"]].merge(
        b[["edit_key", f"{head}_at_1"]], on="edit_key", validate="1:1", suffixes=("_a", "_b"))
    if len(joined) != len(a) or len(joined) != len(b):
        raise ValueError("Different evaluation populations")
    return paired_interval(joined[f"{head}_at_1_a"] - joined[f"{head}_at_1_b"], joined.site)


def main():
    results, tables, hashes = {}, {}, {}
    reference = reference_scores()
    for path in sorted(CACHE.glob("*.parquet")):
        name, surface = path.stem.split("__", 1)
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            continue  # inference still completing the cache, do not read it yet
        hashes[path.name] = json.loads(meta_path.read_text())["checkpoint_sha256"]
        result, table = metrics(pd.read_parquet(path), source=surface == "source_fold0")
        results.setdefault(surface, {})[name] = result
        tables.setdefault(surface, {})[name] = table
        if surface.startswith("target_") and "released_OptiPrime" not in tables[surface]:
            frame = pd.read_parquet(path)[["edit_key", "design_key", "component", "y"]]
            frame = frame.merge(reference, on=["edit_key", "design_key"], validate="1:1", how="left")
            for col, label in (("op", "released_OptiPrime"), ("ours", "published_ensemble")):
                m, t = metrics(frame.assign(prediction=frame[col], selection=frame[col]))
                results[surface][label], tables[surface][label] = m, t
    contrasts = {}
    for surface, tt in tables.items():
        if not surface.startswith("target_"):
            continue
        cc = contrasts.setdefault(surface, {})
        for name in tt:
            if name != "start_checkpoint" and "start_checkpoint" in tt:
                cc[f"{name}_minus_start"] = contrast(tt, name, "start_checkpoint")
            if name not in ("released_OptiPrime", "published_ensemble") and "released_OptiPrime" in tt:
                cc[f"{name}_minus_released_OptiPrime"] = contrast(tt, name, "released_OptiPrime")
        for seed in (20260910, 20260911, 20260912):
            p, s, pair = f"P_s{seed}", f"S_s{seed}", f"S_pairwise_s{seed}"
            for left, right in ((s, p), (pair, p), (pair, s),
                                (pair, f"shared_s{seed}"),
                                (f"M_s{seed}", f"shared_s{seed}")):
                if left in tt and right in tt:
                    cc[f"{left}_minus_{right}"] = contrast(tt, left, right)
        # Paired average-seed contrast: resample loci after averaging each
        # locus's outcomes across matching seeds. This is NOT score ensembling.
        for arm, reference_arm in (("S", "P"), ("S_pairwise", "P"),
                                   ("S_pairwise", "S"), ("S_pairwise", "shared")):
            seeds = [s for s in (20260910, 20260911, 20260912)
                     if f"{arm}_s{s}" in tt and f"{reference_arm}_s{s}" in tt]
            if len(seeds) == 3:
                joined = None
                for seed in seeds:
                    a = tt[f"{arm}_s{seed}"].set_index("edit_key")
                    b = tt[f"{reference_arm}_s{seed}"].set_index("edit_key")
                    part = (a.selection_at_1 - b.selection_at_1).rename(str(seed))
                    joined = part.to_frame() if joined is None else joined.join(part, validate="1:1")
                if joined.isna().any().any():
                    raise ValueError("Missing group in average-seed comparison")
                cc[f"{arm}_minus_{reference_arm}_mean_of_three_seeds"] = paired_interval(
                    joined.mean(axis=1), a.loc[joined.index, "site"])
    source_means = {}
    for name, values in results.get("source_fold0", {}).items():
        if "_s" not in name:
            continue
        arm = name.rsplit("_s", 1)[0]
        source_means.setdefault(arm, []).append(values)
    source_means = {arm: {"n_seeds": len(values), **{
        head: {metric: float(np.mean([v[head][metric] for v in values]))
               for metric in ("achieved_at_1", "spearman", "regret_at_1")}
        for head in ("prediction", "selection")}} for arm, values in source_means.items()}
    result = {"note": "Retrospective audit. Source fold0 and target test are not tuning surfaces. "
                      "Intervals condition on fitted models; no multiple-testing adjustment.",
              "checkpoint_hashes": hashes, "surfaces": results, "paired_contrasts": contrasts,
              "source_means_by_arm": source_means}
    lines = ["# E29: matched-start and both-head audit", "", result["note"], ""]
    for surface, models in results.items():
        lines += [f"## {surface}", "",
                  "| Model | Prediction @1 | Selection @1 | Prediction rho | Selection rho | Groups |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, m in models.items():
            p, s = m["prediction"], m["selection"]
            lines.append(f"| {name} | {p['achieved_at_1']:.5f} | {s['achieved_at_1']:.5f} | "
                         f"{p['spearman']:.4f} | {s['spearman']:.4f} | {m['groups']} |")
        lines.append("")
        if surface in contrasts:
            lines += ["| Paired selection contrast | Difference | 95% locus interval | p |",
                      "|---|---:|---|---:|"]
            for name, v in contrasts[surface].items():
                lo, hi = v["ci95"]
                lines.append(f"| {name} | {v['observed']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {v['two_sided_p']:.3f} |")
            lines.append("")
    lines += ["## Source means across available seeds", "",
              "Descriptive means, not ensembles; seed counts differ for some arms.", "",
              "| Arm | Seeds | Prediction @1 | Selection @1 | Prediction rho | Selection rho |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, values in source_means.items():
        p, s = values["prediction"], values["selection"]
        lines.append(f"| {arm} | {values['n_seeds']} | {p['achieved_at_1']:.5f} | "
                     f"{s['achieved_at_1']:.5f} | {p['spearman']:.4f} | {s['spearman']:.4f} |")
    C.write_outputs("e29_head_audit", result, "\n".join(lines))


if __name__ == "__main__":
    main()
