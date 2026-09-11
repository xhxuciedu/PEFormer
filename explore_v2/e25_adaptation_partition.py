"""E25 - partition the reserved panel into adaptation train/validation/test components.

Every experiment in ADAPTATION_SELECTION_PLAN.md depends on this split, so it is made once,
here, and never remade. Two requirements drive its construction.

**Leakage.** An allele can be reached by several protospacers and a protospacer can install
several alleles, so alleles and protospacers form a bipartite graph whose connected
components are the smallest units that can be split without putting related sequence on both
sides of the partition. Components, not groups, are assigned.

**Honesty about what this can support.** The panel has already informed E11-E24. A partition
made now prevents new *training* leakage and lets adaptation be evaluated retrospectively; it
does not erase that earlier inspection and is not a pristine confirmatory surface. The
original no-target-label comparison stays as the manuscript's claim; adaptation is reported
as a new and separately caveated result.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e25_adaptation_partition.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import CANON_VERSION  # noqa: E402

SPLITS = ("train", "val", "test")
TARGET = {"train": 0.60, "val": 0.20, "test": 0.20}


def build_candidates(p: pd.DataFrame) -> pd.DataFrame:
    """One row per (decision group, distinct design); repeats are measurements, not choices."""
    c = (p.groupby(["edit_key", "design_key"], observed=True, as_index=False)
           .agg(y=("edited_frac", "mean"), n_meas=("edited_frac", "size"),
                protospacer=("protospacer", "first"), edit_type=("edit_type", "first"),
                pbs_len=("pbs_dna", lambda s: len(s.iloc[0])),
                rtt_len=("rtt_dna", lambda s: len(s.iloc[0]))))
    return c


def components(c: pd.DataFrame) -> pd.Series:
    """Connected components of the allele-protospacer bipartite graph."""
    ek, ei = np.unique(c.edit_key.to_numpy(), return_inverse=True)
    ps, pi = np.unique(c.protospacer.to_numpy(), return_inverse=True)
    n = len(ek) + len(ps)
    rows = np.concatenate([ei, len(ek) + pi])
    cols = np.concatenate([len(ek) + pi, ei])
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return pd.Series(lab[ei], index=c.index, name="component")


def assign(summary: pd.DataFrame, seed: int) -> pd.Series:
    """Greedy stratified assignment of whole components to the 60/20/20 targets.

    Components are bucketed by dominant edit class and by whether they contain a deep group,
    then within each bucket the largest components are placed first into whichever split is
    furthest below its target. Largest-first keeps a few big components from overshooting.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(index=summary.index, dtype=object)
    # Balance two quantities at once. Group count sets the headline 60/20/20; deep-group
    # count is tracked separately because a handful of large components can otherwise
    # absorb the deep groups into one split, and depth is where the decision is hardest.
    keys = ("groups", "deep")
    total = {k: max(float(summary[k].sum()), 1.0) for k in keys}
    have = {s: {k: 0.0 for k in keys} for s in SPLITS}
    summary = summary.assign(_r=rng.random(len(summary)))
    for _, blk in summary.groupby(["stratum"], sort=True):
        loc = {s: {k: 0.0 for k in keys} for s in SPLITS}
        btot = {k: float(blk[k].sum()) for k in keys}
        # A key with no mass in this stratum must be skipped: its deficit would otherwise
        # collapse to TARGET[s], a constant bias toward the largest split.
        active = [k for k in keys if btot[k] > 0]
        for cid, r in blk.sort_values(["groups", "_r"], ascending=[False, True]).iterrows():
            def score(s):
                local = sum(TARGET[s] - loc[s][k] / btot[k] for k in active)
                glob = sum(TARGET[s] - have[s][k] / total[k] for k in active)
                return (local, glob)
            s = max(SPLITS, key=score)
            out[cid] = s
            for k in keys:
                loc[s][k] += r[k]
                have[s][k] += r[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    p = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    c = build_candidates(p)
    c["component"] = components(c)

    grp = (c.groupby(["component", "edit_key"], observed=True)
             .agg(depth=("design_key", "nunique"),
                  informative=("y", lambda s: s.max() > s.min()),
                  edit_type=("edit_type", "first")).reset_index())
    summary = (grp.groupby("component")
                  .agg(groups=("edit_key", "nunique"), max_depth=("depth", "max"),
                       eligible=("depth", lambda s: int((s >= 2).sum())),
                       informative=("informative", "sum"),
                       deep=("depth", lambda s: int((s >= 5).sum())),
                       edit_type=("edit_type", lambda s: s.mode().iat[0])))
    # RTT length is the axis on which the models are known to be biased (E23), so an
    # accidental train/test imbalance in it would confound every adaptation comparison.
    # Stratify on it explicitly alongside edit class and depth.
    rtt = c.groupby("component").rtt_len.mean()
    summary["rtt_band"] = pd.qcut(rtt.reindex(summary.index), 4,
                                  labels=["q1", "q2", "q3", "q4"]).astype(str)
    summary["stratum"] = (summary.edit_type.str[:3] + "|"
                          + np.where(summary.max_depth >= 5, "deep",
                                     np.where(summary.max_depth >= 2, "mid", "single"))
                          + "|" + summary.rtt_band)
    summary["split"] = assign(summary, args.seed)
    c = c.merge(summary[["split"]], left_on="component", right_index=True)

    out = C.CACHE / f"adaptation_partition_v{CANON_VERSION}.parquet"
    c.to_parquet(out, index=False)

    sizes = summary.groups.to_numpy()
    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "disclosure": ("The panel already informed E11-E24. This partition prevents new "
                          "training leakage and supports a retrospective adaptation "
                          "evaluation; it is not a pristine confirmatory surface and the "
                          "no-target-label comparison remains the manuscript's claim."),
           "graph": {"alleles": int(c.edit_key.nunique()),
                     "protospacers": int(c.protospacer.nunique()),
                     "candidates": int(len(c)),
                     "components": int(summary.shape[0]),
                     "component_size_in_groups": {
                         "max": int(sizes.max()), "mean": float(sizes.mean()),
                         "median": float(np.median(sizes)),
                         "frac_singleton": float((sizes == 1).mean()),
                         "largest_5": [int(x) for x in np.sort(sizes)[-5:][::-1]]},
                     "alleles_reached_by_multiple_protospacers": int(
                         (c.groupby("edit_key").protospacer.nunique() > 1).sum()),
                     "protospacers_installing_multiple_alleles": int(
                         (c.groupby("protospacer").edit_key.nunique() > 1).sum())},
           "splits": {}}
    for s in SPLITS:
        sub = c[c.split == s]
        g = grp.merge(summary[["split"]], left_on="component", right_index=True)
        g = g[g.split == s]
        res["splits"][s] = {
            "components": int(summary[summary.split == s].shape[0]),
            "groups": int(g.edit_key.nunique()),
            "candidates": int(len(sub)),
            "measurements": int(sub.n_meas.sum()),
            "eligible_groups_depth_ge2": int((g.depth >= 2).sum()),
            "informative_groups": int(g.informative.sum()),
            "groups_depth_ge5": int((g.depth >= 5).sum()),
            "mean_efficiency": float(sub.y.mean()),
            "zero_fraction": float((sub.y == 0).mean()),
            "share_of_groups": float(g.edit_key.nunique() / grp.edit_key.nunique())}
    # a split that accidentally differs in geometry would confound every later comparison
    res["geometry_by_split"] = {
        s: {"pbs_len_mean": float(c[c.split == s].pbs_len.mean()),
            "rtt_len_mean": float(c[c.split == s].rtt_len.mean())} for s in SPLITS}
    res["edit_class_by_split"] = {
        s: (c[c.split == s].edit_type.value_counts(normalize=True).round(4).to_dict())
        for s in SPLITS}
    res["artifact"] = str(out.relative_to(C.CACHE.parent.parent))
    C.write_outputs("e25_adaptation_partition", res, render(res))


def render(r: dict) -> str:
    g, sp = r["graph"], r["splits"]
    L = ["# E25 - the adaptation partition\n",
         "> **Disclosure.** " + r["disclosure"] + "\n",
         "## Why components rather than groups\n",
         f"{g['alleles_reached_by_multiple_protospacers']:,} alleles are reachable by more "
         f"than one protospacer and {g['protospacers_installing_multiple_alleles']:,} "
         "protospacers install more than one allele, so alleles and protospacers are not "
         "independent units. Their bipartite graph has "
         f"**{g['components']:,} connected components** over {g['alleles']:,} alleles and "
         f"{g['protospacers']:,} protospacers; whole components are assigned to a split.\n",
         f"Component size is heavily skewed: median {g['component_size_in_groups']['median']:.0f} "
         f"group(s), mean {g['component_size_in_groups']['mean']:.2f}, "
         f"{g['component_size_in_groups']['frac_singleton']:.1%} singletons, largest "
         f"{g['component_size_in_groups']['largest_5'][0]:,} groups.\n",
         "## Realised split\n",
         "| | components | groups | eligible (d$\\geq$2) | informative | d$\\geq$5 | candidates | measurements | share |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SPLITS:
        v = sp[s]
        L.append(f"| {s} | {v['components']:,} | {v['groups']:,} | "
                 f"{v['eligible_groups_depth_ge2']:,} | {v['informative_groups']:,} | "
                 f"{v['groups_depth_ge5']:,} | {v['candidates']:,} | "
                 f"{v['measurements']:,} | {v['share_of_groups']:.1%} |")
    L.append("\n## The splits are comparable\n")
    L.append("| split | mean efficiency | zero fraction | mean PBS | mean RTT |\n|---|---:|---:|---:|---:|")
    for s in SPLITS:
        L.append(f"| {s} | {sp[s]['mean_efficiency']:.4f} | {sp[s]['zero_fraction']:.3f} | "
                 f"{r['geometry_by_split'][s]['pbs_len_mean']:.2f} | "
                 f"{r['geometry_by_split'][s]['rtt_len_mean']:.2f} |")
    L.append("\nThe assignment balances two quantities at once: total groups, which sets the "
             "headline allocation, and groups of depth five or more, which are where the "
             "decision is hardest and where a few large components could otherwise absorb "
             "all the evaluation power. Balancing both trades a little accuracy on the "
             "60/20/20 group target for a usable spread of deep groups; realised shares are "
             "in the table above rather than assumed.\n")
    L.append(f"Written to `{r['artifact']}`. Model selection uses **validation only**; the "
             "test components are not consulted while choosing architectures, losses, "
             "fine-tuning depth or stopping epochs.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
