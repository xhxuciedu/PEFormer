"""Decision endpoints for fixed-allele pegRNA selection, with the invariants tested.

`NEXT_STEPS_AFTER_E15.md` section 2 identified four defects in E11's inline endpoint code.
All four were reproduced exactly before this module was written:

1. **A candidate was a row, not a design.** 13 panel groups carry more rows than unique
   designs, and 2 groups with only two distinct designs received a top-three result, because
   the budget check tested row count while the reported depth counted unique `design_key`.
   Here a candidate is one distinct design in one decision context, duplicates are
   aggregated before anything is ranked, and no nomination slot can be spent twice on the
   same molecule.
2. **@k was computed on whatever depth existed.** `achieved_efficiency_at_k` is now defined
   only where at least k distinct candidates exist, and every budget curve is reported on
   one fixed eligible population with its own oracle and random baseline.
3. **The random baseline ignored tied maxima.** With full credit for selecting any measured
   maximum, a random pick's hit probability is (number of maximisers)/(candidates), not
   1/n. In an all-zero group every design is a maximiser and the correct value is 1.0; the
   old code returned a mean of 0.429 over the panel's 5,807 all-zero groups.
4. **The bootstrap gave p = 1/n_boot to identical arms.** `frac = (vals > 0).mean()` is 0
   when every resampled difference is exactly 0, so `2*min(frac, 1-frac)` was 0 and the
   floor produced p = 0.0005 for two identical predictors. A degenerate difference
   distribution now yields p = 1.

Passing `verify_report.py` establishes agreement between prose and stored artefacts; it does
not establish that the endpoint algorithms are right. That is what `test_endpoints.py` is
for.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #
def aggregate_candidates(df: pd.DataFrame, group: str = "edit_key",
                         design: str = "design_key", outcome: str = "edited_frac",
                         scores: tuple[str, ...] = ()) -> pd.DataFrame:
    """Collapse to one row per (decision group, distinct design).

    Repeated rows for one design in one context are measurements of the same molecule, not
    two candidates. They are averaged; `n_rows` records how many were combined so a
    sensitivity analysis can exclude groups whose duplicate provenance is ambiguous.
    """
    agg = {outcome: (outcome, "mean"), "n_rows": (outcome, "size")}
    for s in scores:
        agg[s] = (s, "mean")
    out = df.groupby([group, design], observed=True, as_index=False).agg(**agg)
    keep = [c for c in df.columns if c not in (design, outcome, *scores)]
    first = df.groupby(group, observed=True, as_index=False)[keep].first()
    return out.merge(first, on=group, how="left", suffixes=("", "_grp"))


# --------------------------------------------------------------------------- #
# endpoints for one decision group
# --------------------------------------------------------------------------- #
def achieved_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Best measured outcome among the k highest-scored distinct candidates.

    Undefined, not silently truncated, when fewer than k candidates exist.
    """
    if len(y) < k:
        return np.nan
    order = np.argsort(-score, kind="stable")[:k]
    return float(y[order].max())


def hit_at_1(y: np.ndarray, score: np.ndarray) -> float:
    """Whether the top-scored candidate attains the measured maximum; ties get full credit."""
    return float(y[int(np.argmax(score))] == y.max())


def regret_at_1(y: np.ndarray, score: np.ndarray) -> float:
    return float(y.max() - y[int(np.argmax(score))])


def random_achieved_at_k(y: np.ndarray, k: int) -> float:
    """Exact expectation of the best of k candidates drawn without replacement.

    P(max of a random k-subset <= y_(i)) = C(i, k)/C(n, k) over the sorted outcomes, so the
    expectation is a finite sum and needs no simulation.
    """
    n = len(y)
    if n < k:
        return np.nan
    ys = np.sort(y)
    if k == n:
        return float(ys[-1])
    tot, prev, acc = comb(n, k), 0.0, 0.0
    for i in range(k, n + 1):
        cdf = comb(i, k) / tot
        acc += ys[i - 1] * (cdf - prev)
        prev = cdf
    return float(acc)


def random_hit_at_1(y: np.ndarray) -> float:
    """Probability a uniformly random pick attains the maximum, counting tied maximisers."""
    return float((y == y.max()).sum() / len(y))


def random_regret_at_1(y: np.ndarray) -> float:
    return float(y.max() - y.mean())


# --------------------------------------------------------------------------- #
# clustered paired inference
# --------------------------------------------------------------------------- #
def paired_cluster_bootstrap(diff: np.ndarray, cluster: np.ndarray, seed: int,
                             n_boot: int = 2000) -> dict:
    """Percentile interval for a paired mean difference, resampling whole clusters.

    A difference distribution with no variation is the signature of two identical
    predictors, and the honest report is "no evidence against equality", not the smallest
    representable p-value. `degenerate` marks that case explicitly.
    """
    diff = np.asarray(diff, dtype=float)
    ok = np.isfinite(diff)
    diff, cluster = diff[ok], np.asarray(cluster)[ok]
    if diff.size == 0:
        return {"observed": np.nan, "n": 0}
    uniq, inv = np.unique(cluster, return_inverse=True)
    buckets = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([buckets[j] for j in rng.integers(0, len(uniq), len(uniq))])
        vals[b] = diff[idx].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    spread = float(np.ptp(vals))
    if spread == 0.0 and float(np.abs(vals).max()) == 0.0:
        p, degenerate = 1.0, True
    else:
        # two-sided percentile-bootstrap p, floored at the resolution the resamples support
        frac_gt = float((vals > 0).mean())
        frac_lt = float((vals < 0).mean())
        p = min(1.0, 2 * min(frac_gt, frac_lt) if (frac_gt and frac_lt)
                else 2 * (1.0 / n_boot))
        degenerate = False
    return {"observed": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "two_sided_p": float(p), "p_is_resolution_floor": bool(p <= 2.0 / n_boot),
            "degenerate_zero_difference": degenerate,
            "n": int(diff.size), "n_clusters": int(len(uniq)), "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# a whole population
# --------------------------------------------------------------------------- #
def score_population(cand: pd.DataFrame, models: list[str], ks=(1, 3, 5),
                     group: str = "edit_key", outcome: str = "edited_frac",
                     site: str = "protospacer") -> pd.DataFrame:
    """One row per decision group, with every model's endpoints and the random baseline."""
    rows = []
    for gid, s in cand.groupby(group, observed=True, sort=False):
        y = s[outcome].to_numpy()
        rec = {group: gid, "site": s[site].iloc[0], "depth": int(len(y)),
               "oracle": float(y.max()), "mean_y": float(y.mean()),
               "informative": bool(y.max() > y.min()),
               "all_zero": bool(y.max() == 0),
               "n_maximisers": int((y == y.max()).sum())}
        for m in models:
            sc = s[m].to_numpy()
            for k in ks:
                rec[f"{m}_at_{k}"] = achieved_at_k(y, sc, k)
            rec[f"{m}_hit"] = hit_at_1(y, sc)
            rec[f"{m}_regret"] = regret_at_1(y, sc)
        for k in ks:
            rec[f"rand_at_{k}"] = random_achieved_at_k(y, k)
        rec["rand_hit"] = random_hit_at_1(y)
        rec["rand_regret"] = random_regret_at_1(y)
        rows.append(rec)
    return pd.DataFrame(rows)
