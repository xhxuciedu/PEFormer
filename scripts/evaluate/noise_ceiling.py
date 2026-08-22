"""Replicate-level noise ceiling for the Kim (DeepPrime) partition.

Round 6 concluded that ~40 modelling experiments all return what a mechanism-free
re-run returns, and inferred the limit is the data rather than the model. That
inference is testable: if the same pegRNA, in the same experimental condition, is
measured twice and the two measurements disagree, no model can predict either one
better than they predict each other.

**Finding replicates required care.** Grouping on (spacer, rtt, pbs, cell_type,
pe_type, cas9_type) yields 21,994 apparently-replicated keys -- but inspection shows
they differ in `motif`, `epegRNA` and `linker`, i.e. they are epegRNA versus plain
pegRNA constructs, not repeats. Treating them as replicates would badly understate
the ceiling. The key here therefore spans all 16 design and condition covariates, and
the primary analysis further requires an identical `target_name`, which leaves 649
clean groups.

**The statistic.** For two independent noisy measurements y1, y2 of a true value t:

    reliability  R = corr(y1, y2)
    ceiling        = corr(t, y_obs) = sqrt(R)

So sqrt(R) bounds the correlation any model -- however perfect -- can achieve against
a single noisy observation, which is exactly what the benchmark scores against.

Reported with a bootstrap CI over replicate groups, since 649 pairs is not a large
sample and the point estimate alone would overstate the precision.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("noise_ceiling")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
FULL_KEY = ["spacer", "rtt", "pbs", "cell_type", "pe_type", "cas9_type", "cas9_pam",
            "motif", "scaffold_name", "linker", "rt_name", "time",
            "PEmax", "epegRNA", "MLH1dn", "NRCH"]
N_BOOT = 5000
SEED = 20260812


def pairs_for(df: pd.DataFrame, require_same_target: bool) -> tuple[np.ndarray, np.ndarray]:
    """First two measurements of each replicate group, as paired vectors."""
    key = FULL_KEY + (["target_name"] if require_same_target else [])
    g = df.groupby(key, dropna=False)
    a, b = [], []
    for _, sub in g:
        if len(sub) >= 2:
            v = sub.edited.to_numpy()
            a.append(v[0])
            b.append(v[1])
    return np.asarray(a), np.asarray(b)


def report(name: str, a: np.ndarray, b: np.ndarray, out: dict) -> None:
    if len(a) < 30:
        logger.info("%-28s only %d pairs -- too few to report", name, len(a))
        return
    rs = spearmanr(a, b).statistic
    rp = pearsonr(a, b)[0]

    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    idx = np.arange(len(a))
    for i in range(N_BOOT):
        s = rng.choice(idx, size=len(idx), replace=True)
        # A resample can be degenerate (all-zero) for a heavily zero-inflated variable.
        r = spearmanr(a[s], b[s]).statistic
        boot[i] = r if np.isfinite(r) else np.nan
    boot = boot[np.isfinite(boot)]
    lo, hi = np.quantile(boot, [0.025, 0.975])

    ceil = np.sqrt(max(rs, 0.0))
    ceil_lo, ceil_hi = np.sqrt(max(lo, 0.0)), np.sqrt(max(hi, 0.0))
    frac_zero = float(np.mean((a == 0) & (b == 0)))
    both_zero_or_one = float(np.mean((a == 0) | (b == 0)))

    logger.info("%-28s n=%4d  replicate Spearman=%.4f [%.4f, %.4f]  Pearson=%.4f",
                name, len(a), rs, lo, hi, rp)
    logger.info("%-28s implied ceiling sqrt(R) = %.4f [%.4f, %.4f]   (both zero: %.1f%%, "
                "either zero: %.1f%%)", "", ceil, ceil_lo, ceil_hi,
                100 * frac_zero, 100 * both_zero_or_one)
    out[name] = {"n_pairs": int(len(a)), "replicate_spearman": float(rs),
                 "ci95": [float(lo), float(hi)], "replicate_pearson": float(rp),
                 "implied_ceiling": float(ceil),
                 "ceiling_ci95": [float(ceil_lo), float(ceil_hi)],
                 "both_zero_frac": frac_zero}


def stratified_ceiling() -> dict:
    """Reweight the replicate evidence to Kim's ACTUAL efficiency distribution.

    The raw replicate estimate is biased upward: rows that get measured twice are
    enriched for high efficiency (24.2% exact zeros versus 49.5% across Kim, mean
    0.200 versus 0.061), and those are precisely the rows easiest to reproduce.
    Quoting it as "the Kim ceiling" would overstate the achievable score.

    This corrects for that with a variance-components estimate:

      1. Measurement noise is strongly heteroscedastic in this assay, so estimate it
         *as a function of efficiency level*: within bins of the pair mean, the noise
         variance is Var(y1 - y2) / 2.
      2. Assign every Kim row the noise variance of its level bin.
      3. Simulate two replicate measurements for all 69,635 Kim rows using those
         level-specific noise variances, and take the Spearman between them.

    Step 3 gives a reliability for the real Kim distribution rather than for the
    replicated subset. Simulated values are clipped to [0, 1] and the point mass at
    zero is preserved (a truly-zero row stays zero), because efficiency is bounded and
    the zero block is a real feature of the assay rather than a noise artefact.
    """
    d = pd.read_parquet(CORPUS)
    k = d[d.source_study == "deepprime"].copy()
    key = FULL_KEY + ["target_name"]
    k["_key"] = k.groupby(key, dropna=False).ngroup()
    n = k.groupby("_key").size()
    rep = k[k._key.map(n) > 1]

    a, b = [], []
    for _, sub in rep.groupby("_key"):
        v = sub.edited.to_numpy()
        a.append(v[0]); b.append(v[1])
    a, b = np.asarray(a), np.asarray(b)
    m = 0.5 * (a + b)

    edges = np.array([0.0, 1e-9, 0.005, 0.02, 0.05, 0.15, 0.35, 1.01])
    lab = np.digitize(m, edges) - 1
    noise_var = {}
    logger.info("")
    logger.info("--- noise variance by efficiency level (from %d replicate pairs) ---", len(a))
    for i in range(len(edges) - 1):
        sel = lab == i
        if sel.sum() >= 15:
            nv = float(np.var(a[sel] - b[sel]) / 2.0)
            noise_var[i] = nv
            logger.info("  level [%.3g, %.3g): n=%4d  noise sd=%.4f",
                        edges[i], edges[i + 1], sel.sum(), np.sqrt(nv))
    if not noise_var:
        return {}
    default = float(np.median(list(noise_var.values())))

    y = k.edited.to_numpy()
    ylab = np.digitize(y, edges) - 1
    sd = np.sqrt(np.array([noise_var.get(int(l), default) for l in ylab]))
    # A truly-zero measurement has no downward room and the assay reports exact zeros;
    # keeping them zero avoids inventing variation the data does not show.
    sd[y == 0] = 0.0

    rng = np.random.default_rng(SEED)
    rs = []
    for _ in range(25):
        y1 = np.clip(y + rng.normal(0, sd), 0, 1)
        y2 = np.clip(y + rng.normal(0, sd), 0, 1)
        rs.append(spearmanr(y1, y2).statistic)
    R = float(np.mean(rs))
    logger.info("")
    logger.info("REWEIGHTED to the full Kim distribution (n=%d):", len(y))
    logger.info("  simulated replicate Spearman R = %.4f  (sd over 25 sims %.4f)", R, float(np.std(rs)))
    logger.info("  implied ceiling sqrt(R)        = %.4f", np.sqrt(max(R, 0)))
    return {"reweighted_reliability": R, "reweighted_ceiling": float(np.sqrt(max(R, 0))),
            "noise_sd_by_level": {str(i): float(np.sqrt(v)) for i, v in noise_var.items()}}


def main() -> None:
    d = pd.read_parquet(CORPUS)
    out: dict = {}

    logger.info("=== replicate-level reliability ===")
    for study, label in (("deepprime", "Kim"), ("hsu2026", "Liu")):
        sub = d[d.source_study == study]
        a, b = pairs_for(sub, require_same_target=True)
        report(f"{label} (same target site)", a, b, out)
        a2, b2 = pairs_for(sub, require_same_target=False)
        report(f"{label} (any target)", a2, b2, out)

    # Held-out contamination check: replicate groups that straddle the official split
    # mean a held-out row's exact design+condition twin sits in training.
    k = d[d.source_study == "deepprime"].copy()
    k["_key"] = k.groupby(FULL_KEY + ["target_name"], dropna=False).ngroup()
    n = k.groupby("_key").size()
    rep = k[k._key.map(n) > 1]
    straddle = rep.groupby("_key").fold.apply(lambda f: (f == 0).any() and (f != 0).any())
    logger.info("")
    logger.info("replicate groups straddling the official train/held-out split: %d of %d",
                int(straddle.sum()), len(straddle))
    out["straddling_groups"] = int(straddle.sum())
    out["total_replicate_groups"] = int(len(straddle))

    out.update(stratified_ceiling())

    Path("results/round6").mkdir(parents=True, exist_ok=True)
    Path("results/round6/noise_ceiling.json").write_text(json.dumps(out, indent=2))
    logger.info("wrote results/round6/noise_ceiling.json")


if __name__ == "__main__":
    main()
