"""Empirical (assumption-light) noise ceiling for Kim, and the gap to it by condition.

Supersedes the Gaussian variance-components estimate in `noise_ceiling.py`, which
assumed an observed zero is noiseless (sd = 0). The replicate data refutes that:
**21.4% of rows observed as exactly 0 have a non-zero replicate** (median 0.0015 when
non-zero). A zero is a *censored* observation, not a certain one, and assuming
otherwise inflates the ceiling precisely where zeros dominate -- which is exactly the
Kim conditions the model does worst on. That would have manufactured headroom.

This estimator makes no distributional assumption. For a row observed at efficiency y,
it draws a synthetic replicate from the **empirical** distribution of replicate values
observed for rows measured near y, via nearest-neighbour matching on the 1,298 ordered
replicate pairs. Zero-discordance, heteroscedasticity and boundedness are all inherited
from the data rather than modelled.

Two independent draws per row give the replicate-replicate Spearman R for the real
Kim distribution; sqrt(R) bounds what any model can score against a single observation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ceiling_emp")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
PRED = Path("results/round3/dev_recalibration/predictions_r4p2_ordSSM_oof_round3_dev_fold_0.parquet")
FULL_KEY = ["spacer", "rtt", "pbs", "cell_type", "pe_type", "cas9_type", "cas9_pam",
            "motif", "scaffold_name", "linker", "rt_name", "time",
            "PEmax", "epegRNA", "MLH1dn", "NRCH", "target_name"]
K_NEIGHBOURS = 40
N_SIM = 25
SEED = 20260812


def replicate_pairs() -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_parquet(CORPUS)
    k = d[d.source_study == "deepprime"].copy()
    k["_key"] = k.groupby(FULL_KEY, dropna=False).ngroup()
    n = k.groupby("_key").size()
    rep = k[k._key.map(n) > 1]
    a, b = [], []
    for _, s in rep.groupby("_key"):
        v = s.edited.to_numpy()
        a.append(v[0]); b.append(v[1])
    a, b = np.asarray(a), np.asarray(b)
    # Symmetrise: a replicate pair is exchangeable, so use each ordering.
    return np.concatenate([a, b]), np.concatenate([b, a])


class EmpiricalReplicator:
    """Draws y_replicate | y_observed from nearest-neighbour replicate pairs."""

    def __init__(self, y1: np.ndarray, y2: np.ndarray, k: int = K_NEIGHBOURS):
        order = np.argsort(y1)
        self.y1 = y1[order]
        self.y2 = y2[order]
        self.k = k

    def draw(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pos = np.searchsorted(self.y1, y)
        lo = np.clip(pos - self.k // 2, 0, len(self.y1) - self.k)
        offs = rng.integers(0, self.k, size=len(y))
        return self.y2[lo + offs]


def main() -> None:
    y1, y2 = replicate_pairs()
    rep = EmpiricalReplicator(y1, y2)
    rng = np.random.default_rng(SEED)
    logger.info("empirical replicator built from %d ordered pairs", len(y1))

    pred = pd.read_parquet(PRED)
    kim = pred[pred.source_study == "deepprime"]
    y = kim.true_efficiency.to_numpy()

    def ceiling(vals: np.ndarray) -> float:
        rs = [spearmanr(rep.draw(vals, rng), rep.draw(vals, rng)).statistic for _ in range(N_SIM)]
        return float(np.sqrt(max(np.mean(rs), 0.0)))

    overall = ceiling(y)
    ours = spearmanr(y, kim.predicted_efficiency).statistic
    logger.info("")
    logger.info("KIM overall (dev fold 0, n=%d)", len(y))
    logger.info("  Ordinal-SSM        : %.4f", ours)
    logger.info("  empirical ceiling  : %.4f", overall)
    logger.info("  gap                : %+.4f", overall - ours)

    out = {"kim_overall": {"model": float(ours), "ceiling": overall, "gap": overall - ours}}

    logger.info("")
    logger.info("%-24s %6s %7s %9s %9s %8s", "condition", "n", "%zero", "model", "ceiling", "gap")
    rows = []
    for c, s in kim.groupby(["cell_type", "pe_type"]):
        if len(s) < 100:
            continue
        yy = s.true_efficiency.to_numpy()
        r = spearmanr(yy, s.predicted_efficiency).statistic
        cl = ceiling(yy)
        rows.append((cl - r, str(c), len(s), 100 * (yy == 0).mean(), r, cl))
    for gap, c, n, z, r, cl in sorted(rows, reverse=True):
        logger.info("%-24s %6d %6.1f%% %9.4f %9.4f %+8.4f", c, n, z, r, cl, gap)
        out.setdefault("by_condition", {})[c] = {
            "n": n, "pct_zero": z, "model": float(r), "ceiling": cl, "gap": float(gap)}

    # Is the remaining gap concentrated in the zero block or among editing rows?
    logger.info("")
    logger.info("--- where is the gap? split Kim by whether the row edits at all ---")
    for lab, sel in (("zero rows", y == 0), ("editing rows (y>0)", y > 0)):
        yy = y[sel]
        pp = kim.predicted_efficiency.to_numpy()[sel]
        if len(yy) < 100 or np.unique(yy).size < 3:
            logger.info("  %-20s n=%5d  (degenerate: no ordering to score)", lab, len(yy))
            out.setdefault("split", {})[lab] = {"n": int(len(yy)), "degenerate": True}
            continue
        r = spearmanr(yy, pp).statistic
        cl = ceiling(yy)
        logger.info("  %-20s n=%5d  model=%.4f  ceiling=%.4f  gap=%+.4f", lab, len(yy), r, cl, cl - r)
        out.setdefault("split", {})[lab] = {"n": int(len(yy)), "model": float(r),
                                            "ceiling": cl, "gap": float(cl - r)}

    Path("results/round6").mkdir(parents=True, exist_ok=True)
    Path("results/round6/noise_ceiling_empirical.json").write_text(json.dumps(out, indent=2))
    logger.info("wrote results/round6/noise_ceiling_empirical.json")


if __name__ == "__main__":
    main()
