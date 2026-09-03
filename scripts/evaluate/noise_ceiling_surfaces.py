"""Replicate-based noise ceiling for Kim, on every evaluation surface.

Round-9 correction. `noise_ceiling_empirical.py` reported a single number computed on
**development fold 0 alone** (n=19,747), which the manuscript then labelled "all Kim
rows" and compared against a Kim score that appears elsewhere as a held-out figure. Two
problems: a one-fold result is below the standard this project applies to everything
else, and the model score and the ceiling were quoted from different surfaces.

This recomputes the ceiling and the gap on all three matched development folds and on
the official held-out set, so the headroom claim can be stated per surface and as a
three-fold mean. The estimator itself is unchanged: a synthetic replicate is drawn from
the empirical distribution of replicate values observed near each row's efficiency
(nearest-neighbour matching on the ordered replicate pairs), which inherits
zero-discordance, heteroscedasticity and boundedness from the data instead of assuming
a noise model. Two independent draws give the replicate-replicate Spearman R; sqrt(R)
bounds what any model can score against a single observation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ceiling_surfaces")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
DEV = "results/round3/dev_recalibration/predictions_r4p2_ordSSM_oof_round3_dev_fold_{}.parquet"
HELDOUT = Path("results/round5/heldout_calibrated.parquet")
OUT = Path("results/round9/noise_ceiling_surfaces.json")
FULL_KEY = ["spacer", "rtt", "pbs", "cell_type", "pe_type", "cas9_type", "cas9_pam",
            "motif", "scaffold_name", "linker", "rt_name", "time",
            "PEmax", "epegRNA", "MLH1dn", "NRCH", "target_name"]
K_NEIGHBOURS, N_SIM, SEED = 40, 25, 20260812


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
    return np.concatenate([a, b]), np.concatenate([b, a])  # pairs are exchangeable


class EmpiricalReplicator:
    def __init__(self, y1: np.ndarray, y2: np.ndarray, k: int = K_NEIGHBOURS):
        order = np.argsort(y1)
        self.y1, self.y2, self.k = y1[order], y2[order], k

    def draw(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pos = np.searchsorted(self.y1, y)
        lo = np.clip(pos - self.k // 2, 0, len(self.y1) - self.k)
        return self.y2[lo + rng.integers(0, self.k, size=len(y))]


def main() -> None:
    y1, y2 = replicate_pairs()
    rep = EmpiricalReplicator(y1, y2)
    rng = np.random.default_rng(SEED)
    logger.info("empirical replicator built from %d ordered replicate pairs", len(y1))

    def ceiling(vals: np.ndarray) -> float:
        rs = [spearmanr(rep.draw(vals, rng), rep.draw(vals, rng)).statistic for _ in range(N_SIM)]
        return float(np.sqrt(max(np.mean(rs), 0.0)))

    surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in (0, 1, 2):
        d = pd.read_parquet(DEV.format(f))
        k = d[d.source_study == "deepprime"]
        surfaces[f"dev_fold_{f}"] = (k.true_efficiency.to_numpy(), k.predicted_efficiency.to_numpy())

    h = pd.read_parquet(HELDOUT)
    hk = h[h.source_study == "deepprime"]
    surfaces["heldout_ordSSM_member"] = (hk.true_efficiency.to_numpy(), hk.member_ordSSM.to_numpy())
    surfaces["heldout_frozen_ensemble"] = (  # same vector as Table 1, pre-calibration
        hk.true_efficiency.to_numpy(), hk.predicted_efficiency.to_numpy())

    out: dict[str, object] = {}
    logger.info("")
    logger.info("%-26s %7s %8s %9s %9s", "surface", "n", "model", "ceiling", "gap")
    for name, (y, pred) in surfaces.items():
        m = float(spearmanr(y, pred).statistic)
        c = ceiling(y)
        edits = y > 0
        me = float(spearmanr(y[edits], pred[edits]).statistic)
        ce = ceiling(y[edits])
        out[name] = {"n": int(len(y)), "pct_zero": float(100 * (y == 0).mean()),
                     "model": m, "ceiling": c, "gap": c - m,
                     "editing_only": {"n": int(edits.sum()), "model": me,
                                      "ceiling": ce, "gap": ce - me}}
        logger.info("%-26s %7d %8.4f %9.4f %+9.4f", name, len(y), m, c, c - m)

    devs = [out[f"dev_fold_{f}"] for f in (0, 1, 2)]
    out["dev_three_fold_mean"] = {
        "model": float(np.mean([d["model"] for d in devs])),
        "ceiling": float(np.mean([d["ceiling"] for d in devs])),
        "gap": float(np.mean([d["gap"] for d in devs])),
        "editing_only": {
            "model": float(np.mean([d["editing_only"]["model"] for d in devs])),
            "ceiling": float(np.mean([d["editing_only"]["ceiling"] for d in devs])),
            "gap": float(np.mean([d["editing_only"]["gap"] for d in devs])),
        },
    }
    logger.info("")
    logger.info("dev three-fold mean : model %.4f  ceiling %.4f  gap %+.4f",
                out["dev_three_fold_mean"]["model"], out["dev_three_fold_mean"]["ceiling"],
                out["dev_three_fold_mean"]["gap"])
    logger.info("editing rows only   : model %.4f  ceiling %.4f  gap %+.4f",
                out["dev_three_fold_mean"]["editing_only"]["model"],
                out["dev_three_fold_mean"]["editing_only"]["ceiling"],
                out["dev_three_fold_mean"]["editing_only"]["gap"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
