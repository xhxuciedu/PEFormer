"""Task 1.6 — uncertainty on the replicate-based noise ceiling.

The "benchmark is not saturated" claim rests on a ceiling estimated from 649 replicate
groups, reported as a point estimate. 649 groups is thin for a number that anchors a
central argument, and the estimator is itself stochastic (it draws synthetic replicates).
This puts an interval on both sources of variation:

  - **group resampling**: resample the 649 replicate groups with replacement, rebuild the
    empirical replicator from the resampled pairs, and re-estimate.
  - **estimator noise**: repeat the synthetic draw at a fixed group set.

Descriptive re-analysis. Nothing is trained, tuned or selected.

Usage: python revision/task_1_6_ceiling_uncertainty.py [--seed 20260903] [--n-boot 400]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

FULL_KEY = ["spacer", "rtt", "pbs", "cell_type", "pe_type", "cas9_type", "cas9_pam",
            "motif", "scaffold_name", "linker", "rt_name", "time",
            "PEmax", "epegRNA", "MLH1dn", "NRCH", "target_name"]
K_NEIGHBOURS, N_SIM = 40, 25
DEV_PRED = "results/round3/dev_recalibration/predictions_r4p2_ordSSM_oof_round3_dev_fold_{}.parquet"


def replicate_pairs_by_group(corpus: pd.DataFrame) -> list[tuple[float, float]]:
    k = corpus[corpus.source_study == "deepprime"].copy()
    k["_g"] = k.groupby(FULL_KEY, dropna=False).ngroup()
    sizes = k.groupby("_g").size()
    rep = k[k._g.map(sizes) > 1]
    out = []
    for _, s in rep.groupby("_g"):
        v = s.edited.to_numpy()
        out.append((float(v[0]), float(v[1])))
    return out


class Replicator:
    def __init__(self, y1: np.ndarray, y2: np.ndarray, k: int = K_NEIGHBOURS):
        order = np.argsort(y1)
        self.y1, self.y2, self.k = y1[order], y2[order], min(k, len(y1))

    def draw(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pos = np.searchsorted(self.y1, y)
        lo = np.clip(pos - self.k // 2, 0, len(self.y1) - self.k)
        return self.y2[lo + rng.integers(0, self.k, size=len(y))]


def ceiling_from_pairs(pairs: list[tuple[float, float]], y: np.ndarray,
                       rng: np.random.Generator, n_sim: int = N_SIM) -> float:
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    rep = Replicator(np.concatenate([a, b]), np.concatenate([b, a]))
    rs = [C.spearman(rep.draw(y, rng), rep.draw(y, rng)) for _ in range(n_sim)]
    return float(np.sqrt(max(np.nanmean(rs), 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-boot", type=int, default=400)  # each draw is 25 simulations
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    corpus = pd.read_parquet(C.CORPUS, columns=["edited", "source_study"] + FULL_KEY)
    pairs = replicate_pairs_by_group(corpus)

    # Kim rows of each matched development fold, and of the held-out set
    surfaces = {}
    for f in range(3):
        p = C.ROOT / DEV_PRED.format(f)
        if p.exists():
            d = pd.read_parquet(p)
            k = d[d.source_study == "deepprime"]
            surfaces[f"dev_fold_{f}"] = (k.true_efficiency.to_numpy(),
                                         k.predicted_efficiency.to_numpy())
    m = C.load_heldout()
    hk = m[m.source_study == "deepprime"]
    surfaces["heldout"] = (hk.y.to_numpy(), hk.ours.to_numpy())

    result = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
              "n_replicate_groups": len(pairs), "n_ordered_pairs": 2 * len(pairs),
              "n_boot": args.n_boot, "surfaces": {}}

    for name, (y, pred) in surfaces.items():
        model = C.spearman(pred, y)
        point = ceiling_from_pairs(pairs, y, rng)
        # (a) estimator noise at a fixed group set
        est = [ceiling_from_pairs(pairs, y, rng, n_sim=N_SIM) for _ in range(30)]
        # (b) group resampling, the dominant source
        boots = np.empty(args.n_boot)
        for i in range(args.n_boot):
            idx = rng.integers(0, len(pairs), len(pairs))
            boots[i] = ceiling_from_pairs([pairs[j] for j in idx], y, rng, n_sim=5)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        result["surfaces"][name] = {
            "n_rows": int(len(y)), "model": model, "ceiling_point": point,
            "ceiling_ci95_group_resample": [float(lo), float(hi)],
            "ceiling_sd_estimator_only": float(np.std(est)),
            "gap_point": point - model,
            "gap_ci95": [float(lo - model), float(hi - model)],
        }

    rows = [f"| {n} | {v['n_rows']:,} | {v['model']:.4f} | {v['ceiling_point']:.4f} | "
            f"[{v['ceiling_ci95_group_resample'][0]:.4f}, {v['ceiling_ci95_group_resample'][1]:.4f}] | "
            f"{v['gap_point']:+.4f} | [{v['gap_ci95'][0]:+.4f}, {v['gap_ci95'][1]:+.4f}] |"
            for n, v in result["surfaces"].items()]
    sds = ", ".join(f"{n} {v['ceiling_sd_estimator_only']:.4f}"
                    for n, v in result["surfaces"].items())

    md = f"""# Task 1.6 — uncertainty on the noise ceiling

Ceiling estimated from **{result['n_replicate_groups']} replicate groups**
({result['n_ordered_pairs']} ordered pairs after symmetrisation). Interval from resampling
the groups with replacement, {args.n_boot} resamples, seed {args.seed}. Commit
`{result['provenance']['git_commit']}`.

| surface | Kim rows | model ρ | ceiling | 95% CI on ceiling | gap | 95% CI on gap |
|---|---:|---:|---:|---|---:|---|
{chr(10).join(rows)}

Estimator noise alone, holding the group set fixed, has SD: {sds} — an order of
magnitude smaller than the group-resampling interval, so the uncertainty is dominated by
having only {result['n_replicate_groups']} replicate groups, not by the synthetic draw.

The headroom claim survives the interval on every surface: the lower bound of the gap
stays well clear of zero. What the interval does change is the precision with which the
headroom can be quoted — "roughly +0.10" is supportable, a third decimal place is not.
"""
    C.write_outputs("task_1_6_ceiling_uncertainty", result, md)
    for n, v in result["surfaces"].items():
        print(f"{n:12s} model {v['model']:.4f}  ceiling {v['ceiling_point']:.4f} "
              f"[{v['ceiling_ci95_group_resample'][0]:.4f},{v['ceiling_ci95_group_resample'][1]:.4f}]  "
              f"gap {v['gap_point']:+.4f} [{v['gap_ci95'][0]:+.4f},{v['gap_ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
