"""Zero-floor tie matching (round-5 independent lead, not in the spec).

Discovered while doing the §15 calibration work: a PCHIP calibrator whose output was
clipped to [0,1] scored +0.0046 Spearman above the uncalibrated ensemble on dev, which
should be impossible for a monotone map. The cause is tie structure, not monotonicity.

**28.4% of dev rows have efficiency exactly 0.0** -- one tie block of ~10,140 rows. A
strictly-increasing predictor gives every one of them a distinct rank, and all of that
internal ordering is arbitrary. Spearman compares against a target where those rows are
mutually tied, so the arbitrary ordering is pure penalty. Assigning a single tied value
to the lowest-scoring rows matches the target's tie structure and recovers it.

The trade-off is exact and gives an interior optimum: flooring too few rows leaves the
penalty in place, flooring too many destroys real ordering among genuinely non-zero
rows. The floor fraction is therefore tuned -- nested on dev folds, never on the
scoring fold -- and then checked on the round-4 lockbox.

This is a *ranking* change, not a units change, so unlike calibration it cannot be
applied to the frozen round-4 held-out predictions without re-freezing and re-running
the single held-out evaluation. It is evaluated here on dev and lockbox only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tie_floor")

DEV_FOLDS = ["round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"]
PRED = Path("results/round3/dev_recalibration")
MEMBERS = ["r4p2_ordSSM_oof", "r4p2_ssm_oof", "r4p2_ordC_oof", "r4p2_ordA_oof", "r3_familyA_oof"]
FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _rank01(x) -> np.ndarray:
    x = np.asarray(x)
    return rankdata(x) / len(x)


def load(fold: str) -> pd.DataFrame:
    base = None
    for n in MEMBERS:
        d = pd.read_parquet(PRED / f"predictions_{n}_{fold}.parquet")
        keep = (d[["record_id", "true_efficiency", "source_study", "predicted_efficiency"]]
                if base is None else d[["record_id", "predicted_efficiency"]]).rename(
            columns={"predicted_efficiency": n})
        base = keep if base is None else base.merge(keep, on="record_id")
    base["ens"] = np.mean([_rank01(base[m]) for m in MEMBERS], axis=0)
    return base


def apply_floor(s: np.ndarray, frac: float) -> np.ndarray:
    """Collapse the lowest `frac` of scores onto one shared value, preserving the rest."""
    if frac <= 0:
        return s
    out = s.copy()
    cut = np.quantile(s, frac)
    out[s <= cut] = s.min() - 1e-9  # a single value strictly below every survivor
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/round5/tie_floor.json"))
    args = ap.parse_args()

    folds = {f: load(f) for f in DEV_FOLDS}
    out: dict = {}

    z = np.mean([(folds[f].true_efficiency == 0).mean() for f in DEV_FOLDS])
    logger.info("target zero-mass on dev: %.1f%%", 100 * z)

    logger.info("")
    logger.info("--- floor fraction sweep (each fold scored independently) ---")
    logger.info("%-8s %s", "frac", "  ".join(f"{f[-6:]:>8}" for f in DEV_FOLDS) + "     mean")
    sweep = {}
    for frac in FRACTIONS:
        rs = [spearmanr(folds[f].true_efficiency, apply_floor(folds[f].ens.to_numpy(), frac)).statistic
              for f in DEV_FOLDS]
        sweep[frac] = rs
        logger.info("%-8.2f %s  %8.4f", frac, "  ".join(f"{r:8.4f}" for r in rs), np.mean(rs))
    out["sweep"] = {str(k): [float(x) for x in v] for k, v in sweep.items()}

    # Nested selection: choose the fraction on the two other folds, apply to the held one.
    logger.info("")
    logger.info("--- nested: fraction chosen on the OTHER two folds ---")
    base_scores, tuned_scores, chosen = [], [], []
    for held in DEV_FOLDS:
        others = [f for f in DEV_FOLDS if f != held]
        best = max(FRACTIONS, key=lambda fr: np.mean(
            [spearmanr(folds[o].true_efficiency, apply_floor(folds[o].ens.to_numpy(), fr)).statistic
             for o in others]))
        b = spearmanr(folds[held].true_efficiency, folds[held].ens).statistic
        t = spearmanr(folds[held].true_efficiency,
                      apply_floor(folds[held].ens.to_numpy(), best)).statistic
        base_scores.append(b); tuned_scores.append(t); chosen.append(best)
        logger.info("  %s: frac=%.2f  base=%.4f -> %.4f  (%+.4f)", held, best, b, t, t - b)

    delta = float(np.mean(tuned_scores) - np.mean(base_scores))
    logger.info("")
    logger.info("nested mean delta: %+.4f   positive on all folds: %s",
                delta, all(t > b for t, b in zip(tuned_scores, base_scores)))
    out["nested"] = {"base": [float(x) for x in base_scores],
                     "tuned": [float(x) for x in tuned_scores],
                     "chosen_fractions": chosen, "mean_delta": delta,
                     "positive_all_folds": bool(all(t > b for t, b in zip(tuned_scores, base_scores)))}

    # Per-study: the zero-mass differs sharply between Liu and Kim, so the effect should
    # too. If it does not, the mechanism is not what is claimed.
    logger.info("")
    logger.info("--- by study (fraction fixed at the dev-wide optimum) ---")
    bestfrac = max(FRACTIONS, key=lambda fr: np.mean(sweep[fr]))
    out["best_fraction_dev"] = bestfrac
    for study in ("hsu2026", "deepprime"):
        bs, ts, zs = [], [], []
        for f in DEV_FOLDS:
            sub = folds[f][folds[f].source_study == study]
            zs.append((sub.true_efficiency == 0).mean())
            bs.append(spearmanr(sub.true_efficiency, sub.ens).statistic)
            ts.append(spearmanr(sub.true_efficiency,
                                apply_floor(sub.ens.to_numpy(), bestfrac)).statistic)
        logger.info("  %-10s zero-mass=%.1f%%  base=%.4f -> %.4f  (%+.4f)",
                    study, 100 * np.mean(zs), np.mean(bs), np.mean(ts), np.mean(ts) - np.mean(bs))
        out.setdefault("by_study", {})[study] = {
            "zero_mass": float(np.mean(zs)), "base": float(np.mean(bs)),
            "tuned": float(np.mean(ts)), "delta": float(np.mean(ts) - np.mean(bs))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
