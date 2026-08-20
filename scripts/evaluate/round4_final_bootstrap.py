"""Round-4 final significance test: PE-RankFormer vs OptiPrime vs round-3, on the
identical 20,509 held-out rows (spec §25, 5000 resamples).

Resamples **protospacer clusters**, not rows. pegRNAs sharing a protospacer are not
independent, and the protospacer is the unit the CV folds were built on; a row-level
bootstrap would understate the interval badly.

All three prediction sets are joined on record_id and compared on exactly the rows
where all three exist, so the differences are paired and no model is scored on a
different subset than its comparator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("round4_bootstrap")

N_BOOT = 5000
SEED = 20260812
OP_DIR = Path("data/interim/optiprime_heldout_full/predictions_20260818_083148")


def main() -> None:
    r4 = pd.read_parquet("results/round4/heldout/predictions_round4_final.parquet")
    r3 = pd.read_parquet("results/round3/heldout/predictions_r3_final_ensemble.parquet")[
        ["record_id", "predicted_efficiency"]
    ].rename(columns={"predicted_efficiency": "r3"})

    preds = pd.read_csv(OP_DIR / "predictions.csv", index_col=0).reset_index(drop=True)
    joined = pd.read_csv(OP_DIR / "joined_df.csv", index_col=0).reset_index(drop=True)
    op = pd.DataFrame({"record_id": joined["record_id"], "op": preds["mean_pred"]})

    df = r4.merge(r3, on="record_id").merge(op, on="record_id")
    corpus = pd.read_parquet(
        "data/processed/optiprime_official_318471.parquet", columns=["record_id", "spacer"]
    )
    df = df.merge(corpus, on="record_id", how="left")
    assert df.spacer.notna().all(), "failed to attach protospacer for some rows"
    assert len(df) == len(r4), f"lost rows in join: {len(df)} vs {len(r4)}"

    y = df.true_efficiency.to_numpy()
    models = {
        "round4": df.predicted_efficiency.to_numpy(),
        "round3": df.r3.to_numpy(),
        "optiprime": df.op.to_numpy(),
    }
    obs = {k: spearmanr(y, v).statistic for k, v in models.items()}
    logger.info("paired rows: %d across %d protospacer clusters", len(df), df.spacer.nunique())
    for k, v in obs.items():
        logger.info("  %-10s spearman = %.4f", k, v)

    groups = df.spacer.to_numpy()
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)

    comparisons = [("round4", "optiprime"), ("round4", "round3"), ("round3", "optiprime")]
    diffs = {c: np.empty(N_BOOT) for c in comparisons}
    r4_abs = np.empty(N_BOOT)

    for k in range(N_BOOT):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sampled])
        rho = {m: spearmanr(y[idx], v[idx]).statistic for m, v in models.items()}
        for a, b in comparisons:
            diffs[(a, b)][k] = rho[a] - rho[b]
        r4_abs[k] = rho["round4"]

    out = {"n_rows": int(len(df)), "n_clusters": int(len(uniq)), "n_boot": N_BOOT,
           "seed": SEED, "observed": {k: float(v) for k, v in obs.items()}}

    logger.info("")
    logger.info("--- paired protospacer-clustered bootstrap (%d resamples) ---", N_BOOT)
    for (a, b), d in diffs.items():
        lo, hi = np.quantile(d, [0.025, 0.975])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        logger.info("  %-10s - %-10s = %+.4f  95%% CI [%+.4f, %+.4f]  wins %.3f  p=%.2g",
                    a, b, obs[a] - obs[b], lo, hi, (d > 0).mean(), p)
        out[f"{a}_vs_{b}"] = {
            "observed_difference": float(obs[a] - obs[b]),
            "ci95": [float(lo), float(hi)],
            "fraction_wins": float((d > 0).mean()),
            "two_sided_p": float(p),
        }

    lo, hi = np.quantile(r4_abs, [0.025, 0.975])
    frac90 = float((r4_abs > 0.90).mean())
    logger.info("")
    logger.info("  round-4 absolute rho: %.4f  95%% CI [%.4f, %.4f]", obs["round4"], lo, hi)
    logger.info("  fraction of resamples with rho > 0.90: %.3f", frac90)
    out["round4_absolute"] = {"rho": float(obs["round4"]), "ci95": [float(lo), float(hi)],
                              "fraction_above_0.90": frac90}

    # Per-study breakdown on the same rows.
    logger.info("")
    for study, sub in df.groupby("source_study"):
        ys = sub.true_efficiency.to_numpy()
        logger.info("  %-22s n=%5d  round4=%.4f  round3=%.4f  optiprime=%.4f",
                    study, len(sub), spearmanr(ys, sub.predicted_efficiency).statistic,
                    spearmanr(ys, sub.r3).statistic, spearmanr(ys, sub.op).statistic)
        out.setdefault("by_study", {})[study] = {
            "n": int(len(sub)),
            "round4": float(spearmanr(ys, sub.predicted_efficiency).statistic),
            "round3": float(spearmanr(ys, sub.r3).statistic),
            "optiprime": float(spearmanr(ys, sub.op).statistic),
        }

    Path("results/round4").mkdir(parents=True, exist_ok=True)
    Path("results/round4/final_bootstrap.json").write_text(json.dumps(out, indent=2))
    logger.info("wrote results/round4/final_bootstrap.json")


if __name__ == "__main__":
    main()
