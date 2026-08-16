"""Paired target-level bootstrap comparison of PE-RankFormer vs OptiPrime (task spec §33).

Resamples *target groups* (not rows), because pegRNAs sharing a target are not
statistically independent, and computes the paired difference in Spearman rho on the
identical held-out rows.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paired_bootstrap")

N_BOOT = 2000
SEED = 20260812


def main() -> None:
    pdir = Path("data/interim/optiprime_compatible_test/predictions_20260814_124312")
    preds = pd.read_csv(pdir / "predictions.csv", index_col=0).reset_index(drop=True)
    joined = pd.read_csv(pdir / "joined_df.csv", index_col=0).reset_index(drop=True)
    op = pd.DataFrame({"record_id": joined["record_id"], "optiprime": preds["mean_pred"]})

    ours = pd.read_parquet("results/runs/eval_test_fold/predictions_pe_rankformer_ens6.parquet")
    ours = ours[ours.source_study == "hsu2026"]
    df = ours.merge(op, on="record_id", how="inner")

    # Cluster by PROTOSPACER, not by the (target, edit, context) ranking group. In the
    # Hsu libraries each design installs a distinct edit, so ranking groups are almost
    # all singletons and resampling them is just a row-level bootstrap -- which would
    # understate the CI. The protospacer is the real unit of statistical dependence here
    # (many designs share one) and is also the unit the CV folds were built on.
    corpus = pd.read_parquet("data/processed/optiprime_full_297962.parquet")[["record_id", "spacer"]]
    df = df.merge(corpus, on="record_id", how="left")
    assert df.spacer.notna().all(), "failed to attach protospacer for some rows"
    logger.info(
        "paired rows: %d across %d protospacers (vs %d singleton-ish ranking groups)",
        len(df), df.spacer.nunique(), df.target_group.nunique(),
    )

    y = df.true_efficiency.to_numpy()
    a = df.predicted_efficiency.to_numpy()  # PE-RankFormer ensemble
    b = df.optiprime.to_numpy()

    obs_a = spearmanr(y, a).statistic
    obs_b = spearmanr(y, b).statistic
    logger.info("observed: PE-RankFormer rho=%.4f, OptiPrime rho=%.4f, diff=%+.4f", obs_a, obs_b, obs_a - obs_b)

    # Resample target groups, not rows.
    groups = df.spacer.to_numpy()
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)

    diffs = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sampled])
        diffs[k] = spearmanr(y[idx], a[idx]).statistic - spearmanr(y[idx], b[idx]).statistic

    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_two_sided = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    logger.info(
        "paired target-level bootstrap (%d resamples of %d protospacer clusters): mean diff=%+.4f, 95%% CI [%+.4f, %+.4f]",
        N_BOOT, len(uniq), diffs.mean(), lo, hi,
    )
    logger.info("fraction of resamples where PE-RankFormer wins: %.4f", (diffs > 0).mean())
    logger.info("two-sided bootstrap p = %.2g (0 means < 1/%d)", p_two_sided, N_BOOT)

    out = {
        "n_paired_rows": int(len(df)),
        "n_protospacer_clusters": int(len(uniq)),
        "pe_rankformer_spearman": float(obs_a),
        "optiprime_spearman": float(obs_b),
        "observed_difference": float(obs_a - obs_b),
        "bootstrap_mean_difference": float(diffs.mean()),
        "bootstrap_ci95": [float(lo), float(hi)],
        "fraction_resamples_pe_rankformer_wins": float((diffs > 0).mean()),
        "two_sided_p": float(p_two_sided),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }
    Path("results/paired_bootstrap_vs_optiprime.json").write_text(json.dumps(out, indent=2))
    logger.info("wrote results/paired_bootstrap_vs_optiprime.json")


if __name__ == "__main__":
    main()
