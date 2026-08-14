"""Hsu-specific benchmark table: PE-RankFormer vs. OptiPrime on the locked test fold
(task spec §35). Restricts to the Hsu/Liu subset of the test fold, which is the
apples-to-apples comparison the paper itself treats as primary.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.evaluation.metrics import (  # noqa: E402
    global_metrics,
    target_level_bootstrap_ci,
    top_k_regret,
    within_target_spearman,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_hsu_comparison")


def load_optiprime_predictions(pred_dir: Path, test_csv_dir: Path) -> pd.DataFrame:
    """OptiPrime's own output: predictions.csv (row order matches joined_df.csv) and
    joined_df.csv (the concatenated, re-ordered input). Join back to our record_id via
    the row content (spacer+pbs+rtt+full_unedited is a unique enough key here)."""
    preds = pd.read_csv(pred_dir / "predictions.csv", index_col=0)
    joined = pd.read_csv(pred_dir / "joined_df.csv", index_col=0)
    assert len(preds) == len(joined)
    joined = joined.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    out = pd.DataFrame(
        {
            "record_id": joined["record_id"],
            "optiprime_pred": preds["mean_pred"],
            "n_padded": joined["n_padded"],
        }
    )
    return out


def main() -> None:
    eval_dir = Path("results/runs/eval_test_fold")
    a = pd.read_parquet(eval_dir / "predictions_model_a_rank.parquet")
    b_path = eval_dir / "predictions_model_b_norank.parquet"
    b = pd.read_parquet(b_path) if b_path.exists() else None

    hsu_test = a[a.source_study == "hsu2026"].copy()
    logger.info("Hsu test-fold rows: %d", len(hsu_test))

    op_dirs = sorted(Path("data/interim/optiprime_compatible_test").glob("predictions_*"))
    results = {}
    if op_dirs and (op_dirs[-1] / "predictions.csv").exists():
        op = load_optiprime_predictions(op_dirs[-1], Path("data/interim/optiprime_compatible_test"))
        merged = hsu_test.merge(op, on="record_id", how="inner")
        logger.info("matched %d / %d Hsu test rows to OptiPrime predictions", len(merged), len(hsu_test))
        results["OptiPrime"] = merged
    else:
        logger.warning("no OptiPrime predictions found yet")

    results["PE-RankFormer (rank)"] = hsu_test
    if b is not None:
        results["PE-RankFormer (no-rank)"] = b[b.source_study == "hsu2026"].copy()

    rows = []
    for name, df in results.items():
        pred_col = "optiprime_pred" if "optiprime_pred" in df.columns else "predicted_efficiency"
        gm = global_metrics(df.true_efficiency, df[pred_col])
        wt = within_target_spearman(df, "target_group", "true_efficiency", pred_col, min_n=5)
        wt_point, wt_lo, wt_hi = target_level_bootstrap_ci(wt.spearman.tolist())
        r1 = top_k_regret(df, "target_group", "true_efficiency", pred_col, k=1)
        r3 = top_k_regret(df, "target_group", "true_efficiency", pred_col, k=3)
        r1_point, r1_lo, r1_hi = target_level_bootstrap_ci(r1.regret.tolist())
        r3_point, r3_lo, r3_hi = target_level_bootstrap_ci(r3.regret.tolist())
        rows.append(
            {
                "model": name,
                "n": len(df),
                "pearson": gm.pearson,
                "spearman": gm.spearman,
                "mae": gm.mae,
                "rmse": gm.rmse,
                "within_target_spearman": wt_point,
                "within_target_spearman_ci95": f"[{wt_lo:.3f}, {wt_hi:.3f}]",
                "top1_regret": r1_point,
                "top1_regret_ci95": f"[{r1_lo:.4f}, {r1_hi:.4f}]",
                "top3_regret": r3_point,
                "top3_regret_ci95": f"[{r3_lo:.4f}, {r3_hi:.4f}]",
            }
        )

    table = pd.DataFrame(rows)
    out_dir = Path("results")
    table.to_csv(out_dir / "hsu_benchmark_table.csv", index=False)
    logger.info("\n%s", table.to_string(index=False))
    logger.info("wrote %s", out_dir / "hsu_benchmark_table.csv")


if __name__ == "__main__":
    main()
