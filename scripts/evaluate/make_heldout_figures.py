"""Figures for the matched-protocol head-to-head on OptiPrime's own held-out test set."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heldout_figures")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

OP_COLOR = "#8C8C8C"
PR_COLOR = "#4C72B0"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    logger.info("wrote %s", name)


def fig_metrics(table: pd.DataFrame):
    """Grouped bars over all four metrics. Correlations: higher is better; errors: lower."""
    t = table[~table.model.str.contains("80%")].copy()
    short = ["OptiPrime\n(5-model)", "PE-RankFormer\n(5-model CV)"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    for ax, (col, title, better) in zip(
        axes,
        [("pearson", "Pearson r", "higher"), ("spearman", "Spearman ρ", "higher"),
         ("mae", "MAE", "lower"), ("rmse", "RMSE", "lower")],
    ):
        bars = ax.bar(short, t[col], color=[OP_COLOR, PR_COLOR], width=0.6)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
        ax.set_title(f"{title}\n({better} is better)", fontsize=10)
        ax.set_ylim(0, t[col].max() * 1.25)
        plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.suptitle(
        "Matched-protocol head-to-head on OptiPrime's own held-out test set (n=9,175)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "10_heldout_metrics.png")
    fig.savefig(OUT / "10_heldout_metrics.pdf")
    plt.close(fig)
    logger.info("wrote 10_heldout_metrics")


def fig_bootstrap(boot: dict):
    """The paired bootstrap distribution -- the honest summary: it straddles zero."""
    diffs = np.load("results/heldout_bootstrap_diffs.npy")
    lo, hi = boot["ci95"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.hist(diffs, bins=50, color=PR_COLOR, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvspan(lo, hi, color=PR_COLOR, alpha=0.12, label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    ax.axvline(0, color="k", lw=1.3, ls="--", label="no difference")
    ax.axvline(boot["observed_difference"], color="#C44E52", lw=1.8,
               label=f"observed {boot['observed_difference']:+.4f}")
    ax.set_xlabel("Spearman ρ difference (PE-RankFormer − OptiPrime)")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title(
        f"Paired protospacer-clustered bootstrap (2000 resamples)\n"
        f"p = {boot['two_sided_p']:.2f} — the two models are statistically indistinguishable",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    save(fig, "11_heldout_bootstrap")


def fig_scatter(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, (col, name) in zip(axes, [("op", "OptiPrime"), ("predicted_efficiency", "PE-RankFormer")]):
        hb = ax.hexbin(df.true_efficiency, df[col], gridsize=45, cmap="viridis",
                       mincnt=1, norm=LogNorm())
        ax.plot([0, 1], [0, 1], "r--", lw=1, alpha=0.7)
        ax.set_xlabel("Observed editing efficiency")
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Predicted editing efficiency")
    fig.colorbar(hb, ax=axes, label="count (log scale)")
    fig.suptitle("Held-out test set: both models, identical rows", fontsize=11)
    save(fig, "12_heldout_calibration")


def main() -> None:
    import json

    table = pd.read_csv("results/heldout_benchmark_table.csv")
    fig_metrics(table)
    fig_bootstrap(json.loads(Path("results/heldout_bootstrap_cv5.json").read_text()))

    ours = pd.read_parquet("results/runs/eval_test_fold/predictions_cv5_ens.parquet")
    op = pd.read_parquet("results/optiprime_heldout_predictions.parquet")
    fig_scatter(ours.merge(op, on="record_id"))


if __name__ == "__main__":
    main()
