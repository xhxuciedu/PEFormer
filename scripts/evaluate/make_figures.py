"""Publication-quality figures (task spec §43)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.evaluation.metrics import top_k_regret, within_target_spearman  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_figures")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    logger.info("wrote %s.png/.pdf", name)


def fig_pred_vs_observed(df: pd.DataFrame):
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(5, 4.5))
    hb = ax.hexbin(
        df.true_efficiency, df.predicted_efficiency, gridsize=50, cmap="viridis",
        mincnt=1, norm=LogNorm(),
    )
    fig.colorbar(hb, ax=ax, label="count (log scale)")
    ax.plot([0, 1], [0, 1], "r--", lw=1, alpha=0.7, label="y = x")

    # Binned calibration overlay: mean predicted vs. mean observed per decile.
    bins = np.linspace(0, 1, 11)
    df = df.copy()
    df["bin"] = pd.cut(df.true_efficiency, bins)
    means = df.groupby("bin", observed=True).agg(
        mean_true=("true_efficiency", "mean"), mean_pred=("predicted_efficiency", "mean")
    )
    ax.plot(means.mean_true, means.mean_pred, "o-", color="white", ms=4, lw=1.5, label="decile mean")
    ax.plot(means.mean_true, means.mean_pred, "o-", color="black", ms=3, lw=1)

    ax.set_xlabel("Observed editing efficiency")
    ax.set_ylabel("Predicted editing efficiency")
    ax.set_title("PE-RankFormer: predicted vs. observed\n(locked test fold)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "01_predicted_vs_observed")


def fig_global_correlation_comparison(table: pd.DataFrame):
    short = {
        "OptiPrime (official 5-model ensemble)": "OptiPrime\n(official 5-model)",
        "PE-RankFormer v2 ENSEMBLE (3-model, 285k corpus)": "PE-RankFormer v2\n(285k corpus)",
        "PE-RankFormer v1 ENSEMBLE (6-model, 262k corpus)": "PE-RankFormer v1\n(262k corpus)",
        "PE-RankFormer ENSEMBLE (6-model)": "PE-RankFormer\nENSEMBLE (6-model)",
        "PE-RankFormer (no-rank, single)": "PE-RankFormer\n(no-rank, single)",
        "PE-RankFormer (rank, single)": "PE-RankFormer\n(rank, single)",
    }
    labels = [short.get(m, m) for m in table["model"]]
    colors_p = ["#8C8C8C" if "OptiPrime" in m else "#4C72B0" for m in table["model"]]
    colors_s = ["#BFBFBF" if "OptiPrime" in m else "#DD8452" for m in table["model"]]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(table))
    b1 = ax.bar(x - 0.2, table["pearson"], width=0.4, color=colors_p, label="Pearson r")
    b2 = ax.bar(x + 0.2, table["spearman"], width=0.4, color=colors_s, label="Spearman ρ")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.axhline(
        table.loc[table.model.str.contains("OptiPrime"), "spearman"].iloc[0],
        color="k", ls=":", lw=1, alpha=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1.0)
    ax.set_title("Global correlation on the locked Hsu test subset (n=15,022)\ngrey = OptiPrime baseline")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    save(fig, "02_global_correlation_comparison")


def fig_within_target_distribution(df: pd.DataFrame, label: str):
    wt = within_target_spearman(df, "target_group", "true_efficiency", "predicted_efficiency", min_n=5)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(wt.spearman, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(wt.spearman.median(), color="red", ls="--", label=f"median={wt.spearman.median():.2f}")
    ax.set_xlabel("Within-target Spearman ρ")
    ax.set_ylabel("Number of target groups")
    ax.set_title(f"Within-target ranking distribution ({label}, n≥5)")
    ax.legend()
    save(fig, "03_within_target_spearman_distribution")


def fig_topk_regret(metrics_paths: dict[str, Path]):
    """Full test fold only -- Hsu's library design has ~no multi-pegRNA-per-edit
    groups, so top-k regret isn't computable on the Hsu-only OptiPrime comparison
    subset (see reports/pilot_results.md, within-target metrics caveat)."""
    import json

    rows = []
    for name, path in metrics_paths.items():
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        rows.append({"model": name, "top1_regret": m["top1_regret"], "top3_regret": m["top3_regret"]})
    if not rows:
        logger.warning("skipping top-k regret figure: no metrics available")
        return
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["top1_regret"], width=0.4, label="Top-1 regret")
    ax.bar(x + 0.2, table["top3_regret"], width=0.4, label="Top-3 regret")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=20, ha="right")
    ax.set_ylabel("Regret (efficiency units)")
    ax.set_title("Top-k pegRNA-selection regret, full test fold\n(lower is better)")
    ax.legend()
    save(fig, "04_topk_regret_comparison")


def fig_by_context(df: pd.DataFrame):
    from pe_rankformer.evaluation.metrics import global_metrics

    rows = []
    for (cell, pe), g in df.groupby(["cell_type", "pe_type"]):
        if len(g) < 20:
            continue
        gm = global_metrics(g.true_efficiency, g.predicted_efficiency)
        rows.append({"context": f"{cell}\n{pe}", "spearman": gm.spearman, "n": len(g)})
    ctx = pd.DataFrame(rows).sort_values("spearman")
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(ctx)), 4))
    ax.bar(ctx["context"], ctx["spearman"], color="#55A868")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("PE-RankFormer performance by experimental context (test fold)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    save(fig, "05_performance_by_context")


def fig_training_curves(history_paths: dict[str, Path]):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, path in history_paths.items():
        if not path.exists():
            continue
        h = pd.read_csv(path)
        axes[0].plot(h.epoch, h.train_loss, label=name)
        axes[1].plot(h.epoch, h.val_spearman, label=name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Spearman ρ")
    axes[1].set_title("Validation performance")
    axes[0].legend()
    axes[1].legend()
    save(fig, "06_training_curves")


def fig_ablation(metrics_paths: dict[str, Path]):
    """Uses the full-test-fold per-model metrics JSONs (not the Hsu-only comparison
    table), since Hsu's library design has ~no within-target ranking groups -- those
    metrics are only meaningful on the full, DeepPrime/PRIDICT-inclusive test fold."""
    import json

    rows = []
    for name, path in metrics_paths.items():
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        rows.append(
            {
                "model": name,
                "spearman": m["global"]["spearman"],
                "within_target_spearman": m["within_target_spearman_macro_mean"],
                "top3_regret": m["top3_regret"],
            }
        )
    if len(rows) < 2:
        logger.warning("skipping ablation figure: need both PE-RankFormer variants")
        return
    ablation = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    metrics = [("spearman", "Global Spearman ρ"), ("within_target_spearman", "Within-target Spearman ρ"), ("top3_regret", "Top-3 regret")]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(ablation["model"], ablation[col], color=["#4C72B0", "#DD8452"])
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    save(fig, "07_ablation_comparison")


def main() -> None:
    eval_dir = Path("results/runs/eval_test_fold")
    ens = eval_dir / "predictions_pe_rankformer_v2_ens3.parquet"
    a = pd.read_parquet(ens if ens.exists() else eval_dir / "predictions_model_a_rank.parquet")

    fig_pred_vs_observed(a)
    fig_within_target_distribution(a, "PE-RankFormer (rank)")
    fig_by_context(a)

    table_path = Path("results/hsu_benchmark_table.csv")
    if table_path.exists():
        table = pd.read_csv(table_path)
        fig_global_correlation_comparison(table)
    else:
        logger.warning("no hsu_benchmark_table.csv yet -- skipping comparison figures")

    model_metrics_paths = {
        "PE-RankFormer (rank)": eval_dir / "metrics_model_a_rank.json",
        "PE-RankFormer (no-rank)": eval_dir / "metrics_model_b_norank.json",
        "PE-RankFormer (no-context)": eval_dir / "metrics_model_c_nocontext.json",
        "PE-RankFormer v2 ENS": eval_dir / "metrics_pe_rankformer_v2_ens3.json",
    }
    fig_topk_regret(model_metrics_paths)
    fig_ablation(model_metrics_paths)

    import glob

    history_paths = {}
    for run_dir in sorted(glob.glob("results/runs/model_*")):
        name = Path(run_dir).name.split("_")[1]  # 'a' or 'b'
        history_paths[f"model_{name}"] = Path(run_dir) / "training_history.csv"
    if history_paths:
        fig_training_curves(history_paths)


if __name__ == "__main__":
    main()
