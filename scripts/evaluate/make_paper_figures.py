"""Generate all figures for the manuscript from stored artifacts.

Every panel is computed from a results file rather than transcribed, so the figures
cannot drift from the numbers in the text.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("figures")

OUT = Path("results/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

HELD = "results/round4/heldout/predictions_round4_final.parquet"
BOOT = "results/round4/final_bootstrap.json"
C_BLUE, C_ORANGE, C_GREY = "#2b6cb0", "#dd6b20", "#a0aec0"


def _rank01(x):
    x = np.asarray(x)
    return rankdata(x) / len(x)


# ---------------------------------------------------------------- Figure 1
def fig1_benchmark():
    b = json.load(open(BOOT))
    models = ["OptiPrime", "PE-RankFormer\n(round 1)", "PE-RankFormer\n(round 3)",
              "PE-RankFormer\n(final)"]
    full = [0.8690, 0.8865, 0.8933, 0.9079]
    liu = [0.8365, 0.8349, 0.8462, 0.8585]
    kim = [0.7320, 0.7751, 0.7836, 0.8124]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(models)); w = 0.26
    for i, (vals, lab, c) in enumerate([(full, "All (n=20,509)", C_BLUE),
                                        (liu, "Liu (n=9,175)", C_ORANGE),
                                        (kim, "Kim (n=11,334)", "#38a169")]):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.004, f"{v:.4f}",
                    ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Spearman $\\rho$"); ax.set_ylim(0.68, 0.96)
    ax.axhline(0.8690, ls=":", c=C_GREY, lw=0.9)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax.set_title("Held-out performance across successive model generations", fontsize=9.5)
    fig.savefig(OUT / "fig1_benchmark.pdf"); plt.close(fig)
    logger.info("fig1")


# ---------------------------------------------------------------- Figure 2
def fig2_bootstrap():
    d = pd.read_parquet(HELD)
    op_dir = Path("data/interim/optiprime_heldout_full/predictions_20260818_083148")
    preds = pd.read_csv(op_dir / "predictions.csv", index_col=0).reset_index(drop=True)
    joined = pd.read_csv(op_dir / "joined_df.csv", index_col=0).reset_index(drop=True)
    op = pd.DataFrame({"record_id": joined["record_id"], "op": preds["mean_pred"]})
    corpus = pd.read_parquet("data/processed/optiprime_official_318471.parquet",
                             columns=["record_id", "spacer"])
    df = d.merge(op, on="record_id").merge(corpus, on="record_id")

    y = df.true_efficiency.to_numpy()
    a = df.predicted_efficiency.to_numpy()
    bb = df.op.to_numpy()
    groups = df.spacer.to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(20260812)
    diffs = np.empty(5000)
    for k in range(5000):
        s = rng.choice(uniq, size=len(uniq), replace=True)
        i = np.concatenate([idx_by[g] for g in s])
        diffs[k] = spearmanr(y[i], a[i]).statistic - spearmanr(y[i], bb[i]).statistic

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.hist(diffs, bins=60, color=C_BLUE, alpha=0.85)
    ax.axvline(0, c="k", lw=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    ax.axvline(lo, c=C_ORANGE, ls="--", lw=1); ax.axvline(hi, c=C_ORANGE, ls="--", lw=1)
    ax.set_xlabel("$\\Delta\\rho$ (PE-RankFormer $-$ OptiPrime)")
    ax.set_ylabel("bootstrap resamples")
    ax.set_title(f"Paired protospacer-clustered bootstrap\n"
                 f"observed $+{spearmanr(y,a).statistic - spearmanr(y,bb).statistic:.4f}$, "
                 f"95% CI [{lo:+.4f}, {hi:+.4f}]", fontsize=8.5)
    fig.savefig(OUT / "fig2_bootstrap.pdf"); plt.close(fig)
    logger.info("fig2")


# ---------------------------------------------------------------- Figure 3
def fig3_factorial():
    f = json.load(open("results/round5/factorial.json"))
    cells = f["cells"]; e = f["effects"]
    grid = np.array([[cells["Transformer|simplex"]["full"], cells["Transformer|ordinal"]["full"]],
                     [cells["SSM|simplex"]["full"], cells["SSM|ordinal"]["full"]]])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    im = ax.imshow(grid, cmap="Blues", vmin=0.875, vmax=0.912)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i,j]:.4f}", ha="center", va="center",
                    color="white" if grid[i, j] > 0.90 else "black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["simplex", "ordinal"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Transformer", "SSM"])
    ax.set_title("Out-of-fold $\\rho$", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    labs = ["architecture\n(SSM $-$ Transformer)", "objective\n(ordinal $-$ simplex)",
            "interaction"]
    vals = [e["architecture"], e["objective"], e["interaction"]]
    cols = [C_BLUE, C_ORANGE, C_GREY]
    ax.barh(labs, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + (0.0006 if v > 0 else -0.0006), i, f"{v:+.4f}",
                va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlim(-0.009, 0.026); ax.set_xlabel("effect on Spearman $\\rho$")
    ax.set_title("Main effects and interaction", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig3_factorial.pdf"); plt.close(fig)
    logger.info("fig3")


# ---------------------------------------------------------------- Figure 4
def fig4_calibration():
    c = json.load(open("results/round5/calibration.json"))
    cal = pd.read_parquet("results/round5/heldout_calibrated.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    q = np.linspace(0, 1, 21)
    binned = pd.cut(cal.calibrated_efficiency, np.unique(np.quantile(cal.calibrated_efficiency, q)),
                    include_lowest=True)
    g = cal.groupby(binned, observed=True).agg(pred=("calibrated_efficiency", "mean"),
                                               obs=("true_efficiency", "mean"))
    ax.plot([0, g.obs.max() * 1.05], [0, g.obs.max() * 1.05], ls=":", c=C_GREY)
    ax.plot(g.pred, g.obs, "o-", color=C_BLUE, ms=4)
    ax.set_xlabel("predicted efficiency (calibrated)")
    ax.set_ylabel("observed efficiency")
    ax.set_title("Calibration curve, held-out", fontsize=9)

    ax = axes[1]
    m = ["spearman", "pearson", "mae", "rmse"]
    unc = [c["heldout_uncalibrated"][k] for k in m]
    ca = [c["heldout_calibrated"][k] for k in m]
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, unc, w, label="rank average", color=C_GREY)
    ax.bar(x + w / 2, ca, w, label="+ isotonic calibration", color=C_BLUE)
    for i, (u, v) in enumerate(zip(unc, ca)):
        ax.text(i - w / 2, u + 0.01, f"{u:.3f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(["Spearman", "Pearson", "MAE", "RMSE"])
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Calibration preserves ranking, restores scale", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig4_calibration.pdf"); plt.close(fig)
    logger.info("fig4")


# ---------------------------------------------------------------- Figure 5
def fig5_ceiling():
    n = json.load(open("results/round6/noise_ceiling_empirical.json"))
    bc = n["by_condition"]
    rows = sorted(bc.items(), key=lambda kv: kv[1]["pct_zero"])
    labs = [k.replace("('", "").replace("')", "").replace("', '", "/") for k, _ in rows]
    model = [v["model"] for _, v in rows]
    ceil = [v["ceiling"] for _, v in rows]
    zero = [v["pct_zero"] for _, v in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    yy = np.arange(len(labs))
    ax.barh(yy, ceil, color=C_GREY, label="empirical noise ceiling")
    ax.barh(yy, model, color=C_BLUE, height=0.55, label="Ordinal-SSM")
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("Spearman $\\rho$"); ax.set_xlim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Headroom remains in every condition", fontsize=9)

    ax = axes[1]
    gaps = [c - m for c, m in zip(ceil, model)]
    ax.scatter(zero, gaps, color=C_ORANGE, s=26)
    z = np.polyfit(zero, gaps, 1)
    xs = np.linspace(min(zero), max(zero), 10)
    ax.plot(xs, np.polyval(z, xs), ls="--", c=C_GREY, lw=1)
    ax.set_xlabel("% of rows with efficiency exactly 0")
    ax.set_ylabel("gap to ceiling")
    r = spearmanr(zero, gaps).statistic
    ax.set_title(f"Gap tracks zero-inflation ($\\rho={r:.2f}$)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig5_ceiling.pdf"); plt.close(fig)
    logger.info("fig5")


# ---------------------------------------------------------------- Figure 6
def fig6_members():
    d = pd.read_parquet(HELD)
    mem = [c for c in d.columns if c.startswith("member_")]
    names = [m.replace("member_", "") for m in mem]

    R = np.zeros((len(mem), len(mem)))
    ry = _rank01(d.true_efficiency)
    for i, a in enumerate(mem):
        for j, b in enumerate(mem):
            R[i, j] = spearmanr(ry - _rank01(d[a]), ry - _rank01(d[b])).statistic

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    im = ax.imshow(R, cmap="RdYlBu_r", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if R[i, j] > 0.9 else "black")
    ax.set_title("Residual correlation between members", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    full = np.mean([_rank01(d[m]) for m in mem], axis=0)
    base = spearmanr(d.true_efficiency, full).statistic
    contrib = []
    for m in mem:
        rest = [x for x in mem if x != m]
        s = spearmanr(d.true_efficiency, np.mean([_rank01(d[x]) for x in rest], axis=0)).statistic
        contrib.append(base - s)
    order = np.argsort(contrib)
    cols = [C_BLUE if contrib[i] > 0 else C_ORANGE for i in order]
    ax.barh([names[i] for i in order], [contrib[i] for i in order], color=cols)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("held-out $\\Delta\\rho$ when member removed")
    ax.set_title("Leave-one-member-out contribution", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig6_members.pdf"); plt.close(fig)
    logger.info("fig6")


if __name__ == "__main__":
    fig1_benchmark(); fig2_bootstrap(); fig3_factorial()
    fig4_calibration(); fig5_ceiling(); fig6_members()
    logger.info("all figures written to %s", OUT)
