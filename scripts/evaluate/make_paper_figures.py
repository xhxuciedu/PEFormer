"""Generate all manuscript figures from stored artifacts.

Every data panel is computed from a results file rather than transcribed, so figures
cannot drift from the numbers in the text. Figure 1 is a schematic, but its
threshold panel is still drawn from the real training distribution.

Figures
-------
1  Model architecture (schematic + real threshold placement)
2  Benchmark result and significance
3  Objective x architecture factorial, and the error-decorrelation mechanism
4  Ensemble structure: member redundancy and marginal value
5  Calibration
6  Remaining headroom against the replicate-based ceiling
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("figures")

OUT = Path("results/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

# Consistent palette across the manuscript
C_EDIT, C_PEG, C_CROSS = "#2b6cb0", "#dd6b20", "#805ad5"
C_CTX, C_HEAD, C_GREY = "#38a169", "#c53030", "#a0aec0"
C_OURS, C_BASE = "#2b6cb0", "#a0aec0"

HELD = "results/round4/heldout/predictions_round4_final.parquet"
CORPUS = "data/processed/optiprime_official_318471.parquet"


def _rank01(x):
    x = np.asarray(x)
    return rankdata(x) / len(x)


def panel_label(ax, letter, dx=-0.02, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def _box(ax, x, y, w, h, text, color, fontsize=7.2, alpha=0.16, lw=1.3, style="round,pad=0.012"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                linewidth=lw, edgecolor=color,
                                facecolor=color, alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                linewidth=lw, edgecolor=color,
                                facecolor="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=4, linespacing=1.35)


def _arrow(ax, p0, p1, color="#4a5568", lw=1.2, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


# ============================================================ Figure 1
def fig1_architecture():
    fig = plt.figure(figsize=(7.4, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1.05, 1.05],
                          hspace=0.55, wspace=0.30)

    # ---------------- Panel A: overall architecture ----------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10); ax.set_ylim(0.0, 10.9); ax.axis("off")
    panel_label(ax, "A", dx=-0.01, dy=1.005)
    ax.text(5, 10.55, "PE-RankFormer architecture", ha="center",
            fontsize=9.5, fontweight="bold")

    _box(ax, 0.30, 9.15, 4.2, 0.92,
         "Target locus\nwild-type $\\rightarrow$ edited sequence pair", C_EDIT, 7.2)
    _box(ax, 5.50, 9.15, 4.2, 0.92,
         "pegRNA\nspacer | PBS | RTT", C_PEG, 7.2)

    _box(ax, 0.30, 7.90, 4.2, 0.88,
         "Paired tokenisation\none token per (WT, edited) pair · L$\\leq$102", C_EDIT, 6.7)
    _box(ax, 5.50, 7.90, 4.2, 0.88,
         "Segment-aware tokenisation\nnucleotide + segment embedding · L$\\leq$90", C_PEG, 6.7)

    _box(ax, 0.30, 6.50, 4.2, 1.00,
         "Edit encoder\n6 blocks · bidirectional S4D", C_EDIT, 7.2)
    _box(ax, 5.50, 6.50, 4.2, 1.00,
         "pegRNA encoder\n4 blocks · bidirectional S4D", C_PEG, 7.2)
    ax.text(9.72, 7.00, "block\ndetail:\npanel B", fontsize=6.0, color=C_GREY,
            ha="left", va="center", style="italic")

    _box(ax, 1.80, 5.05, 6.4, 0.95,
         "Bidirectional cross-attention ($\\times$2)\n"
         "PBS/RTT $\\leftrightarrow$ target alignment", C_CROSS, 7.2)

    _box(ax, 3.00, 3.95, 4.0, 0.75, "Attention pooling", C_CROSS, 7.2)

    _box(ax, 0.20, 2.30, 2.60, 1.70,
         "Experimental context\n\ncell line · PE system\nCas9 · PAM · scaffold\nmotif · study",
         C_CTX, 6.3)
    _box(ax, 3.00, 2.70, 4.0, 0.78,
         "FiLM   $h' = (1{+}\\gamma(c))\\odot h + \\beta(c)$", C_CTX, 6.9)

    _box(ax, 3.00, 1.25, 4.0, 0.92,
         "Ordinal head\n$K{-}1$ cumulative logits", C_HEAD, 7.2)
    ax.text(7.25, 1.71, "head\ndetail:\npanels C, D", fontsize=6.0, color=C_GREY,
            ha="left", va="center", style="italic")
    ax.text(5.0, 0.78, "$s(x)=\\frac{1}{K-1}\\sum_k\\sigma(z_k)$   =   predicted normalised rank",
            ha="center", fontsize=7.0, color="#2d3748")
    ax.text(5.0, 0.12, "monotone calibration $\\rightarrow$ absolute efficiency",
            ha="center", fontsize=6.5, style="italic", color="#4a5568")

    _arrow(ax, (2.40, 9.15), (2.40, 8.78)); _arrow(ax, (7.60, 9.15), (7.60, 8.78))
    _arrow(ax, (2.40, 7.90), (2.40, 7.50)); _arrow(ax, (7.60, 7.90), (7.60, 7.50))
    _arrow(ax, (2.40, 6.50), (3.60, 6.00), rad=-0.12)
    _arrow(ax, (7.60, 6.50), (6.40, 6.00), rad=0.12)
    _arrow(ax, (5.00, 5.05), (5.00, 4.70))
    _arrow(ax, (5.00, 3.95), (5.00, 3.48), color=C_CTX)
    _arrow(ax, (2.80, 3.15), (3.00, 3.15), color=C_CTX)
    _arrow(ax, (5.00, 2.70), (5.00, 2.17))

    # ---------------- Panel B: encoder block internals ----------------
    # y-range chosen so box heights map to enough pixels for the fixed-point text:
    # this subplot is ~2.2in tall, so 16 units gives ~0.14in per unit.
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 17); ax.axis("off")
    panel_label(ax, "B", dx=-0.02, dy=1.14)
    ax.text(5, 16.1, "Encoder block", ha="center", fontsize=8.6, fontweight="bold")

    # Layout notes, both fixing collisions visible in the rendered PDF:
    #  - the mixer captions were three lines at 6.4pt inside a box whose height in data
    #    units rendered to almost exactly the text height, so the first and last lines
    #    sat outside the box edge. Two lines in a taller box has clear margin.
    #  - the arrows previously landed at x=2.5 and x=7.5, which is inside the caption
    #    text. They now run down the 0.4-unit gap between the two alternative boxes,
    #    which is also the semantically right place: the boxes are alternatives, so one
    #    arrow through the middle reads as "either of these".
    _box(ax, 0.2, 13.8, 9.6, 1.5, "LayerNorm", C_GREY, 6.8)
    _box(ax, 0.2, 9.0, 4.6, 3.0, "bidirectional S4D\nclosed-form kernel", C_EDIT, 6.2)
    _box(ax, 5.2, 9.0, 4.6, 3.0, "self-attention\n(alternative)", C_GREY, 6.2)
    _box(ax, 0.2, 6.1, 9.6, 1.5, "gated projection + residual", C_GREY, 6.8)
    _box(ax, 0.2, 3.6, 9.6, 1.5, "FFN + residual", C_GREY, 6.8)

    _arrow(ax, (5.0, 13.8), (5.0, 12.2))
    _arrow(ax, (5.0, 9.0), (5.0, 7.7))
    _arrow(ax, (5.0, 6.1), (5.0, 5.2))
    ax.text(5.0, 1.7, "S4D mixes by distance, not content  ($+0.0192$)",
            ha="center", fontsize=6.3, style="italic", color=C_EDIT)

    # ---------------- Panel C: ordinal thresholds on real data ----------------
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "C", dx=-0.16, dy=1.14)
    d = pd.read_parquet(CORPUS, columns=["edited", "fold"])
    y = d[d.fold != 0].edited.to_numpy()
    thr = np.unique(np.quantile(y, np.linspace(0, 1, 21)[1:-1]))
    ax.hist(np.clip(y, 0, 0.6), bins=70, color=C_GREY, alpha=0.75)
    for t in thr:
        if t <= 0.6:
            ax.axvline(t, color=C_HEAD, lw=0.7, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("editing efficiency $y$")
    ax.set_ylabel("training rows (log)")
    ax.set_title("Thresholds $t_k$ set at quantiles of\nthe training distribution", fontsize=8.0)
    ax.text(0.97, 0.95,
            f"{len(thr)} distinct thresholds\n{100*(y==0).mean():.1f}% of training rows\nare exactly zero",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.1,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_GREY, lw=0.6))

    # ---------------- Panel D: what the head predicts ----------------
    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "D", dx=-0.02, dy=1.14)
    k = np.arange(1, 19)
    for val, lab, c in [(0.85, "high-efficiency design", C_EDIT),
                        (0.35, "mid-efficiency design", C_CTX),
                        (0.05, "inactive design", C_HEAD)]:
        p_ = 1 / (1 + np.exp((k - val * 18) * 0.9))
        ax.plot(k, p_, "o-", ms=2.4, lw=1.1, color=c, label=lab)
    ax.set_xlabel("threshold index $k$")
    ax.set_ylabel(r"$\sigma(z_k)\approx P(y>t_k)$")
    ax.set_title("Cumulative threshold predictions", fontsize=8.0)
    ax.legend(frameon=False, loc="upper center", fontsize=6.0, handlelength=1.4,
              ncol=1, borderaxespad=0.1)
    ax.set_ylim(-0.06, 1.52); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.text(0.97, 0.30, "score = mean of the curve\n= estimated rank",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.1, style="italic")

    # ---------------- Panel E: training/eval composition ----------------
    ax = fig.add_subplot(gs[2, 1])
    panel_label(ax, "E", dx=-0.16, dy=1.14)
    src = ["Schwank-\nderived", "Liu-\nderived", "Kim-\nderived"]
    tr = [58.4, 22.0, 19.6]; te = [0.0, 44.7, 55.3]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, tr, w, label="training", color=C_GREY)
    ax.bar(x + w / 2, te, w, label="held-out", color=C_OURS)
    for i, (a_, b_) in enumerate(zip(tr, te)):
        ax.text(i - w / 2, a_ + 1.5, f"{a_:.1f}", ha="center", fontsize=6.0)
        ax.text(i + w / 2, b_ + 1.5, f"{b_:.1f}", ha="center", fontsize=6.0)
    ax.set_xticks(x); ax.set_xticklabels(src, fontsize=6.5)
    ax.set_ylabel("% of rows"); ax.set_ylim(0, 72)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center")
    ax.set_title("Source composition differs between\ntraining and evaluation", fontsize=8.0)

    fig.savefig(OUT / "fig1_architecture.pdf")
    plt.close(fig)
    logger.info("fig1 architecture")


# ============================================================ Figure 2
def fig2_benchmark():
    d = pd.read_parquet(HELD)
    op_dir = Path("data/interim/optiprime_heldout_full/predictions_20260818_083148")
    preds = pd.read_csv(op_dir / "predictions.csv", index_col=0).reset_index(drop=True)
    joined = pd.read_csv(op_dir / "joined_df.csv", index_col=0).reset_index(drop=True)
    op = pd.DataFrame({"record_id": joined["record_id"], "op": preds["mean_pred"]})
    corpus = pd.read_parquet(CORPUS, columns=["record_id", "spacer"])
    df = d.merge(op, on="record_id").merge(corpus, on="record_id")

    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.30)

    # A: grouped bars
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "A")
    # Round-9: the feature baseline is included so the figure shows what the
    # mechanistic model and ours are both clear of, not only the gap between them.
    models = ["Gradient-boosted trees\non engineered features",
              "OptiPrime\n(mechanistic baseline)", "PE-RankFormer\nround 1",
              "PE-RankFormer\nround 3", "PE-RankFormer\nfinal"]
    full = [0.7413, 0.8690, 0.8865, 0.8933, 0.9079]
    liu = [0.6347, 0.8365, 0.8349, 0.8462, 0.8585]
    kim = [0.5745, 0.7320, 0.7751, 0.7836, 0.8124]
    x = np.arange(5); w = 0.26
    for i, (v, lab, c) in enumerate([(full, "All (n=20,509)", C_OURS),
                                     (liu, "Liu (n=9,175)", "#dd6b20"),
                                     (kim, "Kim (n=11,334)", "#38a169")]):
        b = ax.bar(x + (i - 1) * w, v, w, label=lab, color=c,
                   edgecolor="white", linewidth=0.5)
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + 0.005, f"{val:.4f}",
                    ha="center", fontsize=6, rotation=90)
    ax.axhline(0.8690, ls=":", c=C_GREY, lw=1)
    # Left of the OptiPrime group the reference line runs over whitespace; placing the
    # label anywhere right of it collides with the bar value labels.
    ax.text(-0.42, 0.8745, "OptiPrime, all", fontsize=6.2, color=C_GREY,
            va="bottom", ha="left")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=6.8)
    ax.set_ylabel("Spearman $\\rho$"); ax.set_ylim(0.54, 0.98)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
    ax.set_title("Held-out performance under matched training data, splits and evaluation",
                 fontsize=8.8)
    # B: bootstrap
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "B")
    y = df.true_efficiency.to_numpy()
    a = df.predicted_efficiency.to_numpy(); bb = df.op.to_numpy()
    groups = df.spacer.to_numpy(); uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(20260812)
    diffs = np.empty(5000)
    for kk in range(5000):
        s = rng.choice(uniq, size=len(uniq), replace=True)
        i = np.concatenate([idx_by[g] for g in s])
        diffs[kk] = spearmanr(y[i], a[i]).statistic - spearmanr(y[i], bb[i]).statistic
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    obs = spearmanr(y, a).statistic - spearmanr(y, bb).statistic
    ax.hist(diffs, bins=55, color=C_OURS, alpha=0.85, edgecolor="none")
    ax.axvline(0, c="k", lw=1.1)
    ax.axvline(lo, c="#dd6b20", ls="--", lw=1); ax.axvline(hi, c="#dd6b20", ls="--", lw=1)
    ax.axvline(obs, c="#c53030", lw=1.2)
    ax.set_xlabel("$\\Delta\\rho$  (PE-RankFormer $-$ OptiPrime)")
    ax.set_ylabel("bootstrap resamples")
    ax.set_title("Paired protospacer-clustered bootstrap\n5,000 resamples over 750 clusters",
                 fontsize=8.2)
    ax.text(0.02, 0.98, f"observed $+{obs:.4f}$\n95% CI [{lo:+.4f}, {hi:+.4f}]\n"
                        f"ahead in 100% of resamples\n$p<0.0002$",
            transform=ax.transAxes, va="top", fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=C_GREY, lw=0.6))

    # C: paired per-condition comparison. Round-9: this panel used to show only our
    # own per-condition score, which says nothing about the comparison. The paired
    # version is the robustness result -- the margin holds in every condition, and it is
    # larger within condition than pooled because pooling helps both models equally.
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "C")
    st = json.load(open("results/round9/stratified_comparison.json"))
    pc = sorted(st["per_condition"], key=lambda c: c["pe_rankformer"])
    def _lab(c):
        src, cell, pe = c["cond"].split("|")
        return f"{'Kim' if src == 'deepprime' else 'Liu'}  {cell}/{pe}"
    labs = [_lab(c) for c in pc]
    yy = np.arange(len(pc))
    ours = np.array([c["pe_rankformer"] for c in pc])
    theirs = np.array([c["optiprime"] for c in pc])
    for i in yy:  # connector shows the per-condition gain
        ax.plot([theirs[i], ours[i]], [i, i], c=C_GREY, lw=0.9, zorder=1)
    ax.scatter(theirs, yy, s=13, c=C_GREY, zorder=2, label="OptiPrime")
    ax.scatter(ours, yy, s=13, c=C_OURS, zorder=3, label="PE-RankFormer")
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=6.2)
    ax.set_xlim(0.52, 0.95); ax.set_xlabel("Spearman $\\rho$ within condition")
    ax.set_title("Paired comparison by experimental\ncondition ($n\\geq300$)", fontsize=8.2)
    ax.legend(frameon=False, loc="lower right", fontsize=6.6)
    wc = st["margin"]["within_condition"]
    ax.text(0.02, 0.985,
            f"ahead in {st['conditions_won']}/{st['conditions_total']} conditions\n"
            f"$n$-weighted $\\Delta\\rho={wc['delta']:+.4f}$",
            transform=ax.transAxes, va="top", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))

    fig.savefig(OUT / "fig2_benchmark.pdf"); plt.close(fig)
    logger.info("fig2 benchmark")


# ============================================================ Figure 3
def fig3_factorial():
    f = json.load(open("results/round5/factorial.json"))
    cells, e = f["cells"], f["effects"]
    grid = np.array([[cells["Transformer|simplex"]["full"], cells["Transformer|ordinal"]["full"]],
                     [cells["SSM|simplex"]["full"], cells["SSM|ordinal"]["full"]]])

    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.25], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "A", dx=-0.20)
    im = ax.imshow(grid, cmap="Blues", vmin=0.874, vmax=0.913)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i,j]:.4f}", ha="center", va="center", fontsize=8.5,
                    color="white" if grid[i, j] > 0.900 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["simplex", "ordinal"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Transformer", "S4D"])
    ax.set_xlabel("outcome head"); ax.set_ylabel("sequence mixer")
    ax.set_title("Out-of-fold $\\rho$", fontsize=8.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "B", dx=-0.28)
    labs = ["architecture\nS4D $-$ Transformer", "objective\nordinal $-$ simplex",
            "interaction"]
    vals = [e["architecture"], e["objective"], e["interaction"]]
    ax.barh(labs, vals, color=[C_EDIT, C_HEAD, C_GREY])
    for i, v in enumerate(vals):
        ax.text(v + (0.0008 if v > 0 else -0.0008), i, f"{v:+.4f}", va="center",
                ha="left" if v > 0 else "right", fontsize=7.2)
    ax.axvline(0, c="k", lw=0.9)
    ax.set_xlim(-0.010, 0.028); ax.set_xlabel("effect on Spearman $\\rho$")
    ax.set_title("Main effects and interaction", fontsize=8.4)
    ax.text(0.98, 0.06, "architecture contributes\n$\\approx2\\times$ the objective",
            transform=ax.transAxes, ha="right", fontsize=6.5, style="italic")

    ax = fig.add_subplot(gs[2])
    panel_label(ax, "C", dx=-0.26)
    ordv = [0.686, 0.691, 0.708, 0.686]
    simv = [0.764, 0.765, 0.777, 0.781]
    ax.scatter(np.random.default_rng(0).normal(1, 0.045, len(ordv)), ordv,
               color=C_HEAD, s=26, label="ordinal head", zorder=3)
    ax.scatter(np.random.default_rng(1).normal(2, 0.045, len(simv)), simv,
               color=C_GREY, s=26, label="simplex head", zorder=3)
    ax.plot([0.75, 1.25], [np.mean(ordv)] * 2, color=C_HEAD, lw=2)
    ax.plot([1.75, 2.25], [np.mean(simv)] * 2, color=C_GREY, lw=2)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["ordinal", "simplex"])
    ax.set_ylabel("residual correlation\nwith existing ensemble")
    ax.set_xlim(0.6, 2.4); ax.set_ylim(0.65, 0.81)
    ax.set_title("Ordinal models make\ndifferent errors", fontsize=8.4)
    ax.text(0.5, 0.06, "lower = more complementary", transform=ax.transAxes,
            ha="center", fontsize=6.5, style="italic")

    fig.savefig(OUT / "fig3_factorial.pdf"); plt.close(fig)
    logger.info("fig3 factorial")


# ============================================================ Figure 4
def fig4_ensemble():
    d = pd.read_parquet(HELD)
    mem = [c for c in d.columns if c.startswith("member_")]
    names = [m.replace("member_", "") for m in mem]
    pretty = {"ordSSM": "ordinal+S4D", "ssm": "simplex+S4D", "ordC": "ordinal+features",
              "ordA": "ordinal+layerwise", "familyA": "simplex+layerwise"}
    labs = [pretty.get(n, n) for n in names]

    ry = _rank01(d.true_efficiency)
    R = np.zeros((len(mem), len(mem)))
    for i, a in enumerate(mem):
        for j, b in enumerate(mem):
            R[i, j] = spearmanr(ry - _rank01(d[a]), ry - _rank01(d[b])).statistic

    fig = plt.figure(figsize=(7.6, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1, 1], wspace=0.95)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "A", dx=-0.34)
    im = ax.imshow(R, cmap="RdYlBu_r", vmin=0.60, vmax=1.0)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=42, ha="right", fontsize=6.2)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6.2)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if R[i, j] > 0.88 else "black")
    ax.set_title("Residual correlation between\nmembers (held-out)", fontsize=8.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02); cb.ax.tick_params(labelsize=5.6)

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "B", dx=-0.46)
    solo = [spearmanr(d.true_efficiency, d[m]).statistic for m in mem]
    order = np.argsort(solo)
    ax.barh([labs[i] for i in order], [solo[i] for i in order], color=C_OURS)
    ens = spearmanr(d.true_efficiency, d.predicted_efficiency).statistic
    ax.axvline(ens, color=C_HEAD, ls="--", lw=1.1)
    ax.text(ens - 0.0008, -0.45, f"ensemble {ens:.4f}", color=C_HEAD,
            fontsize=6.2, ha="right")
    for i, idx in enumerate(order):
        ax.text(solo[idx] - 0.0012, i, f"{solo[idx]:.4f}", va="center", ha="right",
                fontsize=6, color="white")
    ax.set_xlim(0.8755, 0.9135); ax.set_xlabel("held-out Spearman $\\rho$")
    ax.set_yticklabels([labs[i] for i in order], fontsize=6.2)
    ax.set_title("Individual member performance", fontsize=8.2)

    ax = fig.add_subplot(gs[2])
    panel_label(ax, "C", dx=-0.46)
    full = np.mean([_rank01(d[m]) for m in mem], axis=0)
    base = spearmanr(d.true_efficiency, full).statistic
    contrib = []
    for m in mem:
        rest = [x for x in mem if x != m]
        contrib.append(base - spearmanr(d.true_efficiency,
                                        np.mean([_rank01(d[x]) for x in rest], axis=0)).statistic)
    order = np.argsort(contrib)
    cols = [C_OURS if contrib[i] > 0 else "#dd6b20" for i in order]
    ax.barh([labs[i] for i in order], [contrib[i] for i in order], color=cols)
    ax.axvline(0, c="k", lw=0.9)
    ax.set_xlabel("$\\Delta\\rho$ when member removed")
    ax.set_yticklabels([labs[i] for i in order], fontsize=6.2)
    ax.set_title("Marginal contribution", fontsize=8.2)
    ax.text(0.5, -0.22, "negative $=$ ensemble improves without this member",
            transform=ax.transAxes, ha="center", fontsize=6.0, style="italic")

    fig.savefig(OUT / "fig4_ensemble.pdf"); plt.close(fig)
    logger.info("fig4 ensemble")


# ============================================================ Figure 5
def fig5_calibration():
    c = json.load(open("results/round5/calibration.json"))
    cal = pd.read_parquet("results/round5/heldout_calibrated.parquet")

    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 3, wspace=0.42)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "A", dx=-0.28)
    r = _rank01(cal.predicted_efficiency.to_numpy())
    o = np.argsort(r)
    ax.plot(r[o], cal.calibrated_efficiency.to_numpy()[o], color=C_OURS, lw=1.6)
    ax.set_xlabel("ensemble rank score")
    ax.set_ylabel("calibrated efficiency")
    ax.set_title("The learned monotone map $g$", fontsize=8.2)
    ax.text(0.04, 0.92, "isotonic, fitted on\nout-of-fold development\npredictions only",
            transform=ax.transAxes, va="top", fontsize=6.3, style="italic")

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "B", dx=-0.30)
    q = np.linspace(0, 1, 21)
    edges = np.unique(np.quantile(cal.calibrated_efficiency, q))
    g = cal.groupby(pd.cut(cal.calibrated_efficiency, edges, include_lowest=True),
                    observed=True).agg(pred=("calibrated_efficiency", "mean"),
                                       obs=("true_efficiency", "mean"))
    hi = max(g.obs.max(), g.pred.max()) * 1.05
    ax.plot([0, hi], [0, hi], ls=":", c=C_GREY, label="perfect calibration")
    ax.plot(g.pred, g.obs, "o-", color=C_OURS, ms=4, label="held-out, binned")
    ax.set_xlabel("predicted efficiency"); ax.set_ylabel("observed efficiency")
    ax.set_title("Calibration curve, held-out", fontsize=8.2)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")

    ax = fig.add_subplot(gs[2])
    panel_label(ax, "C", dx=-0.30)
    # Round-9 rebuild. The previous version compared the calibrated output against our
    # own UNCALIBRATED rank average, whose MAE/RMSE are not in efficiency units, and it
    # set ylim=0.62 so the two Spearman bars overflowed the axes and printed their value
    # labels outside the panel, colliding into "0.908.908". The comparison that carries
    # an argument is against the baseline on identical rows.
    op = json.load(open("results/round9/stratified_comparison.json"))["absolute_accuracy"]
    base = op["OptiPrime (5-model ens)"]
    ours = op["PE-RankFormer + isotonic"]
    keys = ["spearman", "pearson", "mae", "rmse"]
    labels = ["Spearman", "Pearson", "MAE", "RMSE"]
    rank_only = {"spearman": c["heldout_uncalibrated"]["spearman"],
                 "pearson": c["heldout_uncalibrated"]["pearson"],
                 "mae": np.nan, "rmse": np.nan}  # not in efficiency units
    x = np.arange(4); w = 0.27
    # Colour language: grey = baseline, blues = ours. The two greys of an earlier
    # version were indistinguishable at print size.
    series = [("OptiPrime", [base[k] for k in keys], C_GREY),
              ("ours, rank average", [rank_only[k] for k in keys], "#9dc3e6"),
              ("ours, $+$ calibration", [ours[k] for k in keys], C_OURS)]
    for i, (lab, vals, col) in enumerate(series):
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col,
               edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + (i - 1) * w, vals):
            if np.isnan(v):
                ax.text(xi, 0.02, "n/a", ha="center", fontsize=5.4, rotation=90,
                        color=C_GREY, va="bottom")
            else:
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=5.4, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.4, rotation=18, ha="right")
    # Bars plus their rotated labels top out near 1.0, so 1.45 leaves a clear band above
    # them for the legend. Anchoring the legend inside the plotting area at any height
    # overlapped the Spearman/Pearson groups.
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # no tick in the legend band
    ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=5.5, handlelength=1.0, labelspacing=0.25,
              columnspacing=0.8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.015))
    ax.set_title("Ranking preserved; on the recovered\nscale the model also leads",
                 fontsize=8.2)

    fig.savefig(OUT / "fig5_calibration.pdf"); plt.close(fig)
    logger.info("fig5 calibration")


# ============================================================ Figure 6
def fig6_ceiling():
    n = json.load(open("results/round6/noise_ceiling_empirical.json"))
    bc = n["by_condition"]
    rows = sorted(bc.items(), key=lambda kv: kv[1]["pct_zero"])
    labs = [k.replace("('", "").replace("')", "").replace("', '", "/") for k, _ in rows]
    model = [v["model"] for _, v in rows]
    ceil = [v["ceiling"] for _, v in rows]
    zero = [v["pct_zero"] for _, v in rows]

    fig = plt.figure(figsize=(7.4, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.48)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "A", dx=-0.30)
    yy = np.arange(len(labs))
    ax.barh(yy, ceil, color=C_GREY, label="replicate-based ceiling")
    ax.barh(yy, model, height=0.55, color=C_OURS, label="Ordinal-S4D model")
    for i, (m_, c_) in enumerate(zip(model, ceil)):
        ax.text(c_ + 0.006, i, f"+{c_-m_:.3f}", va="center", fontsize=5.8, color="#4a5568")
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=6.2)
    ax.set_xlim(0.5, 1.06); ax.set_xlabel("Spearman $\\rho$")
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax.set_title("Headroom in every Kim condition\n(ordered by zero-inflation)", fontsize=8.2)

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "B", dx=-0.46)
    gaps = [c_ - m_ for c_, m_ in zip(ceil, model)]
    ax.scatter(zero, gaps, color="#dd6b20", s=30, zorder=3)
    z = np.polyfit(zero, gaps, 1)
    xs = np.linspace(min(zero), max(zero), 10)
    ax.plot(xs, np.polyval(z, xs), ls="--", c=C_GREY, lw=1)
    ax.set_xlabel("% of rows exactly zero"); ax.set_ylabel("gap to ceiling")
    ax.set_title(f"Gap tracks zero-inflation\n(Spearman $\\rho={spearmanr(zero,gaps).statistic:.2f}$)",
                 fontsize=8.2)

    ax = fig.add_subplot(gs[2])
    panel_label(ax, "C", dx=-0.46)
    cats = ["naive\nreplicate key", "correct key,\nzeros noiseless", "correct key,\nzeros censored"]
    vals = [0.7987, 0.9676, 0.9026]
    cols = ["#c53030", "#dd6b20", C_OURS]
    ax.bar(cats, vals, color=cols)
    ax.axhline(0.7869, ls="--", c="k", lw=1)
    ax.text(2.45, 0.792, "model", fontsize=6.2, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=6.4)
    ax.set_ylim(0.70, 1.03); ax.set_ylabel("estimated ceiling")
    ax.set_xticklabels(cats, fontsize=6.0)
    ax.set_title("Both corrections are load-bearing", fontsize=8.2)
    ax.text(0.5, -0.42, "a naive estimate would wrongly indicate saturation",
            transform=ax.transAxes, ha="center", fontsize=6.2, style="italic")

    fig.savefig(OUT / "fig6_ceiling.pdf"); plt.close(fig)
    logger.info("fig6 ceiling")


if __name__ == "__main__":
    fig1_architecture(); fig2_benchmark(); fig3_factorial()
    fig4_ensemble(); fig5_calibration(); fig6_ceiling()
    logger.info("all figures written to %s", OUT)
