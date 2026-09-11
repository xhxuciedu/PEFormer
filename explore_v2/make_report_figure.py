"""Figure 1 for reports/explore_v2_report.tex, from cached panel predictions.

(a) Achieved efficiency of the top pick as a function of how many distinct designs the user
    is choosing between. The gap between the two models is invisible at depth 2 and opens as
    the choice gets harder, which is the shape that matters for deployment.
(b) The geometry of the nominated design against the best-measured design. OptiPrime's RTT
    distribution sits left of zero and PE-RankFormer's right of it -- the models fail in
    opposite directions, not by different amounts.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/make_report_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import endpoints as E  # noqa: E402
from canon import CANON_VERSION  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures_explore_v2"
CLR = {"rand": "#9e9e9e", "op": "#0b6fa4", "ours": "#c1440e", "oracle": "#2e7d32"}
NAME = {"rand": "Random choice", "op": "OptiPrime", "ours": "PE-RankFormer",
        "oracle": "Perfect chooser"}


def main() -> None:
    p = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    p = p.merge(pd.read_parquet(C.CACHE / "optiprime_panel_predictions.parquet"),
                on="record_id", validate="1:1")
    p = p.merge(pd.read_parquet(
        C.CACHE / f"panel_predictions_ours_v{CANON_VERSION}.parquet")[["record_id", "ours"]],
        on="record_id", validate="1:1")
    p["pbs_len"], p["rtt_len"] = p.pbs_dna.str.len(), p.rtt_dna.str.len()

    # one row per (decision group, distinct design); the panel already carries `rtt`, so the
    # nominated-design geometry gets its own column names
    cand = (p.groupby(["edit_key", "design_key"], observed=True, as_index=False)
              .agg(edited_frac=("edited_frac", "mean"), protospacer=("protospacer", "first"),
                   pbs_n=("pbs_len", "first"), rtt_n=("rtt_len", "first"),
                   op=("op", "mean"), ours=("ours", "mean")))
    nd = cand.groupby("edit_key", observed=True).design_key.transform("nunique")
    cand = cand[nd >= 2].sort_values("edit_key", kind="stable")
    t = E.score_population(cand, ["op", "ours"], ks=(1,))
    t = t[t.informative]

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))

    # ---- (a) achieved efficiency against candidate depth ----------------------------
    bins = [(2, 2), (3, 3), (4, 4), (5, 7), (8, 99)]
    labels, series = [], {k: [] for k in ("rand", "op", "ours", "oracle")}
    ns = []
    for lo, hi in bins:
        s = t[(t.depth >= lo) & (t.depth <= hi)]
        labels.append(f"{lo}" if lo == hi else (f"{lo}\u2013{hi}" if hi < 99
                                                else f"$\\geq${lo}"))
        ns.append(len(s))
        for k in series:
            series[k].append(float(s[f"{k}_at_1"].mean() if k != "oracle"
                                   else s.oracle.mean()))
    x = np.arange(len(bins))
    for k in ("oracle", "op", "ours", "rand"):
        ax[0].plot(x, series[k], marker="o", color=CLR[k], label=NAME[k], lw=1.8,
                   ms=5, ls="--" if k in ("oracle", "rand") else "-")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{a}\n$n$={b:,}" for a, b in zip(labels, ns)], fontsize=8)
    ax[0].set_xlabel("Distinct designs to choose between", fontsize=9)
    ax[0].set_ylabel("Achieved efficiency of the top pick", fontsize=9)
    ax[0].set_title("a  The choice gets harder with depth", fontsize=10, loc="left")
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    ax[0].tick_params(labelsize=8)
    ax[0].grid(alpha=0.25, lw=0.5)

    # ---- (b) RTT length of the nominated design, against the best design ------------
    d = {"op": [], "ours": []}
    for _, s in cand.groupby("edit_key", observed=True, sort=False):
        y = s.edited_frac.to_numpy()
        if y.max() == y.min():
            continue
        rb = s.rtt_n.iloc[int(np.argmax(y))]
        for m in d:
            d[m].append(s.rtt_n.iloc[int(np.argmax(s[m].to_numpy()))] - rb)
    # The zero bin holds most of the mass and hides the asymmetry, so report the direction
    # of the error instead of its full distribution.
    cats = ("shorter", "same length", "longer")
    w, xs = 0.36, np.arange(3)
    for i, m in enumerate(("op", "ours")):
        v = np.array(d[m])
        frac = [float((v < 0).mean()), float((v == 0).mean()), float((v > 0).mean())]
        bars = ax[1].bar(xs + (i - 0.5) * w, frac, w, color=CLR[m], alpha=0.85,
                         label=f"{NAME[m]} (mean {v.mean():+.2f} nt)")
        for b, f in zip(bars, frac):
            ax[1].text(b.get_x() + b.get_width() / 2, f + 0.008, f"{f:.0%}",
                       ha="center", fontsize=7.5, color=CLR[m])
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"RTT {c}\nthan best" if c != "same length" else "RTT same\nas best"
                           for c in cats], fontsize=8.5)
    ax[1].set_ylim(0, 0.72)
    ax[1].set_ylabel("Fraction of decision groups", fontsize=9)
    ax[1].set_title("b  The two models err in opposite directions", fontsize=10, loc="left")
    ax[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    ax[1].tick_params(labelsize=8)
    ax[1].grid(alpha=0.25, lw=0.5, axis="y")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_decision.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "fig_decision.pdf")
    print("depth strata:", dict(zip(labels, ns)))
    print("mean RTT bias:", {m: round(float(np.mean(v)), 3) for m, v in d.items()})


if __name__ == "__main__":
    main()
