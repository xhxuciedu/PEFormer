"""Task 1.5 — calibration against a floor, and per-decile behaviour.

MAE = 0.0478 is uninterpretable on its own. On a target whose mean is a few per cent,
a constant predictor is already close in absolute error, so the question is how much of
that 0.0478 is skill and how much is the scale of the target. This reports the trivial
floors, the comparator, and per-decile calibration with attention to the high-efficiency
tail, where isotonic regression has the least data and where absolute predictions
actually matter for design.

Descriptive re-analysis of frozen predictions. The isotonic map was fitted on
out-of-fold development predictions only; nothing here refits it.

Usage: python revision/task_1_5_calibration_baselines.py [--seed 20260903]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def err(pred: np.ndarray, y: np.ndarray) -> dict:
    d = pred - y
    return {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d * d).mean())),
            "bias": float(d.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT_DEFAULT)
    args = ap.parse_args()

    m = C.load_heldout()
    y = m.y.to_numpy()
    # Floors are fitted on TRAINING rows only, as any honest baseline must be.
    tr = pd.read_parquet(C.CORPUS, columns=["edited", "fold"])
    tr = tr[tr.fold != 0].edited.to_numpy()
    train_median, train_mean = float(np.median(tr)), float(tr.mean())

    preds = {
        "constant at training median": np.full(len(m), train_median),
        "constant at training mean": np.full(len(m), train_mean),
        "constant at held-out mean (oracle floor)": np.full(len(m), float(y.mean())),
        "OptiPrime": m.op.to_numpy(),
        "PE-RankFormer + isotonic": m.calibrated_efficiency.to_numpy(),
    }
    result = {"provenance": C.provenance([C.CORPUS, C.H2H, C.CAL], args.seed),
              "training_median": train_median, "training_mean": train_mean,
              "heldout_mean": float(y.mean()), "absolute_error": {}}
    for name, p in preds.items():
        e = err(p, y)
        e["bootstrap_mae"] = C.cluster_bootstrap(
            m.assign(_p=p), lambda d: float(np.abs(d._p - d.y).mean()),
            args.seed, args.n_boot)
        result["absolute_error"][name] = e

    # paired MAE difference vs OptiPrime
    result["mae_delta_vs_optiprime"] = C.cluster_bootstrap(
        m, lambda d: float(np.abs(d.calibrated_efficiency - d.y).mean()
                           - np.abs(d.op - d.y).mean()),
        args.seed, args.n_boot)

    # ---------------- per-decile calibration ----------------
    m2 = m.copy()
    m2["decile"] = pd.qcut(m2.ours.rank(method="first"), 10, labels=False)
    dec = []
    for d, s in m2.groupby("decile"):
        dec.append({"decile": int(d) + 1, "n": int(len(s)),
                    "mean_predicted": float(s.calibrated_efficiency.mean()),
                    "mean_observed": float(s.y.mean()),
                    "median_observed": float(s.y.median()),
                    "gap": float(s.calibrated_efficiency.mean() - s.y.mean()),
                    "mae": float(np.abs(s.calibrated_efficiency - s.y).mean()),
                    "optiprime_mean_predicted": float(s.op.mean()),
                    "optiprime_gap": float(s.op.mean() - s.y.mean())})
    result["per_decile"] = dec

    # the tail specifically: rows the model puts in its top 5% and top 1%
    for frac, lab in ((0.05, "top_5pct"), (0.01, "top_1pct")):
        k = int(len(m) * frac)
        idx = np.argsort(-m.ours.to_numpy(), kind="stable")[:k]
        s = m.iloc[idx]
        result[lab] = {"n": int(k),
                       "mean_predicted": float(s.calibrated_efficiency.mean()),
                       "mean_observed": float(s.y.mean()),
                       "gap": float(s.calibrated_efficiency.mean() - s.y.mean()),
                       "mae": float(np.abs(s.calibrated_efficiency - s.y).mean()),
                       "optiprime_gap": float(s.op.mean() - s.y.mean())}

    ae = result["absolute_error"]
    rows = [f"| {n} | {v['mae']:.4f} | {v['rmse']:.4f} | {v['bias']:+.4f} | "
            f"[{v['bootstrap_mae']['ci95'][0]:.4f}, {v['bootstrap_mae']['ci95'][1]:.4f}] |"
            for n, v in ae.items()]
    drows = [f"| {d['decile']} | {d['n']:,} | {d['mean_predicted']:.4f} | "
             f"{d['mean_observed']:.4f} | {d['gap']:+.4f} | {d['optiprime_gap']:+.4f} |"
             for d in dec]
    b = result["mae_delta_vs_optiprime"]
    t5, t1 = result["top_5pct"], result["top_1pct"]

    md = f"""# Task 1.5 — calibration against a floor

Protospacer-clustered bootstrap, {args.n_boot} resamples, seed {args.seed}.
Commit `{result['provenance']['git_commit']}`. Constant baselines are fitted on the
297,962 training rows (median {train_median:.4f}, mean {train_mean:.4f}); the held-out
mean {result['heldout_mean']:.4f} is shown as an oracle floor no honest model could use.

| predictor | MAE | RMSE | bias | 95% CI on MAE |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

MAE against OptiPrime: {b['observed']:+.4f}
(95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}], p = {b['two_sided_p']:.4f}); negative
favours PE-RankFormer.

## Per-decile calibration, by predicted rank

| decile | n | mean predicted | mean observed | gap (ours) | gap (OptiPrime) |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(drows)}

## The high-efficiency tail, where absolute predictions matter for design

| slice | n | mean predicted | mean observed | gap | MAE | OptiPrime gap |
|---|---:|---:|---:|---:|---:|---:|
| model's top 5% | {t5['n']:,} | {t5['mean_predicted']:.4f} | {t5['mean_observed']:.4f} | {t5['gap']:+.4f} | {t5['mae']:.4f} | {t5['optiprime_gap']:+.4f} |
| model's top 1% | {t1['n']:,} | {t1['mean_predicted']:.4f} | {t1['mean_observed']:.4f} | {t1['gap']:+.4f} | {t1['mae']:.4f} | {t1['optiprime_gap']:+.4f} |
"""
    C.write_outputs("task_1_5_calibration_baselines", result, md)
    for n, v in ae.items():
        print(f"{n:44s} MAE {v['mae']:.4f}  RMSE {v['rmse']:.4f}  bias {v['bias']:+.4f}")
    print(f"\ntop 5%: predicted {t5['mean_predicted']:.4f} vs observed "
          f"{t5['mean_observed']:.4f} (gap {t5['gap']:+.4f})")
    print(f"top 1%: predicted {t1['mean_predicted']:.4f} vs observed "
          f"{t1['mean_observed']:.4f} (gap {t1['gap']:+.4f})")


if __name__ == "__main__":
    main()
