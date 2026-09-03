"""Does this problem need the full epoch budget? A reproducible survey.

Round-9: the manuscript justified `patience = max_epochs` (no early stopping) from a
survey of 25 official-fold runs. The population was not defined precisely enough to
regenerate, so this replaces it with an explicitly specified one: every recorded run
that used the official cross-validation folds (not the matched development folds) and
was given the full 30-epoch budget. For each, it reports the epoch of peak validation
Spearman, and simulates what a patience-5 early-stopping rule would have selected
instead.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("earlystop")
OUT = Path("results/round9/early_stopping_survey.json")
MIN_BUDGET = 25
PATIENCE = 5


def simulate_patience(hist: pd.Series, patience: int) -> int:
    """Index the run would have stopped at under `patience` epochs without improvement."""
    best_i, best_v, since = 0, -np.inf, 0
    for i, v in enumerate(hist):
        if v > best_v:
            best_i, best_v, since = i, v, 0
        else:
            since += 1
            if since >= patience:
                break
    return best_i


def main() -> None:
    rows = []
    for f in sorted(glob.glob("results/runs/*/run_info.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("dev_folds_file"):            # official folds only
            continue
        if d.get("total_epochs_run", 0) < MIN_BUDGET:
            continue
        h = Path(f).parent / "training_history.csv"
        if not h.exists():
            continue
        hist = pd.read_csv(h)
        col = next((c for c in hist.columns if "spearman" in c.lower()
                    and "train" not in c.lower()), None)
        if col is None:
            continue
        v = hist[col].to_numpy()
        stop_i = simulate_patience(hist[col], PATIENCE)
        rows.append(dict(run=d["run_name"], best_epoch=int(np.argmax(v)),
                         n_epochs=len(v), best=float(v.max()),
                         patience5=float(v[stop_i]), patience5_epoch=int(stop_i),
                         cost=float(v.max() - v[stop_i])))
    t = pd.DataFrame(rows)
    be = t.best_epoch.to_numpy()
    final5 = be >= (t.n_epochs.to_numpy() - 5)
    summary = dict(
        n_runs=int(len(t)),
        best_epoch_median=float(np.median(be)),
        best_epoch_max=int(be.max()),
        best_epoch_min=int(be.min()),
        frac_peaking_in_final_five=float(final5.mean()),
        patience5_cost_mean=float(t.cost.mean()),
        patience5_cost_max=float(t.cost.max()),
        frac_patience5_costs_over_0005=float((t.cost > 0.005).mean()),
        patience=PATIENCE, min_budget=MIN_BUDGET,
    )
    for k, val in summary.items():
        logger.info("%-34s %s", k, val)
    logger.info("")
    logger.info("worst five runs under patience-5 early stopping:")
    for r in t.nlargest(5, "cost").itertuples():
        logger.info("  %-28s best %.4f @ep%-3d  patience5 %.4f @ep%-3d  cost %+.4f",
                    r.run, r.best, r.best_epoch, r.patience5, r.patience5_epoch, -r.cost)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
