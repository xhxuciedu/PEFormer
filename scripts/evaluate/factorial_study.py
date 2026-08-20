"""Objective x architecture factorial study (round-5 spec §11).

All four cells already exist as round-4 out-of-fold predictions, so this is computed
from stored artifacts rather than retrained:

               | simplex head        | ordinal head
  Transformer  | round1_baseline_oof | r4p2_ordB_oof
  SSM          | r4p2_ssm_oof        | r4p2_ordSSM_oof

Important comparability caveat, checked rather than assumed: the four runs are not a
purpose-built factorial. round1_baseline is the frozen round-1 model, while the other
three are round-4 Phase-2 runs. They share the corpus, the official 5-fold split, the
OOF scoring rule, d_model/depth, and the optimiser recipe, but round1_baseline was
trained at an earlier commit. The main effects below are therefore *estimates* from
observational cells, and the interaction term is the quantity most exposed to that
mismatch -- it is reported with that caveat attached rather than presented as a clean
2x2 result.

Reports main effects, the interaction, and the diversity structure (prediction and
residual correlations) that determines ensemble value.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("factorial")

DEV_FOLDS = ["round3_dev_fold_0", "round3_dev_fold_1", "round3_dev_fold_2"]
PRED = Path("results/round3/dev_recalibration")

CELLS = {
    ("Transformer", "simplex"): "round1_baseline_oof",
    ("Transformer", "ordinal"): "r4p2_ordB_oof",
    ("SSM", "simplex"): "r4p2_ssm_oof",
    ("SSM", "ordinal"): "r4p2_ordSSM_oof",
}


def _rank01(x) -> np.ndarray:
    x = np.asarray(x)
    return rankdata(x) / len(x)


def load(fold: str, names: list[str]) -> pd.DataFrame:
    base = None
    for n in names:
        d = pd.read_parquet(PRED / f"predictions_{n}_{fold}.parquet")
        keep = (d[["record_id", "true_efficiency", "source_study", "predicted_efficiency"]]
                if base is None else d[["record_id", "predicted_efficiency"]])
        keep = keep.rename(columns={"predicted_efficiency": n})
        base = keep if base is None else base.merge(keep, on="record_id")
    return base


def main() -> None:
    names = list(CELLS.values())
    folds = {f: load(f, names) for f in DEV_FOLDS}

    out: dict = {"cells": {}, "note": "observational cells, not a purpose-built factorial"}
    logger.info("%-12s %-9s %8s %8s %8s", "arch", "objective", "full", "Liu", "Kim")
    scores: dict[tuple[str, str], float] = {}
    for (arch, obj), name in CELLS.items():
        full, liu, kim = [], [], []
        for f in DEV_FOLDS:
            d = folds[f]
            full.append(spearmanr(d.true_efficiency, d[name]).statistic)
            for study, acc in (("hsu2026", liu), ("deepprime", kim)):
                s = d[d.source_study == study]
                acc.append(spearmanr(s.true_efficiency, s[name]).statistic)
        scores[(arch, obj)] = float(np.mean(full))
        logger.info("%-12s %-9s %8.4f %8.4f %8.4f", arch, obj, np.mean(full), np.mean(liu), np.mean(kim))
        out["cells"][f"{arch}|{obj}"] = {
            "model": name, "full": float(np.mean(full)),
            "liu": float(np.mean(liu)), "kim": float(np.mean(kim)),
        }

    ts, to = scores[("Transformer", "simplex")], scores[("Transformer", "ordinal")]
    ss, so = scores[("SSM", "simplex")], scores[("SSM", "ordinal")]
    obj_effect = 0.5 * ((to - ts) + (so - ss))
    arch_effect = 0.5 * ((ss - ts) + (so - to))
    interaction = (so - ss) - (to - ts)

    logger.info("")
    logger.info("main effect of OBJECTIVE (ordinal - simplex), averaged over arch : %+.4f", obj_effect)
    logger.info("main effect of ARCHITECTURE (SSM - Transformer), averaged over obj: %+.4f", arch_effect)
    logger.info("INTERACTION (ordinal gain on SSM) - (ordinal gain on Transformer) : %+.4f", interaction)
    logger.info("  ordinal gain on Transformer: %+.4f", to - ts)
    logger.info("  ordinal gain on SSM        : %+.4f", so - ss)
    out["effects"] = {"objective": float(obj_effect), "architecture": float(arch_effect),
                      "interaction": float(interaction),
                      "ordinal_gain_transformer": float(to - ts),
                      "ordinal_gain_ssm": float(so - ss)}

    # Diversity structure: which pairs are most complementary?
    logger.info("")
    logger.info("--- pairwise prediction / residual rank correlation (fold-averaged) ---")
    out["pairs"] = {}
    for a, b in itertools.combinations(names, 2):
        pc, rc = [], []
        for f in DEV_FOLDS:
            d = folds[f]
            ra, rb = _rank01(d[a]), _rank01(d[b])
            ry = _rank01(d.true_efficiency)
            pc.append(spearmanr(ra, rb).statistic)
            rc.append(spearmanr(ry - ra, ry - rb).statistic)
        la = [k for k, v in CELLS.items() if v == a][0]
        lb = [k for k, v in CELLS.items() if v == b][0]
        logger.info("  %-22s vs %-22s pred=%.4f resid=%.4f",
                    "/".join(la), "/".join(lb), np.mean(pc), np.mean(rc))
        out["pairs"][f"{a}|{b}"] = {"pred_corr": float(np.mean(pc)), "resid_corr": float(np.mean(rc))}

    Path("results/round5").mkdir(parents=True, exist_ok=True)
    Path("results/round5/factorial.json").write_text(json.dumps(out, indent=2))
    logger.info("wrote results/round5/factorial.json")


if __name__ == "__main__":
    main()
