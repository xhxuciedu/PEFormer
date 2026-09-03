"""Gradient-boosted-tree baseline on engineered features, under the identical protocol.

Round-9 addition. The manuscript compared against exactly one external model, which
leaves open the question a reviewer asks first: does a 25M-parameter sequence model earn
its complexity over a well-tuned tree ensemble on hand-engineered features? That is also
the closest fair stand-in for the feature-based predictors we could not run end-to-end
(PRIDICT2.0, DeepPrime-FT both depend on reproducing a separate pretrained on-target
cutting model exactly), since it uses the same *kind* of information they do: PBS/RTT
lengths and GC, melting temperatures, ViennaRNA folding energies, edit type and offset,
and a RuleSet3 on-target score.

Protocol is matched to both PE-RankFormer and OptiPrime: five models, model k trained
with official fold k held out, held-out predictions from the 5-model ensemble average.
Two target parameterisations are tried and the better reported, so the baseline is given
its best shot rather than a strawman: raw efficiency, and the within-training-fold rank
transform (metric-matched, the tree analogue of our ordinal head).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gbm")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
FEATS = Path("data/processed/family_c_features.parquet")
H2H = Path("results/heldout_full_head_to_head.parquet")
OUT = Path("results/round9/gbm_baseline.json")
PRED_OUT = Path("results/round9/gbm_heldout_predictions.parquet")

CTX = ["cell_type", "pe_type", "cas9_type", "cas9_pam", "scaffold_name", "motif",
       "source_study"]
BASE = dict(early_stopping=False, random_state=20260903)
# The baseline is given a sweep so the comparison is against a tuned tree ensemble, not
# a strawman. Selection is on pooled held-out Spearman, which is generous to the
# baseline -- it is the only model here allowed to pick its hyperparameters on the test
# set, and it still loses.
GRID = [
    dict(max_iter=400,  learning_rate=0.06, max_leaf_nodes=31,  min_samples_leaf=40, l2_regularization=1.0),
    dict(max_iter=600,  learning_rate=0.06, max_leaf_nodes=63,  min_samples_leaf=40, l2_regularization=1.0),
    dict(max_iter=1200, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=20, l2_regularization=1.0),
    dict(max_iter=2000, learning_rate=0.02, max_leaf_nodes=255, min_samples_leaf=10, l2_regularization=0.1),
]


def build() -> tuple[pd.DataFrame, list[str], list[int]]:
    corpus = pd.read_parquet(
        CORPUS, columns=["record_id", "edited", "fold", "source_study", "cell_type",
                         "pe_type", "cas9_type", "cas9_pam", "scaffold_name", "motif"])
    feats = pd.read_parquet(FEATS)
    df = corpus.merge(feats, on="record_id", validate="1:1")
    feat_cols = [c for c in feats.columns if c != "record_id"]
    # Encode for the model but keep the human-readable labels for the breakdown.
    df["source_label"] = df.source_study.astype(str)
    df["cond_label"] = (df.source_study.astype(str) + "|" + df.cell_type.astype(str)
                        + "|" + df.pe_type.astype(str))
    for c in CTX:
        df[c] = df[c].astype("category").cat.codes.astype(np.int32)
    cols = feat_cols + CTX
    cat_idx = [cols.index(c) for c in CTX]
    logger.info("design matrix: %d rows x %d features (%d engineered + %d categorical)",
                len(df), len(cols), len(feat_cols), len(CTX))
    return df, cols, cat_idx


def main() -> None:
    df, cols, cat_idx = build()
    X = df[cols].to_numpy(dtype=np.float32)
    y = df.edited.to_numpy(dtype=np.float64)
    fold = df.fold.to_numpy()
    held = fold == 0
    cat_mask = np.zeros(len(cols), dtype=bool)
    cat_mask[cat_idx] = True

    out: dict[str, object] = {}
    best = None
    for gi, grid in enumerate(GRID):
        for target_name in ("raw", "rank"):
            preds = np.zeros(int(held.sum()))
            for k in (1, 2, 3, 4, 5):
                tr = (fold != 0) & (fold != k)
                yt = y[tr]
                if target_name == "rank":
                    yt = rankdata(yt) / len(yt)  # within-training-fold normalised rank
                m_ = HistGradientBoostingRegressor(
                    categorical_features=cat_mask, **BASE, **grid)
                m_.fit(X[tr], yt)
                preds += m_.predict(X[held]) / 5.0
            rho = float(spearmanr(preds, y[held]).statistic)
            logger.info("grid %d (%s), %s target: held-out pooled Spearman = %.4f",
                        gi, grid["max_iter"], target_name, rho)
            out[f"grid{gi}_{target_name}"] = {"heldout_spearman": rho, **grid}
            if best is None or rho > best[1]:
                best = ((gi, target_name), rho, preds)

    (gi, target_name), rho, preds = best
    logger.info("")
    logger.info("=== best GBM: grid %d, %s target, pooled rho = %.4f ===", gi, target_name, rho)

    h = df[held].copy()
    h["gbm"] = preds
    h2h = pd.read_parquet(H2H)[["record_id", "target_group", "op"]]
    # The FINAL frozen model, not the round-1 vector stored in the head-to-head file.
    final = pd.read_parquet(
        "results/round5/heldout_calibrated.parquet")[["record_id", "predicted_efficiency"]]
    m = h.merge(h2h, on="record_id", validate="1:1").merge(
        final, on="record_id", validate="1:1")
    m["cond"] = m.cond_label

    logger.info("")
    logger.info("%-26s %9s %9s %9s", "partition", "GBM", "OptiPrime", "PE-RankFormer")
    rows = {}
    for lab, sel in (("all", np.ones(len(m), bool)),
                     ("Liu (hsu2026)", (m.source_label == "hsu2026").to_numpy()),
                     ("Kim (deepprime)", (m.source_label == "deepprime").to_numpy())):
        if sel.sum() == 0:
            continue
        g = m[sel]
        rows[lab] = {"n": int(len(g)),
                     "gbm": float(spearmanr(g.gbm, g.edited).statistic),
                     "optiprime": float(spearmanr(g.op, g.edited).statistic),
                     "pe_rankformer": float(spearmanr(g.predicted_efficiency, g.edited).statistic)}
        r = rows[lab]
        logger.info("%-26s %9.4f %9.4f %9.4f", f"{lab} (n={r['n']})",
                    r["gbm"], r["optiprime"], r["pe_rankformer"])
    out["partitions"] = rows

    # within-condition, n-weighted
    dd, ww = [], []
    for _, g in m.groupby("cond"):
        if len(g) < 50 or g.edited.nunique() < 5:
            continue
        dd.append((float(spearmanr(g.gbm, g.edited).statistic),
                   float(spearmanr(g.predicted_efficiency, g.edited).statistic)))
        ww.append(len(g))
    gbm_w = float(np.average([a for a, _ in dd], weights=ww))
    pe_w = float(np.average([b for _, b in dd], weights=ww))
    logger.info("")
    logger.info("within-condition, n-weighted (k=%d): GBM %.4f  PE-RankFormer %.4f  delta %+.4f",
                len(dd), gbm_w, pe_w, pe_w - gbm_w)
    out["within_condition"] = {"n_strata": len(dd), "gbm": gbm_w,
                               "pe_rankformer": pe_w, "delta": pe_w - gbm_w}
    out["best"] = {"grid": gi, "target": target_name, "pooled_spearman": rho,
                   **GRID[gi]}
    out["n_features"] = len(cols)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    m[["record_id", "gbm"]].to_parquet(PRED_OUT, index=False)
    logger.info("wrote %s and %s", OUT, PRED_OUT)


if __name__ == "__main__":
    main()
