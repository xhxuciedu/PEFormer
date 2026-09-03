"""Within-condition comparison against OptiPrime, and absolute-error head-to-head.

Round-9 addition, prompted by two reviewer objections that the pooled headline number
invites.

**Objection 1: the pooled figure is inflated by between-source offsets.** Pooled
Spearman on the held-out set (0.9079) exceeds both partition scores (Liu 0.8585, Kim
0.8124), which is possible because the two studies differ in efficiency level, so
between-study variance contributes signal that no within-condition ranking task would
supply. The model is additionally given `source_study` as one of its seven context
fields. Both are true, and both are reasons to check the *margin* rather than the
absolute level: this script recomputes the comparison inside each experimental
condition, where no between-condition offset can contribute, and attaches a
protospacer-clustered bootstrap interval to the stratified difference.

**Objection 2: the eight-fold MAE reduction has a meaningless baseline.** The
uncalibrated rank average is not measured in efficiency units, so comparing its MAE to
the calibrated model's tells the reader nothing about absolute accuracy. The comparison
that matters is against the baseline, on identical rows, which this script reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stratified")

H2H = Path("results/heldout_full_head_to_head.parquet")
CORPUS = Path("data/processed/optiprime_official_318471.parquet")
CAL = Path("results/round5/heldout_calibrated.parquet")
OUT = Path("results/round9/stratified_comparison.json")
MIN_ROWS_TABLE = 300   # reporting threshold for the per-condition table
MIN_ROWS_BOOT = 50     # inclusion threshold inside a bootstrap resample
N_BOOT, SEED = 5000, 20260903


def load() -> pd.DataFrame:
    h = pd.read_parquet(H2H)[["record_id", "source_study", "cell_type", "pe_type", "y", "op"]]
    c = pd.read_parquet(CAL)[["record_id", "predicted_efficiency", "calibrated_efficiency"]]
    # Cluster on the PROTOSPACER, which is the unit of dependence the paper resamples on
    # and which gives 750 clusters over these rows. The `target_group` column in the
    # head-to-head file is a much finer target-site key (15,661 groups) and clustering on
    # it would treat correlated designs as independent, understating every interval --
    # the exact error the clustered bootstrap exists to avoid.
    sp = pd.read_parquet(CORPUS, columns=["record_id", "spacer"])
    m = h.merge(c, on="record_id", validate="1:1").merge(sp, on="record_id", validate="1:1")
    assert len(m) == 20509, f"expected the full held-out set, got {len(m)}"
    assert m.spacer.nunique() == 750, f"expected 750 protospacer clusters, got {m.spacer.nunique()}"
    m["cond"] = m.source_study + "|" + m.cell_type + "|" + m.pe_type
    return m


def _ranks(v: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks. The target has a large exact-zero block, so ties must be
    handled the way Spearman's convention does or the statistic is not the one reported."""
    return rankdata(v, method="average")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def stratified_delta_fast(codes: np.ndarray, y: np.ndarray, op: np.ndarray,
                          pr: np.ndarray, n_codes: int, min_rows: int) -> tuple[float, int]:
    """n-weighted mean within-condition Spearman difference, on plain arrays.

    `codes` are integer condition labels for the rows supplied. Splitting by argsort
    rather than a pandas groupby is what makes a 5,000-resample bootstrap affordable.
    """
    order = np.argsort(codes, kind="stable")
    cs, ys, ops, prs = codes[order], y[order], op[order], pr[order]
    bounds = np.searchsorted(cs, np.arange(n_codes + 1))
    num = den = 0.0
    k = 0
    for c in range(n_codes):
        lo, hi = bounds[c], bounds[c + 1]
        if hi - lo < min_rows:
            continue
        yy = ys[lo:hi]
        if np.unique(yy).size < 5:
            continue
        a = _spearman(ops[lo:hi], yy)
        b = _spearman(prs[lo:hi], yy)
        if np.isnan(a) or np.isnan(b):
            continue
        w = float(hi - lo)
        num += w * (b - a)
        den += w
        k += 1
    return (num / den if den > 0 else np.nan), k


def main() -> None:
    m = load()
    pred = "calibrated_efficiency"  # monotone in the frozen ensemble score; same ranking
    out: dict[str, object] = {}

    # ---- absolute-error head-to-head on identical rows ----
    logger.info("")
    logger.info("=== absolute accuracy on identical %d held-out rows ===", len(m))
    logger.info("%-28s %9s %9s %8s %8s", "model", "Spearman", "Pearson", "MAE", "RMSE")
    metrics = {}
    for name, col in (("OptiPrime (5-model ens)", "op"), ("PE-RankFormer + isotonic", pred)):
        v, y = m[col].to_numpy(), m.y.to_numpy()
        metrics[name] = dict(
            spearman=float(spearmanr(v, y).statistic), pearson=float(np.corrcoef(v, y)[0, 1]),
            mae=float(np.abs(v - y).mean()), rmse=float(np.sqrt(((v - y) ** 2).mean())))
        r = metrics[name]
        logger.info("%-28s %9.4f %9.4f %8.4f %8.4f",
                    name, r["spearman"], r["pearson"], r["mae"], r["rmse"])
    out["absolute_accuracy"] = metrics

    # ---- per-condition table ----
    logger.info("")
    logger.info("=== per-condition Spearman (n >= %d) ===", MIN_ROWS_TABLE)
    rows = []
    for c, g in m.groupby("cond", observed=True):
        if len(g) < MIN_ROWS_TABLE:
            continue
        rows.append(dict(cond=c, n=int(len(g)),
                         optiprime=float(spearmanr(g.op, g.y).statistic),
                         pe_rankformer=float(spearmanr(g[pred], g.y).statistic)))
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    t["delta"] = t.pe_rankformer - t.optiprime
    for r in t.itertuples():
        logger.info("%-28s %6d  %.4f  %.4f  %+.4f",
                    r.cond, r.n, r.optiprime, r.pe_rankformer, r.delta)
    logger.info("conditions favouring PE-RankFormer: %d/%d", int((t.delta > 0).sum()), len(t))
    out["per_condition"] = t.to_dict(orient="records")
    out["conditions_won"] = int((t.delta > 0).sum())
    out["conditions_total"] = int(len(t))

    # ---- pooled vs stratified, with a clustered bootstrap on the stratified difference ----
    y = m.y.to_numpy(); op = m.op.to_numpy(); pr = m[pred].to_numpy()
    pooled = float(_spearman(pr, y) - _spearman(op, y))

    src_codes, src_uniq = pd.factorize(m.source_study, sort=True)
    by_source, k_src = stratified_delta_fast(
        src_codes, y, op, pr, len(src_uniq), MIN_ROWS_BOOT)
    cond_codes, cond_uniq = pd.factorize(m.cond, sort=True)
    by_cond, k_cond = stratified_delta_fast(
        cond_codes, y, op, pr, len(cond_uniq), MIN_ROWS_BOOT)

    # Protospacers, not rows, are resampled: many designs share a target, so a row-level
    # bootstrap would treat correlated observations as independent.
    codes, uniq = pd.factorize(m.spacer)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    clusters = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    logger.info("resampling %d protospacer clusters over %d rows", len(clusters), len(m))
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[j] for j in pick])
        boots[i], _ = stratified_delta_fast(
            cond_codes[idx], y[idx], op[idx], pr[idx], len(cond_uniq), MIN_ROWS_BOOT)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    logger.info("")
    logger.info("=== is the margin an artifact of pooling? ===")
    logger.info("pooled difference                    : %+.4f", pooled)
    logger.info("within-source, n-weighted   (k=%d)    : %+.4f", k_src, by_source)
    logger.info("within-condition, n-weighted (k=%2d)  : %+.4f  95%% CI [%+.4f, %+.4f]  "
                "ahead in %.1f%% of %d resamples",
                k_cond, by_cond, lo, hi, 100 * float((boots > 0).mean()), N_BOOT)
    out["margin"] = {
        "pooled": pooled,
        "within_source": {"delta": by_source, "n_strata": k_src},
        "within_condition": {"delta": by_cond, "n_strata": k_cond,
                             "ci95": [float(lo), float(hi)],
                             "frac_ahead": float((boots > 0).mean()),
                             "n_boot": N_BOOT, "seed": SEED},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
