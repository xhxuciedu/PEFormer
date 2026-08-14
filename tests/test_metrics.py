"""Evaluation metrics sanity checks (task spec §30-33)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pe_rankformer.evaluation.metrics import (
    global_metrics,
    ndcg_at_k,
    target_level_bootstrap_ci,
    top_k_recall,
    top_k_regret,
    within_target_spearman,
)


def test_global_metrics_perfect_prediction():
    y = np.array([0.1, 0.5, 0.9, 0.3])
    m = global_metrics(y, y)
    assert m.pearson > 0.999
    assert m.spearman > 0.999
    assert m.mae < 1e-9
    assert m.rmse < 1e-9


def test_global_metrics_anticorrelated():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    pred = y[::-1]
    m = global_metrics(y, pred)
    assert m.spearman < -0.99


def _grouped_df():
    rows = []
    for g in range(3):
        for i in range(6):
            rows.append({"group": g, "true": i + g * 0.01, "pred": i + (5 - i) * 0.001})
    return pd.DataFrame(rows)


def test_within_target_spearman_respects_min_n():
    df = _grouped_df()
    out = within_target_spearman(df, "group", "true", "pred", min_n=10)
    assert len(out) == 0  # groups only have 6 members


def test_within_target_spearman_computed_per_group():
    df = _grouped_df()
    out = within_target_spearman(df, "group", "true", "pred", min_n=5)
    assert len(out) == 3
    assert set(out.columns) >= {"group", "n", "spearman"}


def test_top_k_regret_zero_when_pred_matches_true_order():
    df = pd.DataFrame({"group": [0] * 4, "true": [0.1, 0.5, 0.9, 0.3], "pred": [0.1, 0.5, 0.9, 0.3]})
    out = top_k_regret(df, "group", "true", "pred", k=1)
    assert out.iloc[0]["regret"] == 0.0


def test_top_k_regret_positive_when_pred_picks_wrong_item():
    df = pd.DataFrame({"group": [0] * 4, "true": [0.1, 0.5, 0.9, 0.3], "pred": [0.9, 0.5, 0.1, 0.3]})
    out = top_k_regret(df, "group", "true", "pred", k=1)
    assert out.iloc[0]["regret"] == pytest_approx(0.8)


def pytest_approx(x, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol

    return _Approx()


def test_top_k_recall_perfect_when_predictions_match():
    df = pd.DataFrame({"group": [0] * 4, "true": [0.1, 0.5, 0.9, 0.3], "pred": [0.1, 0.5, 0.9, 0.3]})
    recall = top_k_recall(df, "group", "true", "pred", k=1)
    assert recall == 1.0


def test_ndcg_at_k_perfect_ranking():
    df = pd.DataFrame({"group": [0] * 4, "true": [0.1, 0.5, 0.9, 0.3], "pred": [0.1, 0.5, 0.9, 0.3]})
    ndcg = ndcg_at_k(df, "group", "true", "pred", k=3)
    assert ndcg > 0.99


def test_bootstrap_ci_contains_point_estimate():
    vals = [0.5, 0.6, 0.7, 0.4, 0.55]
    point, lo, hi = target_level_bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo <= point <= hi


def test_bootstrap_ci_handles_nans():
    vals = [0.5, float("nan"), 0.7]
    point, lo, hi = target_level_bootstrap_ci(vals, n_boot=200, seed=1)
    assert not np.isnan(point)
