"""Global and within-target evaluation metrics (task spec §30-32)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class GlobalMetrics:
    pearson: float
    spearman: float
    mae: float
    rmse: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {"pearson": self.pearson, "spearman": self.spearman, "mae": self.mae, "rmse": self.rmse, "n": self.n}


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> GlobalMetrics:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    r = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return GlobalMetrics(pearson=float(r), spearman=float(rho), mae=mae, rmse=rmse, n=len(y_true))


def within_target_spearman(
    df: pd.DataFrame, group_col: str, true_col: str, pred_col: str, min_n: int = 5
) -> pd.DataFrame:
    """Per-group Spearman rho, restricted to groups with >= min_n members with
    nonzero variance in the true values."""
    rows = []
    for key, g in df.groupby(group_col):
        if len(g) < min_n:
            continue
        if g[true_col].nunique() < 2:
            continue
        rho = spearmanr(g[true_col], g[pred_col]).statistic
        rows.append({"group": key, "n": len(g), "spearman": rho})
    return pd.DataFrame(rows)


def top_k_regret(df: pd.DataFrame, group_col: str, true_col: str, pred_col: str, k: int, min_n: int = 2) -> pd.DataFrame:
    """R_k = max(true) - max(true among predicted top-k)."""
    rows = []
    for key, g in df.groupby(group_col):
        if len(g) < min_n:
            continue
        y_max = g[true_col].max()
        top_k = g.nlargest(min(k, len(g)), pred_col)
        regret = y_max - top_k[true_col].max()
        rows.append({"group": key, "n": len(g), "regret": regret})
    return pd.DataFrame(rows)


def top_k_recall(df: pd.DataFrame, group_col: str, true_col: str, pred_col: str, k: int, min_n: int = 2) -> float:
    """Fraction of groups where the truly-best design is among the model's top-k picks."""
    hits, total = 0, 0
    for _, g in df.groupby(group_col):
        if len(g) < min_n:
            continue
        total += 1
        true_best = g[true_col].idxmax()
        pred_top_k = set(g.nlargest(min(k, len(g)), pred_col).index)
        hits += int(true_best in pred_top_k)
    return hits / total if total else float("nan")


def ndcg_at_k(df: pd.DataFrame, group_col: str, true_col: str, pred_col: str, k: int, min_n: int = 2) -> float:
    scores = []
    for _, g in df.groupby(group_col):
        if len(g) < min_n:
            continue
        g_sorted_by_pred = g.sort_values(pred_col, ascending=False).head(k)
        gains = g_sorted_by_pred[true_col].to_numpy()
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))

        ideal = g.sort_values(true_col, ascending=False).head(k)[true_col].to_numpy()
        idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else float("nan")


def target_level_bootstrap_ci(
    values_by_target: list[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap over target groups (not rows) -- task spec §33."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values_by_target, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(arr))
    boots = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return point, float(lo), float(hi)
