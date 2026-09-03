"""Shared helpers for the Phase 1 re-analyses.

Every Phase 1 script is a descriptive re-analysis of already-frozen predictions. None of
them trains, tunes or selects anything, so none of them can leak the held-out set. The
helpers here exist so that provenance recording and the clustered bootstrap are written
once and identically everywhere, rather than seven times with seven chances to differ.

Clustering unit: the **protospacer** (`spacer`), which gives 750 clusters over the
20,509 held-out rows. The head-to-head file's `target_group` column is a finer
target-site key (15,661 groups); clustering on it treats correlated designs as
independent and understates every interval.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "revision"
CORPUS = ROOT / "data/processed/optiprime_official_318471.parquet"
H2H = ROOT / "results/heldout_full_head_to_head.parquet"
CAL = ROOT / "results/round5/heldout_calibrated.parquet"
N_BOOT_DEFAULT = 5000


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def _sha(p: Path, cap: int = 1 << 25) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            if f.tell() > cap:
                break
    return h.hexdigest()[:16]


def provenance(inputs: Sequence[Path], seed: int) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()[:12]
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", True
    return {"git_commit": commit, "working_tree_dirty": dirty, "seed": seed,
            "inputs": {str(Path(p).relative_to(ROOT)): _sha(Path(p))
                       for p in inputs if Path(p).exists()}}


def write_outputs(stem: str, result: dict, markdown: str) -> None:
    REV.mkdir(parents=True, exist_ok=True)
    (REV / f"{stem}.json").write_text(json.dumps(result, indent=2, default=float))
    (REV / f"{stem}.md").write_text(markdown)
    print(f"wrote {REV / (stem + '.json')} and {REV / (stem + '.md')}")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman. The target has a large exact-zero block, so average ranks
    (Spearman's own convention) are required or this is not the reported statistic."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """tau-b, which corrects for ties in both variables.

    Computed in O(n^2) over the pair matrix, which is fine at per-target sizes and for
    the full 20,509-row set is done once per model rather than inside a bootstrap.
    """
    from scipy.stats import kendalltau
    return float(kendalltau(a, b, variant="b", nan_policy="omit").statistic)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC, tie-corrected (equivalent to the Mann-Whitney statistic)."""
    label = np.asarray(label).astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_heldout() -> pd.DataFrame:
    """The 20,509 held-out rows with the FINAL frozen predictions, OptiPrime, and the
    protospacer. Note the head-to-head file's own `predicted_efficiency` is the round-1
    model (0.8865), so the final vector is joined from the round-5 calibrated file."""
    h = pd.read_parquet(H2H, columns=["record_id", "source_study", "cell_type",
                                      "pe_type", "y", "op"])
    c = pd.read_parquet(CAL, columns=["record_id", "predicted_efficiency",
                                      "calibrated_efficiency", "member_ordSSM"])
    s = pd.read_parquet(CORPUS, columns=["record_id", "spacer", "rtt", "pbs"])
    m = h.merge(c, on="record_id", validate="1:1").merge(s, on="record_id", validate="1:1")
    assert len(m) == 20509, f"expected 20,509 held-out rows, got {len(m)}"
    assert m.spacer.nunique() == 750, f"expected 750 protospacer clusters, got {m.spacer.nunique()}"
    m["cond"] = m.source_study + "|" + m.cell_type + "|" + m.pe_type
    m = m.rename(columns={"predicted_efficiency": "ours"})
    return m


# --------------------------------------------------------------------------- #
# clustered bootstrap
# --------------------------------------------------------------------------- #
def clusters_of(df: pd.DataFrame, key: str = "spacer") -> list[np.ndarray]:
    """Positional row indices grouped by `key`, for resampling clusters not rows."""
    codes, uniq = pd.factorize(df[key])
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float],
                      seed: int, n_boot: int = N_BOOT_DEFAULT,
                      key: str = "spacer") -> dict:
    """Protospacer-clustered bootstrap of any statistic of the frame.

    Returns the observed value, the percentile interval, the fraction of resamples above
    zero, and a two-sided empirical p-value bounded below by 1/n_boot. Resamples whose
    statistic is undefined (e.g. a partition that lost all its variation) are dropped and
    counted, rather than silently propagating NaN.
    """
    obs = float(stat(df))
    cl = clusters_of(df, key)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    vals[:] = np.nan
    for i in range(n_boot):
        idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
        vals[i] = stat(df.iloc[idx])
    ok = np.isfinite(vals)
    v = vals[ok]
    if v.size == 0:
        return {"observed": obs, "ci95": [np.nan, np.nan], "n_valid": 0,
                "n_boot": n_boot, "n_clusters": len(cl)}
    lo, hi = np.percentile(v, [2.5, 97.5])
    frac = float((v > 0).mean())
    p = max(2 * min(frac, 1 - frac), 1.0 / v.size)
    return {"observed": obs, "ci95": [float(lo), float(hi)],
            "bootstrap_mean": float(v.mean()), "frac_above_zero": frac,
            "two_sided_p": float(p), "n_valid": int(v.size),
            "n_boot": n_boot, "n_clusters": len(cl)}
