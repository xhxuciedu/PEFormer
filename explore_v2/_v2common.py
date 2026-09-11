"""Shared plumbing for the explore_v2 program (research plan section 8A).

Everything here is descriptive analysis of frozen artefacts plus small models fitted
inside declared splits. The held-out fold 0 is used for audit and diagnostics only; every
choice that could be a model selection is made on development folds, which is why
`load_corpus` exposes the fold column rather than hiding it behind a "held-out" loader.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "explore_v2"
CACHE = ROOT / "explore_v2" / "cache"
CORPUS = ROOT / "data/processed/optiprime_official_318471.parquet"


def manifest_path(version: int | None = None) -> Path:
    """Path to the decision manifest for a given canonicaliser version.

    The manifest is versioned because the allele key defines it. E01-E09 as published were
    built on version 1, whose indel keys were not strand-invariant (see CORRECTIONS.md C3);
    version 2 fixes that. Mixing them would compare groups defined two different ways, so a
    downstream script that cannot find its version fails loudly instead.
    """
    from canon import CANON_VERSION
    v = CANON_VERSION if version is None else version
    return CACHE / f"decision_manifest_v{v}.parquet"


def cache_path(stem: str, version: int | None = None) -> Path:
    """Version any derived cache on the canonicaliser, for the same reason as the manifest."""
    from canon import CANON_VERSION
    v = CANON_VERSION if version is None else version
    return CACHE / f"{stem}_v{v}.parquet"


def require_manifest(version: int | None = None) -> Path:
    p = manifest_path(version)
    if not p.exists():
        raise FileNotFoundError(
            f"{p.name} is missing. The canonicaliser version changed, so the cached "
            "manifest is stale rather than absent: re-run "
            "`explore_v2/e01_decision_manifest.py` to rebuild it, and re-run E02-E09 on "
            "top of it. To reproduce the published numbers instead, set CANON_VERSION = 1 "
            "in explore_v2/canon.py."
        )
    return p
H2H = ROOT / "results/heldout_full_head_to_head.parquet"
CAL = ROOT / "results/round5/heldout_calibrated.parquet"

sys.path.insert(0, str(Path(__file__).resolve().parent))


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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.json").write_text(json.dumps(result, indent=2, default=float))
    (OUT / f"{stem}.md").write_text(markdown)
    print(f"wrote {OUT / (stem + '.json')} and {OUT / (stem + '.md')}")


# --------------------------------------------------------------------------- #
# metrics (same definitions as revision/_common so numbers stay comparable)
# --------------------------------------------------------------------------- #
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_corpus(canonical: bool = True) -> pd.DataFrame:
    """The full 318,471-row official corpus with canonical edit/design/context keys.

    Cached, because canonicalisation is a few seconds and every downstream script wants
    the same keys; the cache is keyed on nothing, so delete it if `canon.py` changes.
    """
    from canon import CANON_VERSION
    CACHE.mkdir(parents=True, exist_ok=True)
    # The key definition is part of the cache identity. A manifest built by an older
    # canonicaliser must not be picked up by newer code.
    cached = CACHE / f"corpus_canonical_v{CANON_VERSION}.parquet"
    if canonical and cached.exists():
        return pd.read_parquet(cached)
    df = pd.read_parquet(CORPUS)
    if canonical:
        from canon import add_canonical_extended, add_group_keys
        df = add_group_keys(add_canonical_extended(df))
        df.to_parquet(cached)
    return df


def load_heldout_predictions() -> pd.DataFrame:
    """Fold-0 rows with the frozen final ensemble, its ordinal-S4D member, and OptiPrime.

    `ours` is the round-4 final ensemble (the paper's headline predictor) and `ordssm` is
    the single ordinal-S4D member the research plan proposes to keep as the backbone;
    reporting both keeps the audit honest about which predictor a number describes.
    """
    h = pd.read_parquet(H2H, columns=["record_id", "y", "op"])
    c = pd.read_parquet(CAL, columns=["record_id", "predicted_efficiency",
                                      "calibrated_efficiency", "member_ordSSM"])
    m = h.merge(c, on="record_id", validate="1:1").rename(
        columns={"predicted_efficiency": "ours", "member_ordSSM": "ordssm"})
    assert len(m) == 20509, len(m)
    return m


# --------------------------------------------------------------------------- #
# clustered bootstrap
# --------------------------------------------------------------------------- #
def clusters_of(df: pd.DataFrame, key: str) -> list[np.ndarray]:
    codes, uniq = pd.factorize(df[key])
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float],
                      seed: int, key: str, n_boot: int = 2000) -> dict:
    """Percentile bootstrap resampling whole clusters of `key`.

    The dependence unit for every decision metric here is the locus, not the row and not
    the candidate pair: designs at one site share sequence, library, and measurement
    batch, so resampling rows would understate every interval.
    """
    obs = float(stat(df))
    cl = clusters_of(df, key)
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
        vals[i] = stat(df.iloc[idx])
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return {"observed": obs, "ci95": [np.nan, np.nan], "n_valid": 0,
                "n_clusters": len(cl)}
    lo, hi = np.percentile(v, [2.5, 97.5])
    frac = float((v > 0).mean())
    return {"observed": obs, "ci95": [float(lo), float(hi)],
            "bootstrap_mean": float(v.mean()), "frac_above_zero": frac,
            "two_sided_p": float(max(2 * min(frac, 1 - frac), 1.0 / v.size)),
            "n_valid": int(v.size), "n_boot": n_boot, "n_clusters": len(cl)}
