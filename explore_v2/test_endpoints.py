"""Tests for the decision endpoints. Run: .venv/bin/python -m pytest explore_v2/test_endpoints.py -q

Each test corresponds to an invariant that the inline E11 code violated, plus the edge cases
`NEXT_STEPS_AFTER_E15.md` section 2 asked for: identical predictions, all-zero outcomes, tied
scores, duplicate candidates, row permutation, and k exceeding depth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import endpoints as E


# --------------------------------------------------------------------- candidates
def test_duplicate_designs_are_one_candidate():
    df = pd.DataFrame({"edit_key": ["a"] * 3, "design_key": ["d1", "d1", "d2"],
                       "edited_frac": [0.1, 0.3, 0.5], "protospacer": ["P"] * 3})
    out = E.aggregate_candidates(df)
    assert len(out) == 2, "two distinct designs must give two candidates"
    assert out.loc[out.design_key == "d1", "edited_frac"].item() == pytest.approx(0.2)
    assert out.loc[out.design_key == "d1", "n_rows"].item() == 2


def test_depth_after_aggregation_blocks_at_k():
    """The E11 bug: 3 rows over 2 designs must not yield a top-three result."""
    df = pd.DataFrame({"edit_key": ["a"] * 3, "design_key": ["d1", "d1", "d2"],
                       "edited_frac": [0.1, 0.3, 0.5], "protospacer": ["P"] * 3,
                       "m": [1.0, 1.0, 0.0]})
    cand = E.aggregate_candidates(df, scores=("m",))
    t = E.score_population(cand, ["m"], ks=(1, 3))
    assert t.depth.item() == 2
    assert np.isnan(t.m_at_3.item()), "@3 is undefined at depth 2"
    assert np.isnan(t.rand_at_3.item())


# --------------------------------------------------------------------- endpoints
def test_k_exceeding_depth_is_nan_not_truncated():
    y, s = np.array([0.1, 0.4]), np.array([1.0, 0.0])
    assert E.achieved_at_k(y, s, 1) == pytest.approx(0.1)
    assert np.isnan(E.achieved_at_k(y, s, 3))


def test_all_zero_group_random_hit_is_one():
    y = np.zeros(4)
    assert E.random_hit_at_1(y) == pytest.approx(1.0), "every design is a maximiser"
    assert E.random_regret_at_1(y) == pytest.approx(0.0)


def test_random_hit_counts_tied_maximisers():
    y = np.array([0.5, 0.5, 0.1, 0.0])
    assert E.random_hit_at_1(y) == pytest.approx(0.5)   # 2 maximisers of 4, not 1/4


def test_tied_maximum_gives_full_credit():
    y = np.array([0.5, 0.5, 0.1])
    assert E.hit_at_1(y, np.array([1.0, 0.0, 0.0])) == 1.0
    assert E.hit_at_1(y, np.array([0.0, 1.0, 0.0])) == 1.0
    assert E.hit_at_1(y, np.array([0.0, 0.0, 1.0])) == 0.0


def test_random_achieved_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.random(6)
    from itertools import combinations
    for k in (1, 2, 3, 6):
        brute = np.mean([max(y[list(c)]) for c in combinations(range(6), k)])
        assert E.random_achieved_at_k(y, k) == pytest.approx(brute)


def test_row_permutation_invariance():
    rng = np.random.default_rng(1)
    y, s = rng.random(5), rng.random(5)
    p = rng.permutation(5)
    for k in (1, 2, 5):
        assert E.achieved_at_k(y, s, k) == pytest.approx(E.achieved_at_k(y[p], s[p], k))
    assert E.hit_at_1(y, s) == pytest.approx(E.hit_at_1(y[p], s[p]))
    assert E.random_hit_at_1(y) == pytest.approx(E.random_hit_at_1(y[p]))


def test_tied_scores_resolve_deterministically():
    """Equal scores must not make the endpoint depend on row order beyond a stable rule."""
    y = np.array([0.1, 0.9])
    a = E.achieved_at_k(y, np.array([1.0, 1.0]), 1)
    b = E.achieved_at_k(y[::-1], np.array([1.0, 1.0]), 1)
    assert a == pytest.approx(0.1) and b == pytest.approx(0.9), (
        "stable argsort takes the first row; the caller must therefore not read a "
        "tied-score result as a model preference")


# --------------------------------------------------------------------- inference
def test_identical_arms_give_no_evidence_against_equality():
    """The E11 bug: an all-zero difference distribution returned p = 1/n_boot."""
    diff = np.zeros(300)
    cl = np.repeat(np.arange(100), 3)
    r = E.paired_cluster_bootstrap(diff, cl, seed=1)
    assert r["observed"] == 0.0
    assert r["two_sided_p"] == pytest.approx(1.0)
    assert r["degenerate_zero_difference"] is True


def test_clear_difference_is_flagged_at_the_resolution_floor():
    diff = np.full(300, 0.05)
    cl = np.repeat(np.arange(100), 3)
    r = E.paired_cluster_bootstrap(diff, cl, seed=1)
    assert r["observed"] == pytest.approx(0.05)
    assert r["p_is_resolution_floor"] is True
    assert r["degenerate_zero_difference"] is False


def test_bootstrap_resamples_clusters_not_rows():
    rng = np.random.default_rng(2)
    # one cluster carries an extreme value; row resampling would understate the interval
    diff = np.concatenate([np.zeros(297), np.full(3, 10.0)])
    cl = np.repeat(np.arange(100), 3)
    r = E.paired_cluster_bootstrap(diff, cl, seed=3)
    assert r["n_clusters"] == 100
    assert r["ci95"][1] > r["observed"], "a heavy cluster must widen the upper tail"


def test_nan_differences_are_dropped_not_propagated():
    diff = np.array([0.1, np.nan, 0.3, 0.2])
    cl = np.array([0, 1, 2, 3])
    r = E.paired_cluster_bootstrap(diff, cl, seed=1)
    assert r["n"] == 3 and np.isfinite(r["observed"])
