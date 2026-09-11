"""Invariants for the follow-up protocol, independent of expensive neural runs."""
import numpy as np
import pandas as pd
import pytest

from e31_label_budgets import nested_components
from e29_summarize_audit import paired_interval, metrics
from endpoints import paired_cluster_bootstrap, score_population
from e32_score_scale_diagnostic import summarize_scores


def toy():
    return pd.DataFrame([
        {"component": component, "edit_key": f"{component}_{group}", "design_key": str(design),
         "split": "train" if component < 8 else "test"}
        for component in range(10) for group in range(component % 3 + 1) for design in range(3)
    ])


def test_nested_complete_and_split_isolated():
    frame = toy()
    subsets = nested_components(frame, [1, 5, 9, None], 42)
    previous = set()
    for nominal, selected in subsets.items():
        assert previous <= set(selected)
        sub = frame.loc[frame.component.isin(selected)]
        assert set(sub.split) == {"train"}
        assert not sub.duplicated(["edit_key", "design_key"]).any()
        assert sub.groupby("component").size().equals(
            frame.groupby("component").size().loc[np.sort(selected)])
        if nominal is not None:
            assert sub.edit_key.nunique() >= nominal
            prior = frame.loc[frame.component.isin(selected[:-1])]
            assert prior.edit_key.nunique() < nominal
        previous = set(selected)
    assert previous == set(frame.loc[frame.split == "train", "component"])


def test_deterministic_and_row_order_independent():
    frame = toy()
    a = nested_components(frame, [5, 9], 42)
    b = nested_components(frame.sample(frac=1, random_state=11), [5, 9], 42)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_outcomes_cannot_change_budgets():
    frame = toy()
    a = nested_components(frame.assign(y=0), [5], 42)
    b = nested_components(frame.assign(y=np.arange(len(frame))), [5], 42)
    np.testing.assert_array_equal(a[5], b[5])


def test_reject_invalid_budget():
    with pytest.raises(ValueError):
        nested_components(toy(), [0], 42)


def test_fast_bootstrap_matches_existing_endpoint():
    rng = np.random.default_rng(123)
    diff = rng.normal(0.1, 1, 60)
    cluster = np.repeat(np.arange(20), 3)
    a = paired_interval(diff, cluster, seed=44, n_boot=333)
    b = paired_cluster_bootstrap(diff, cluster, seed=44, n_boot=333)
    np.testing.assert_allclose(a["ci95"], b["ci95"], atol=1e-14)
    assert a["two_sided_p"] == b["two_sided_p"]
    assert paired_interval(np.zeros(60), cluster)["two_sided_p"] == 1


def test_metrics_actually_evaluate_both_heads_and_source_eligibility():
    frame = pd.DataFrame({"edit_key": ["a", "a", "b", "b"],
                          "design_key": ["1", "2", "1", "2"],
                          "component": ["x", "x", "z", "z"],
                          "y": [0., 1., 0., 0.],
                          "prediction": [0., 1., 0., 0.],
                          "selection": [1., 0., 0., 0.]})
    target, _ = metrics(frame)
    source, _ = metrics(frame, source=True)
    assert target["prediction"]["achieved_at_1"] == .5
    assert target["selection"]["achieved_at_1"] == 0
    assert source["prediction"]["achieved_at_1"] == 1
    assert source["selection"]["achieved_at_1"] == 0


def test_score_scale_diagnostic_shift_invariance_and_bound():
    frame = pd.DataFrame({"edit_key": ["a"] * 8, "selection": [1.] + [0.] * 7})
    a = summarize_scores(frame, 1)
    b = summarize_scores(frame.assign(selection=frame.selection + 10), 1)
    assert a == b
    assert a["mean_max_probability"] == pytest.approx(np.e / (np.e + 7))
    assert summarize_scores(frame, .1)["mean_max_probability"] > a["mean_max_probability"]


def test_fast_metrics_match_endpoint_with_ties_and_duplicates():
    rng = np.random.default_rng(17)
    frame = toy().assign(y=rng.random(len(toy())), prediction=rng.integers(0, 3, len(toy())),
                         selection=rng.integers(0, 3, len(toy())))
    frame = pd.concat([frame, frame.iloc[:4]], ignore_index=True)
    _, fast = metrics(frame)
    cand = frame.groupby(["edit_key", "design_key"], as_index=False).agg(
        y=("y", "mean"), prediction=("prediction", "mean"), selection=("selection", "mean"),
        component=("component", "first"))
    ref = score_population(cand, ["prediction", "selection"], ks=(1,), outcome="y", site="component")
    for col in ("prediction_at_1", "selection_at_1", "prediction_regret", "selection_regret"):
        np.testing.assert_allclose(fast.set_index("edit_key")[col].sort_index(),
                                   ref.set_index("edit_key")[col].sort_index())
