"""Cheap protocol/loss tests plus checks of the actual prepared manifests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest
import torch
from common import CACHE, ROOT
from model import Selector, pairwise, utility, preserve, slices
from train import budget_indices

def test_identity_and_permutation():
    torch.manual_seed(1)
    h, q = torch.randn(12, 8), torch.linspace(-2, 2, 12)
    m = Selector(8, residual=True).eval()
    assert torch.equal(m(h, q), q)
    perm = torch.randperm(12)
    assert torch.equal(m(h[perm], q[perm]), q[perm])
    assert torch.equal(m(h[:4], q[:4]), m(h, q)[:4])

@pytest.mark.parametrize("loss", [pairwise, lambda s,y,g: utility(s,y,g,.1), preserve])
def test_empty_group_gradient_is_valid(loss):
    score = torch.randn(4, requires_grad=True)
    value = loss(score, torch.zeros(4), [slice(i,i+1) for i in range(4)])
    value.backward()
    assert value.item() == 0
    assert torch.equal(score.grad, torch.zeros_like(score))

def test_constant_outcomes_no_rank_or_utility_gradient():
    for loss in (pairwise, lambda s,y,g: utility(s,y,g,.1)):
        score = torch.randn(5, requires_grad=True)
        loss(score, torch.ones(5), [slice(0,5)]).backward()
        assert torch.equal(score.grad, torch.zeros_like(score))

def test_pairwise_prefers_correct_order_and_is_shift_invariant():
    y = torch.tensor([0., .01, .03])
    good = torch.tensor([-1., 0., 1.])
    g = [slice(0,3)]
    assert pairwise(good,y,g) < pairwise(-good,y,g)
    assert pairwise(good,y,g) == pairwise(good+5,y,g)

def test_preservation_acts_on_selector_and_not_teacher():
    teacher = torch.tensor([-1., 1.], requires_grad=True)
    s = teacher.detach().clone().requires_grad_()
    v = preserve(s,teacher,[slice(0,2)])
    v.backward()
    assert abs(v.item()) < 1e-6
    assert s.grad.abs().max() < 1e-6
    assert teacher.grad is None
    s = torch.tensor([1.,-1.], requires_grad=True)
    v = preserve(s,teacher,[slice(0,2)])
    v.backward()
    assert v > 0 and s.grad[0] > 0 and s.grad[1] < 0

def test_ties_are_design_key_deterministic():
    from train import endpoint_frame
    from e29_summarize_audit import metrics
    f = pd.DataFrame(dict(group_id=["a","a"], design_key=["z","a"], component=["x","x"], y=[1.,0.]))
    score = np.zeros(2)
    m, _ = metrics(endpoint_frame(f, np.arange(2), score, score))
    assert m["selection"]["achieved_at_1"] == 0

def test_actual_budget_and_source_isolation():
    path = CACHE / "index.parquet"
    if not path.exists():
        pytest.skip("Run preparation for actual-manifest integration checks")
    f = pd.read_parquet(path)
    b = pd.read_parquet(ROOT / "explore_v2/cache/adapt_followup_budget_components.parquet")
    part = pd.read_parquet(ROOT / "explore_v2/cache/adaptation_partition_v2.parquet")
    assert "test" not in set(f.surface)
    assert f.groupby("component").surface.nunique().max() == 1
    prev = {}
    for budget in (200,1000):
        idx = budget_indices(f,b,budget,20260910)
        total = sum(f.iloc[ix].gid.nunique() for ix in idx.values())
        assert total == (202 if budget == 200 else 1001)
        for role, ix in idx.items():
            assert set(f.iloc[ix].surface) == {"target_budget_pool"}
            current = set(f.iloc[ix].component_numeric)
            if role in prev:
                assert prev[role] <= current
            prev[role] = current
            expected = part.loc[(part.split == "train") & part.component.isin(current)]
            assert len(expected) == len(ix)
    audit = f.loc[f.surface == "source_audit"]
    sr = f.loc[f.surface.isin(["source_replay","source_val"])]
    assert not set(audit.spacer) & set(sr.spacer)
    assert not set(audit.edit_key) & set(sr.edit_key)
    target = part
    assert not set(sr.spacer) & set(target.protospacer)
    assert not set(sr.edit_key) & set(target.edit_key)

def test_report_endpoint_matches_training_endpoint():
    from summarize import group_table
    from e29_summarize_audit import metrics
    f = pd.DataFrame(dict(edit_key=["a","a","b","b","c"],
                          design_key=["z","a","a","b","a"],
                          component=["x","x","y","y","z"],
                          y=[.1,.2,0.,0.,.3], selection=[1.,1.,3.,2.,1.]))
    for source in (True,False):
        expected,_ = metrics(f.assign(prediction=f.selection), source=source)
        t = group_table(f,source=source)
        assert t.achieved.mean() == expected["selection"]["achieved_at_1"]
        assert len(t) == expected["groups"]

def test_external_sequence_ids_are_normalized_not_allele_labels():
    from audit_data import shortname
    assert shortname("1_NM_PE2_1_1") == shortname("1-NM") == "1NM"
    assert shortname("503NM_End_PE_HEK_1_1") == "503NM"
    assert shortname("control") == "unmapped"
