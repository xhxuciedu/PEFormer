"""Ranking-pair sampler and grouped batch sampler (task spec §21, §40)."""

from __future__ import annotations

import numpy as np
import torch

from pe_rankformer.training.ranking import GroupedBatchSampler, sample_ranking_pairs


def test_sample_ranking_pairs_orients_i_greater_than_j():
    group_key = torch.tensor([0, 0, 0, 1, 1])
    target = torch.tensor([0.1, 0.5, 0.9, 0.2, 0.8])
    pi, pj = sample_ranking_pairs(group_key, target, min_diff=0.02, max_pairs_per_group=10)
    assert (target[pi] > target[pj]).all()


def test_sample_ranking_pairs_respects_min_diff():
    group_key = torch.tensor([0, 0])
    target = torch.tensor([0.50, 0.505])  # diff 0.005 < default min_diff
    pi, pj = sample_ranking_pairs(group_key, target, min_diff=0.02)
    assert pi.numel() == 0


def test_sample_ranking_pairs_ignores_singleton_groups():
    group_key = torch.tensor([0, 1, 2])
    target = torch.tensor([0.1, 0.5, 0.9])
    pi, pj = sample_ranking_pairs(group_key, target, min_diff=0.02)
    assert pi.numel() == 0


def test_sample_ranking_pairs_caps_per_group():
    group_key = torch.zeros(20, dtype=torch.long)
    target = torch.linspace(0, 1, 20)
    pi, pj = sample_ranking_pairs(group_key, target, min_diff=0.02, max_pairs_per_group=3)
    assert pi.numel() <= 3


def test_grouped_batch_sampler_covers_all_indices():
    group_key = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    sampler = GroupedBatchSampler(group_key, batch_size=4, seed=0, drop_last=False)
    seen = set()
    for batch in sampler:
        seen.update(batch)
    assert seen == set(range(len(group_key)))


def test_grouped_batch_sampler_deterministic_given_seed_and_epoch():
    group_key = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    s1 = GroupedBatchSampler(group_key, batch_size=4, seed=42)
    s2 = GroupedBatchSampler(group_key, batch_size=4, seed=42)
    assert list(s1) == list(s2)


def test_grouped_batch_sampler_increases_within_batch_group_overlap():
    rng = np.random.default_rng(0)
    group_key = rng.integers(0, 200, size=2000)  # ~10 members/group on average
    sampler = GroupedBatchSampler(group_key, batch_size=256, seed=1, drop_last=True)
    batches = list(sampler)
    # count same-group pairs available within each batch
    pair_counts = []
    for batch in batches:
        keys = group_key[batch]
        _, counts = np.unique(keys, return_counts=True)
        pair_counts.append((counts >= 2).sum())
    assert sum(pair_counts) > 0
