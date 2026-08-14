"""Within-target ranking-pair sampling (task spec §21).

Two pieces:
  1. `sample_ranking_pairs` -- given a batch's group keys and targets, returns index
     pairs (i, j) with target[i] > target[j] by at least `min_diff`, capped per group so
     a handful of very large groups can't dominate the loss.
  2. `GroupedBatchSampler` -- a batch sampler that deliberately clusters same-group rows
     into the same batch (uniform random batching would rarely put enough same-target
     rows together for the pair sampler to find anything, since most ranking groups are
     small relative to a 512-1024 row batch).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


def sample_ranking_pairs(
    group_key: torch.Tensor,
    target: torch.Tensor,
    min_diff: float = 0.02,
    max_pairs_per_group: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample informative within-group pairs from one batch.

    Returns (pairs_i, pairs_j), local batch indices, oriented so target[i] > target[j].
    """
    device = group_key.device
    order = torch.argsort(group_key)
    gk_sorted = group_key[order]
    # boundaries between consecutive distinct groups
    change = torch.ones_like(gk_sorted, dtype=torch.bool)
    change[1:] = gk_sorted[1:] != gk_sorted[:-1]
    group_starts = torch.nonzero(change, as_tuple=True)[0]
    group_ends = torch.cat([group_starts[1:], torch.tensor([len(gk_sorted)], device=device)])

    all_i: list[torch.Tensor] = []
    all_j: list[torch.Tensor] = []
    for start, end in zip(group_starts.tolist(), group_ends.tolist()):
        size = end - start
        if size < 2:
            continue
        members = order[start:end]
        t = target[members]
        # all pairwise diffs within this (typically small) group
        diff = t[:, None] - t[None, :]
        eligible = torch.nonzero(diff >= min_diff, as_tuple=False)  # (k, 2) -> (row=i idx, col=j idx)
        if eligible.numel() == 0:
            continue
        if eligible.size(0) > max_pairs_per_group:
            if generator is not None:
                perm = torch.randperm(eligible.size(0), generator=generator)[:max_pairs_per_group]
            else:
                perm = torch.randperm(eligible.size(0))[:max_pairs_per_group]
            eligible = eligible[perm]
        all_i.append(members[eligible[:, 0]])
        all_j.append(members[eligible[:, 1]])

    if not all_i:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(all_i), torch.cat(all_j)


class GroupedBatchSampler(Sampler[list[int]]):
    """Batches biased toward containing multiple rows from the same ranking group.

    Each batch is filled by repeatedly popping a whole group (up to `max_group_take`
    members) from a shuffled group order, then topping up with individually shuffled
    leftover rows, until `batch_size` is reached. Reproducible given `seed`; a fresh
    shuffle is used each epoch via `set_epoch`.
    """

    def __init__(
        self,
        group_key: np.ndarray,
        batch_size: int,
        max_group_take: int = 8,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.max_group_take = max_group_take
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        order = np.argsort(group_key, kind="stable")
        gk_sorted = group_key[order]
        boundaries = np.nonzero(np.diff(gk_sorted) != 0)[0] + 1
        self.groups: list[np.ndarray] = np.split(order, boundaries)
        self.n = len(group_key)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = rng.permutation(len(self.groups))
        pool: list[np.ndarray] = [self.groups[g].copy() for g in group_order]
        for members in pool:
            rng.shuffle(members)

        batch: list[int] = []
        gi = 0
        while gi < len(pool):
            take = pool[gi][: self.max_group_take]
            pool[gi] = pool[gi][self.max_group_take :]
            batch.extend(int(x) for x in take)
            if len(pool[gi]) == 0:
                gi += 1
            while len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        return self.n // self.batch_size if self.drop_last else -(-self.n // self.batch_size)
