"""Standalone deployed-score adaptation. No comparator imports or inputs."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

def slices(gid):
    import numpy as np
    if len(gid) == 0:
        return []
    a = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    return [slice(int(i), int(j)) for i, j in zip(a, np.r_[a[1:], len(gid)])]

def pairwise(s, y, groups, cap=.05):
    terms = []
    for sl in groups:
        yy, ss = y[sl], s[sl]
        i, j = torch.triu_indices(len(yy), len(yy), 1, device=s.device)
        gap = (yy[i] - yy[j]).clamp(-cap, cap)
        weight = gap.abs()
        if not len(i) or float(weight.sum()) == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(ss[i]-ss[j], (gap > 0).float(), reduction="none")
        terms.append((loss * weight).sum() / weight.sum())
    return torch.stack(terms).mean() if terms else s.sum() * 0

def utility(s, y, groups, scale):
    terms = []
    for sl in groups:
        yy = y[sl]
        if len(yy) < 2 or float(yy.max()-yy.min()) == 0:
            continue
        terms.append((yy.max() - (s[sl].softmax(0)*yy).sum()) / scale)
    return torch.stack(terms).mean() if terms else s.sum() * 0

def preserve(s, teacher, groups):
    """Group-mean Bernoulli KL on all pairs of OUR frozen source scores."""
    terms = []
    teacher = teacher.detach()
    for sl in groups:
        ss, tt = s[sl], teacher[sl]
        i, j = torch.triu_indices(len(ss), len(ss), 1, device=s.device)
        if not len(i):
            continue
        td = tt[i]-tt[j]
        prob = td.sigmoid()
        loss = F.binary_cross_entropy_with_logits(ss[i]-ss[j], prob, reduction="none")
        entropy = F.binary_cross_entropy_with_logits(td, prob, reduction="none")
        terms.append((loss-entropy).mean())
    return torch.stack(terms).mean() if terms else s.sum()*0

class Selector(nn.Module):
    def __init__(self, dim, residual=False, hidden=256):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(hidden, 1))
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, q):
        r = self.net(h).squeeze(-1)
        return q + r if self.residual else r
