"""Versioned follow-up helpers; historical v3 files are read-only dependencies."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'explore_v3'))
from common import CACHE, provenance, write_json, sha256
import numpy as np
import pandas as pd
import torch
from model import Selector, slices, pairwise, preserve, utility

def table(frame, score, source=False):
    f = frame.assign(selection=np.asarray(score)).sort_values(['group_id', 'design_key'])
    g = f.groupby('group_id', sort=True)
    t = g.agg(component=('component','first'), depth=('y','size'),
              oracle=('y','max'), minimum=('y','min'), study=('source_study','first'),
              cell=('cell_type','first'), editor=('pe_type','first'))
    top = f.loc[g.selection.idxmax()].set_index('group_id')
    t['achieved'], t['winner'] = top.y, top.design_key
    t['regret'] = t.oracle - t.achieved
    if source:
        t = t.loc[(t.depth >= 2) & (t.oracle > t.minimum)]
    return t

def source_domains(frame, score):
    t = table(frame, score, source=True)
    return {str(k): {'groups': len(g), 'achieved': float(g.achieved.mean())}
            for k, g in t.groupby('study')}

def feasible(domains, initial, margin=.001):
    if set(domains) != set(initial) or not initial:
        raise ValueError('Missing source validation domain')
    return all(domains[k]['achieved'] >= initial[k]['achieved'] - margin - 1e-12
               for k in initial)

def top_margin(s, teacher, groups, cap):
    terms = []
    for sl in groups:
        ss, tt = s[sl], teacher[sl].detach()
        if len(tt) < 2:
            continue
        best = tt.argmax()
        mask = torch.arange(len(tt), device=tt.device) != best
        wanted = (tt[best]-tt[mask]).clamp(min=0, max=cap)
        terms.append(torch.relu(wanted-(ss[best]-ss[mask])).mean())
    return torch.stack(terms).mean() if terms else s.sum()*0

def replay_groups(frame):
    # Shared eligibility for every source objective; no singleton training waste.
    out = {}
    for _, g in frame.loc[frame.surface == 'source_replay'].groupby('gid'):
        if len(g) >= 2 and g.y.max() > g.y.min():
            out.setdefault(str(g.source_study.iloc[0]), []).append(g.row.to_numpy())
    return out

def source_batch(pool, rng, balanced, max_rows=256):
    studies = sorted(pool)
    counts = np.array([len(pool[k]) for k in studies], dtype=float)
    probs = np.ones(len(studies))/len(studies) if balanced else counts/counts.sum()
    # Whole groups, sampled with replacement. Sort to restore contiguous groups;
    # avoid duplicate groups within one batch, so loss grouping remains exact.
    selected, seen, size = [], set(), 0
    for _ in range(max_rows * 10):
        study = studies[rng.choice(len(studies), p=probs)]
        g = pool[study][rng.integers(len(pool[study]))]
        if int(g[0]) in seen:
            continue
        if size + len(g) > max_rows:
            break
        selected.extend(g.tolist()); seen.add(int(g[0])); size += len(g)
        if size == max_rows:
            break
    if not selected:
        raise ValueError('No eligible whole source group fits batch')
    return np.sort(np.asarray(selected, dtype=int))

def norm(grads):
    return float(torch.sqrt(sum((g.detach().float().square().sum()
                                for g in grads if g is not None), torch.tensor(0., device='cuda'))))
