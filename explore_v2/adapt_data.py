"""Candidate-level data for the adaptation experiments (ADAPTATION_SELECTION_PLAN.md §2).

One row per (decision group, distinct design). Repeated measurements of a design are
averaged before anything else happens, because they are measurements of one molecule and
not alternative choices -- the same rule the endpoint module enforces at evaluation time.

Group-complete batching is mandatory here: a group loss on a partial candidate list would
silently optimise the wrong decision. `group_batches` therefore packs whole groups and
never splits one, and `MAX_ROWS` is a cap on the packed batch rather than on the group.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, featurize  # noqa: E402

GEOM_COLS = ["pbs_len", "rtt_len", "rtt_overhang", "edit_len", "edit_pos_from_nick",
             "pbs_gc", "rtt_gc"]


NICK = 21  # protospacer begins at index 4 of the window; the nick is 17 nt into it


def _gc(s: pd.Series) -> np.ndarray:
    return np.array([(x.count("G") + x.count("C")) / max(len(x), 1) for x in s])


def _first_mismatch(a: pd.Series, b: pd.Series) -> pd.Series:
    """Index of the first differing base between the unedited and edited windows."""
    out = np.empty(len(a), dtype=np.int32)
    for k, (x, y) in enumerate(zip(a.to_numpy(), b.to_numpy())):
        n = min(len(x), len(y))
        i = 0
        while i < n and x[i] == y[i]:
            i += 1
        out[k] = i
    return pd.Series(out, index=a.index)


def load_candidates() -> pd.DataFrame:
    """Candidates joined to the E25 partition, with one representative record per design."""
    part = pd.read_parquet(C.CACHE / f"adaptation_partition_v{CANON_VERSION}.parquet")
    panel = pd.read_parquet(C.OUT / f"reserved_panel_v{CANON_VERSION}.parquet")
    # the representative row supplies sequence and context; all rows of a design share them
    rep = (panel.sort_values("record_id")
                .drop_duplicates(["edit_key", "design_key"], keep="first"))
    c = part.merge(rep.drop(columns=["edited_frac"]), on=["edit_key", "design_key"],
                   validate="1:1", suffixes=("", "_rep"))
    c["edited_frac"] = c.y
    c["pbs_len"] = c.pbs_dna.str.len()
    c["rtt_len"] = c.rtt_dna.str.len()
    # the RTT must span the edit; how much sequence it adds beyond the edit is the
    # geometry quantity OptiPrime models explicitly as a synthesis repeat count
    c["edit_len"] = c.edit_type.str[-1].astype(int)
    # Where the edit sits inside the RTT, read off the sequences rather than guessed. The
    # protospacer starts at index 4 of the reconstructed window and the nick falls 17 nt
    # into it, so NICK = 21; every observed first mismatch is at or after that index. The
    # 3' overhang is the homology the RTT adds beyond the edit, which is the quantity
    # OptiPrime treats as a synthesis repeat count.
    c["edit_pos_from_nick"] = (_first_mismatch(c.full_unedited, c.full_edited)
                               - NICK).clip(lower=0)
    c["rtt_overhang"] = (c.rtt_len - c.edit_pos_from_nick - c.edit_len).clip(lower=0)
    c["pbs_gc"] = _gc(c.pbs_dna)
    c["rtt_gc"] = _gc(c.rtt_dna)
    return c.sort_values(["split", "edit_key", "design_key"]).reset_index(drop=True)


def featurize_candidates(c: pd.DataFrame):
    d = c.copy()
    d["edited"] = d.edited_frac
    d["indel"] = 0.0
    d["fold"] = 0
    d["target_name"] = d.protospacer
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    return featurize(d, vocab)


def geometry_matrix(c: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    """Standardised geometry, with mean and SD fitted on the training split only."""
    raw = c[GEOM_COLS].to_numpy(dtype=np.float64)
    mu = raw[train_mask].mean(axis=0)
    sd = raw[train_mask].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((raw - mu) / sd).astype(np.float32)


def group_batches(gid: np.ndarray, max_rows: int, rng: np.random.Generator,
                  shuffle: bool = True) -> list[np.ndarray]:
    """Pack WHOLE groups into batches. A group is never split across batches."""
    order = np.argsort(gid, kind="stable")
    g = gid[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    blocks = [order[a:b] for a, b in zip(starts, ends)]
    if shuffle:
        rng.shuffle(blocks)
    out, cur, n = [], [], 0
    for b in blocks:
        if cur and n + len(b) > max_rows:
            out.append(np.concatenate(cur))
            cur, n = [], 0
        cur.append(b)
        n += len(b)
    if cur:
        out.append(np.concatenate(cur))
    return out


class Batcher:
    """Materialises model batches for a set of candidate indices."""

    def __init__(self, corpus, geom: np.ndarray, y: np.ndarray, gid_codes: np.ndarray):
        self.ds = PEDataset(corpus)
        self.geom = torch.from_numpy(geom)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.gid = gid_codes

    def make(self, idx: np.ndarray, device: str):
        # rows must be contiguous by group for the group losses to slice correctly
        idx = idx[np.argsort(self.gid[idx], kind="stable")]
        batch = {k: v.to(device, non_blocking=True)
                 for k, v in collate([self.ds[int(i)] for i in idx]).items()}
        return (batch, self.geom[idx].to(device), self.y[idx].to(device),
                self.gid[idx])
