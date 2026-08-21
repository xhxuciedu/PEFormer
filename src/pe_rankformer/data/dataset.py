"""PyTorch dataset for PE-RankFormer.

All tokenization is precomputed once into numpy arrays (`featurize`) rather than done
lazily per `__getitem__`, since the corpus is small enough (~260k rows, <100 tokens
each) to fit comfortably in memory and this keeps the training hot loop pure tensor
indexing with no Python-level string work.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .context import ContextVocab
from .tokenizer import encode_edit_pair, encode_pegrna

EDIT_MAX_LEN = 100  # core aligned-token budget; +2 for BOS/EOS
PEG_MAX_LEN = 90  # total budget including BOS/EOS


def ranking_group_key(df: pd.DataFrame) -> np.ndarray:
    """Hash (target/edit, experimental context) so different pegRNA designs for the
    same edit in the same context land in the same ranking group (task spec §20)."""
    keys = (
        df["full_unedited"].astype(str)
        + "|"
        + df["full_edited"].astype(str)
        + "|"
        + df["cell_type"].astype(str)
        + "|"
        + df["pe_type"].astype(str)
    )
    return np.array(
        [int(hashlib.sha256(k.encode()).hexdigest()[:15], 16) for k in keys], dtype=np.int64
    )


@dataclass
class FeaturizedCorpus:
    edit_ids: np.ndarray  # (N, EDIT_MAX_LEN+2) int16
    peg_nuc_ids: np.ndarray  # (N, PEG_MAX_LEN) int16
    peg_seg_ids: np.ndarray  # (N, PEG_MAX_LEN) int8
    context_ids: dict[str, np.ndarray]  # each (N,) int16
    target: np.ndarray  # (N,) float32, editing efficiency in [0,1]
    target_indel: np.ndarray  # (N,) float32, indel rate in [0,1]
    group_key: np.ndarray  # (N,) int64
    fold: np.ndarray  # (N,) int8
    record_id: np.ndarray  # (N,) object (str)
    # Round-2 Family C (§9): attached post-hoc by the training script, not by
    # `featurize()`/`load_featurized()`, since normalization must be fit from
    # training-split rows only -- a concern the corpus cache has no notion of.
    # See scripts/train/attach_family_c_features.py.
    features: np.ndarray | None = None  # (N, F) float32, normalized + NaN-imputed
    features_missing: np.ndarray | None = None  # (N, F) float32, 1.0 where imputed
    # Round-5 §8: each row's efficiency quantile *within its experimental context*,
    # computed from training rows only. Attached by the training script rather than by
    # featurize(), since it depends on the train/val split.
    target_ctx_q: np.ndarray | None = None  # (N,) float32 in [0,1]
    # Round-6 Lead 1: per-row loss weight, used to correct the train/eval source
    # mismatch (58.4% of training rows are Schwank, which is 0% of the evaluation set).
    sample_weight: np.ndarray | None = None  # (N,) float32

    def __len__(self) -> int:
        return len(self.target)


def featurize(df: pd.DataFrame, vocab: ContextVocab) -> FeaturizedCorpus:
    n = len(df)
    edit_ids = np.empty((n, EDIT_MAX_LEN + 2), dtype=np.int16)
    peg_nuc_ids = np.empty((n, PEG_MAX_LEN), dtype=np.int16)
    peg_seg_ids = np.empty((n, PEG_MAX_LEN), dtype=np.int8)

    for i, (wt, ed, sp, pb, rt) in enumerate(
        zip(df.full_unedited.values, df.full_edited.values, df.spacer.values, df.pbs.values, df.rtt.values)
    ):
        ids, _ = encode_edit_pair(wt, ed, EDIT_MAX_LEN)
        edit_ids[i] = ids
        peg = encode_pegrna(sp, pb, rt, PEG_MAX_LEN)
        peg_nuc_ids[i] = peg.nuc_ids
        peg_seg_ids[i] = peg.segment_ids

    context_ids: dict[str, np.ndarray] = {f_: np.empty(n, dtype=np.int16) for f_ in vocab.fields}
    for f_ in vocab.fields:
        vlist = vocab.vocabs[f_]
        index = {v: i for i, v in enumerate(vlist)}
        col = df[f_].astype(str)
        context_ids[f_] = col.map(lambda s: index.get(s, 0)).to_numpy(dtype=np.int16)

    target = df["edited"].to_numpy(dtype=np.float32)
    target_indel = df["indel"].to_numpy(dtype=np.float32) if "indel" in df.columns else np.zeros(n, dtype=np.float32)
    group_key = ranking_group_key(df)
    fold = df["fold"].to_numpy(dtype=np.int8)
    record_id = df["record_id"].to_numpy(dtype=object)

    return FeaturizedCorpus(
        edit_ids=edit_ids,
        peg_nuc_ids=peg_nuc_ids,
        peg_seg_ids=peg_seg_ids,
        context_ids=context_ids,
        target=target,
        target_indel=target_indel,
        group_key=group_key,
        fold=fold,
        record_id=record_id,
    )


def load_featurized(npz_path: str, vocab: ContextVocab) -> FeaturizedCorpus:
    """Reload a corpus cached by scripts/data/featurize_corpus.py."""
    data = np.load(npz_path, allow_pickle=True)
    context_ids = {f_: data[f"ctx_{f_}"] for f_ in vocab.fields}
    return FeaturizedCorpus(
        edit_ids=data["edit_ids"],
        peg_nuc_ids=data["peg_nuc_ids"],
        peg_seg_ids=data["peg_seg_ids"],
        context_ids=context_ids,
        target=data["target"],
        target_indel=data["target_indel"] if "target_indel" in data else np.zeros_like(data["target"]),
        group_key=data["group_key"],
        fold=data["fold"],
        record_id=data["record_id"],
    )


class PEDataset(Dataset):
    """Indexes into a FeaturizedCorpus, optionally restricted to a subset of rows."""

    def __init__(self, corpus: FeaturizedCorpus, indices: np.ndarray | None = None):
        self.corpus = corpus
        self.indices = np.arange(len(corpus)) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = self.indices[i]
        c = self.corpus
        item = {
            "edit_ids": torch.from_numpy(c.edit_ids[j].astype(np.int64)),
            "peg_nuc_ids": torch.from_numpy(c.peg_nuc_ids[j].astype(np.int64)),
            "peg_seg_ids": torch.from_numpy(c.peg_seg_ids[j].astype(np.int64)),
            "target": torch.tensor(c.target[j], dtype=torch.float32),
            "target_indel": torch.tensor(c.target_indel[j], dtype=torch.float32),
            "group_key": torch.tensor(c.group_key[j], dtype=torch.int64),
        }
        if c.target_ctx_q is not None:
            item["target_ctx_q"] = torch.tensor(c.target_ctx_q[j], dtype=torch.float32)
        if c.sample_weight is not None:
            item["sample_weight"] = torch.tensor(c.sample_weight[j], dtype=torch.float32)
        if c.features is not None:
            item["features"] = torch.from_numpy(c.features[j].astype(np.float32))
            item["features_missing"] = torch.from_numpy(c.features_missing[j].astype(np.float32))
        for f_, arr in c.context_ids.items():
            item[f"ctx_{f_}"] = torch.tensor(int(arr[j]), dtype=torch.int64)
        return item


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
