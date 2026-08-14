"""Dataset featurization: shapes, ranges, ranking-group correctness (task spec §40)."""

from __future__ import annotations

import pandas as pd
import torch

from pe_rankformer.data.context import ContextVocab
from pe_rankformer.data.dataset import (
    EDIT_MAX_LEN,
    PEG_MAX_LEN,
    PEDataset,
    collate,
    featurize,
    ranking_group_key,
)


def _toy_corpus(n=6):
    rows = []
    for i in range(n):
        rows.append(
            {
                "full_unedited": "ACGTACGTACGTACGT",
                "full_edited": "ACGTACGTACCTACGT" if i % 2 == 0 else "ACGTACGTACGTACGT",
                "spacer": "ACGTACGTACGTACGTACGT",
                "pbs": "ACGTACGTACGTA",
                "rtt": "ACGTACGTACGTACGTACGT",
                "cell_type": "HEK293T" if i < 3 else "HeLa",
                "pe_type": "PE2",
                "cas9_type": "PEmax-Cas9",
                "cas9_pam": "SpNGG",
                "scaffold_name": "BlpI_F+E",
                "motif": "tevoPreQ1",
                "source_study": "hsu2026",
                "edited": 0.1 * i,
                "fold": i % 5,
                "record_id": f"r{i}",
            }
        )
    return pd.DataFrame(rows)


def test_featurize_shapes():
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    corpus = featurize(df, vocab)
    assert corpus.edit_ids.shape == (len(df), EDIT_MAX_LEN + 2)
    assert corpus.peg_nuc_ids.shape == (len(df), PEG_MAX_LEN)
    assert corpus.peg_seg_ids.shape == (len(df), PEG_MAX_LEN)
    assert corpus.target.shape == (len(df),)


def test_target_in_unit_range():
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    corpus = featurize(df, vocab)
    assert (corpus.target >= 0).all() and (corpus.target <= 1).all()


def test_ranking_group_key_same_edit_context_matches():
    df = _toy_corpus()
    keys = ranking_group_key(df)
    # rows 0 and 2 share full_unedited/full_edited/cell_type/pe_type
    assert keys[0] == keys[2]


def test_ranking_group_key_differs_across_cell_type():
    df = _toy_corpus()
    keys = ranking_group_key(df)
    # row 0 (HEK293T) vs row 3 (HeLa) differ in cell_type even if edit matches
    assert keys[0] != keys[3] or df.loc[0, "full_edited"] != df.loc[3, "full_edited"]


def test_dataset_getitem_and_collate():
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    corpus = featurize(df, vocab)
    ds = PEDataset(corpus)
    batch = collate([ds[i] for i in range(4)])
    assert batch["edit_ids"].shape == (4, EDIT_MAX_LEN + 2)
    assert batch["target"].dtype == torch.float32
    assert "ctx_cell_type" in batch


def test_dataset_subset_by_indices():
    df = _toy_corpus()
    vocab = ContextVocab.fit(df)
    corpus = featurize(df, vocab)
    import numpy as np

    ds = PEDataset(corpus, indices=np.array([0, 2, 4]))
    assert len(ds) == 3
