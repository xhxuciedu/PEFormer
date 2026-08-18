"""Round-2 Family C feature attachment: normalization-from-train-only, missingness tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pe_rankformer.data.context import ContextVocab
from pe_rankformer.data.dataset import featurize
from pe_rankformer.data.family_c_features import FEATURE_COLS, attach_family_c_features


def _toy_corpus_and_features(tmp_path):
    rows = []
    for i in range(10):
        rows.append(
            {
                "full_unedited": "ACGTACGTACGTACGTACGT",
                "full_edited": "ACGTACGTACCTACGTACGT" if i % 2 == 0 else "ACGTACGTACGTACGTACGT",
                "spacer": "ACGTACGTACGTACGTACGT",
                "pbs": "ACGTACGTACGTA",
                "rtt": "ACGTACGTACGTACGTACGT",
                "cell_type": "HEK293T",
                "pe_type": "PE2",
                "cas9_type": "PEmax-Cas9",
                "cas9_pam": "SpNGG",
                "scaffold_name": "BlpI_F+E",
                "motif": "tevoPreQ1",
                "source_study": "hsu2026",
                "edited": (i % 7) / 7,
                "fold": i % 5,
                "record_id": f"r{i}",
            }
        )
    df = pd.DataFrame(rows)
    vocab = ContextVocab.fit(df)
    corpus = featurize(df, vocab)

    rng = np.random.default_rng(0)
    feat_df = pd.DataFrame({"record_id": df.record_id.values})
    for c in FEATURE_COLS:
        feat_df[c] = rng.normal(loc=10.0, scale=2.0, size=len(df))
    # Row 0 is a training row (fold 2); make one of its features unobserved.
    feat_df.loc[0, "ruleset3_score"] = np.nan
    path = tmp_path / "features.parquet"
    feat_df.to_parquet(path)
    return corpus, path


def test_normalization_uses_only_train_rows(tmp_path):
    corpus, path = _toy_corpus_and_features(tmp_path)
    train_idx = np.array([2, 3, 4, 5, 6, 7])  # excludes rows 0,1,8,9
    val_idx = np.array([0, 1, 8, 9])

    corpus = attach_family_c_features(corpus, str(path), train_idx)

    col = FEATURE_COLS.index("pbs_length")
    train_vals = corpus.features[train_idx, col]
    # z-scored against its own sample -> mean ~0, std ~1 on the training rows themselves.
    assert abs(train_vals.mean()) < 1e-4
    assert abs(train_vals.std() - 1.0) < 1e-4
    # Validation rows are transformed with train statistics, not their own -- no such
    # guarantee holds for them.
    assert corpus.features[val_idx, col].std() != pytest.approx(1.0, abs=1e-4)


def test_missingness_mask_flags_imputed_entries(tmp_path):
    corpus, path = _toy_corpus_and_features(tmp_path)
    train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    corpus = attach_family_c_features(corpus, str(path), train_idx)

    col = FEATURE_COLS.index("ruleset3_score")
    assert corpus.features_missing[0, col] == 1.0
    assert corpus.features_missing[1:, col].sum() == 0.0
    # Imputed value equals the (normalized) train-mean of the observed entries, i.e. 0.
    assert abs(corpus.features[0, col]) < 1e-4


def test_raises_on_record_id_mismatch(tmp_path):
    corpus, path = _toy_corpus_and_features(tmp_path)
    feat_df = pd.read_parquet(path)
    feat_df = feat_df[feat_df.record_id != "r3"]  # drop one row's features
    feat_df.to_parquet(path)
    with pytest.raises(ValueError):
        attach_family_c_features(corpus, str(path), np.array([0, 1, 2]))
