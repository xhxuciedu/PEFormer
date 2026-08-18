"""Attach round-2 Family C continuous features to a loaded corpus (task spec §9).

Normalization is fit from training-split rows only ("Use normalization based only on
training data") -- this cannot happen inside `featurize()`/`load_featurized()`, which
have no notion of which rows are "training" for a given run's fold configuration, so
it is done here, called explicitly by the training script after `train_idx` is known.

Missingness is tracked rather than silently imputed away (§9): a NaN feature value is
replaced by the training-set mean of the observed values in that column, and a
parallel 0/1 mask records which entries were imputed, so the model can learn to
discount them (see `FeatureBranch` in models/pe_rankformer.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import FeaturizedCorpus

FEATURE_COLS = [
    "pbs_length", "rtt_length", "pbs_gc", "rtt_gc", "extension_gc",
    "edit_length", "edit_position", "edit_position_from_nick", "n_mismatch",
    "pbs_tm", "rtt_tm", "proto_mfe", "rtt_mfe", "pbs_mfe", "extension_mfe",
    "ruleset3_score",
]


def attach_family_c_features(
    corpus: FeaturizedCorpus, features_path: str, train_idx: np.ndarray
) -> FeaturizedCorpus:
    feat_df = pd.read_parquet(features_path)
    order = pd.Series(np.arange(len(feat_df)), index=feat_df.record_id)
    aligned = order.reindex(corpus.record_id)
    if aligned.isna().any():
        missing = aligned.isna().sum()
        raise ValueError(f"{missing} corpus rows have no matching row in {features_path}")
    feat_df = feat_df.iloc[aligned.to_numpy(dtype=np.int64)].reset_index(drop=True)

    raw = feat_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    missing_mask = np.isnan(raw).astype(np.float32)

    train_mean = np.nanmean(raw[train_idx], axis=0)
    train_mean = np.nan_to_num(train_mean, nan=0.0)  # an all-NaN column in train -> impute 0
    imputed = np.where(np.isnan(raw), train_mean[None, :], raw)

    train_std = imputed[train_idx].std(axis=0)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)  # guard constant columns

    normalized = (imputed - train_mean[None, :]) / train_std[None, :]

    corpus.features = normalized.astype(np.float32)
    corpus.features_missing = missing_mask
    return corpus
