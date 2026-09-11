"""Frozen Family C feature standardisation.

`pe_rankformer.data.family_c_features.attach_family_c_features` refits the per-column mean
and standard deviation every time it is called, from whichever rows are passed as
`train_idx`. The statistics are therefore a property of the call, not of the checkpoint, and
they are not stored in the checkpoint file.

That is correct during training, where `train_idx` is the run's own training split. It is
wrong at evaluation time on a different surface: standardising the reserved panel by the
panel's own mean and standard deviation rescales the inputs away from the scale the model was
trained to read, and erases exactly the distribution shift the evaluation exists to measure.
On this corpus the effect is not cosmetic -- panel `pbs_length` sits 1.35 training SDs below
the training mean with twice the spread, and `pbs_tm` 0.86 SDs below with 2.2x the spread.

These helpers recover a checkpoint's training statistics from its training folds and apply
them, frozen, to any surface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pe_rankformer.data.family_c_features import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
DEV_FEATURES = ROOT / "data/processed/family_c_features.parquet"


def train_stats(corpus_parquet: Path, train_folds, features_path: Path = DEV_FEATURES):
    """Mean and SD over the development rows a checkpoint actually trained on.

    Reproduces `attach_family_c_features` exactly: NaN-mean first, impute, then SD of the
    imputed column, with constant columns guarded to 1.
    """
    fold = pd.read_parquet(corpus_parquet, columns=["record_id", "fold"])
    feat = pd.read_parquet(features_path)
    d = feat.merge(fold, on="record_id", validate="1:1")
    raw = d.loc[d.fold.isin(list(train_folds)), FEATURE_COLS].to_numpy(dtype=np.float64)
    mean = np.nan_to_num(np.nanmean(raw, axis=0), nan=0.0)
    imputed = np.where(np.isnan(raw), mean[None, :], raw)
    sd = imputed.std(axis=0)
    return mean, np.where(sd < 1e-6, 1.0, sd)


def attach_frozen(corpus, features_frame: pd.DataFrame, mean: np.ndarray, sd: np.ndarray):
    """Attach features to `corpus`, standardised by the supplied frozen statistics."""
    order = pd.Series(np.arange(len(features_frame)), index=features_frame.record_id)
    aligned = order.reindex(corpus.record_id)
    if aligned.isna().any():
        raise ValueError(f"{int(aligned.isna().sum())} corpus rows have no feature row")
    f = features_frame.iloc[aligned.to_numpy(dtype=np.int64)].reset_index(drop=True)
    raw = f[FEATURE_COLS].to_numpy(dtype=np.float64)
    corpus.features_missing = np.isnan(raw).astype(np.float32)
    imputed = np.where(np.isnan(raw), mean[None, :], raw)
    corpus.features = ((imputed - mean[None, :]) / sd[None, :]).astype(np.float32)
    return corpus
