"""
Construct protospacer-disjoint fivefold cross-validation assignments.

Uses the same deterministic_hash as OptiPrime source code:
    sha256(s.encode('ascii')).hexdigest()[:10]

Fold assignment: int(hash, 16) % 5
This guarantees: same spacer -> same hash -> same fold (protospacer-disjoint
by construction).

The original OptiPrime `split` column is preserved separately; these fold
assignments are for PE-RankFormer training and evaluation.

Output:
    data/processed/fold_assignments.parquet  (spacer, spacer_hash_op, fold)
    Updated optiprime_full_297962.parquet with `fold` column
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# Match OptiPrime source code: scripts/utils.py deterministic_hash
HASH_SIZE = 10  # from scripts/constants.py


def deterministic_hash(s: str, length: int = HASH_SIZE) -> str:
    """SHA256 hash truncated to `length` hex chars. Matches OptiPrime source."""
    return hashlib.sha256(s.encode('ascii')).hexdigest()[:length]


def main():
    data_path = Path('/workspace/data/processed/optiprime_full_297962.parquet')
    out_path = Path('/workspace/data/processed/fold_assignments.parquet')

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows, {df['spacer'].nunique()} unique spacers")

    # --- Compute OptiPrime-compatible hash on spacer ---
    unique_spacers = df['spacer'].unique()
    spacer_to_hash = {s: deterministic_hash(s) for s in unique_spacers}
    spacer_to_fold = {s: int(spacer_to_hash[s], 16) % 5 for s in unique_spacers}

    df['spacer_hash_op'] = df['spacer'].map(spacer_to_hash)
    df['fold'] = df['spacer'].map(spacer_to_fold)

    # --- Verify protospacer-disjointness ---
    fold_check = df.groupby('spacer')['fold'].nunique()
    assert (fold_check == 1).all(), \
        f"{(fold_check > 1).sum()} spacers span multiple folds!"
    print(f"Verification passed: all {len(fold_check)} spacers in exactly one fold")

    # --- Fold balance ---
    print(f"\nFold distribution (rows):")
    for f in range(5):
        n = (df['fold'] == f).sum()
        n_spacers = df[df['fold'] == f]['spacer'].nunique()
        print(f"  Fold {f}: {n:>7} rows ({n/len(df)*100:5.1f}%), {n_spacers:>6} spacers")

    # --- Per-group per-fold distribution ---
    print(f"\nPer-group fold distribution (rows):")
    gf = df.groupby(['group', 'fold']).size().unstack(fill_value=0)
    print(gf.to_string())

    # --- Check overlap with original OptiPrime split ---
    has_split = df[df['split'].notna()]
    if len(has_split) > 0:
        overlap = has_split.groupby(['split', 'fold']).size().unstack(fill_value=0)
        print(f"\nOriginal split vs new fold (rows with original split only, {len(has_split)} rows):")
        print(overlap.to_string())

    # --- Save fold assignments (one row per unique spacer) ---
    fold_df = (
        df[['spacer', 'spacer_hash_op', 'fold']]
        .drop_duplicates(subset=['spacer'])
        .sort_values('spacer_hash_op')
        .reset_index(drop=True)
    )
    fold_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(fold_df)} spacer-fold assignments to {out_path}")

    # --- Update full dataset with fold column ---
    df.to_parquet(data_path, index=False)
    print(f"Updated full dataset with `fold` column at {data_path}")

    # --- Copy to /mnt/results ---
    import shutil
    shutil.copy(out_path, '/mnt/results/fold_assignments.parquet')
    shutil.copy(data_path, '/mnt/results/optiprime_full_297962.parquet')
    print("Copied to /mnt/results/")


if __name__ == '__main__':
    main()
