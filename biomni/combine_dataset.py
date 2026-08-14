"""
Combine all partitions into the full OptiPrime dataset and validate.

Inputs:
  - /workspace/data/processed/hsu2026_74769.parquet  (Liu, 74,769 rows)
  - /workspace/data/processed/schwank_combined.parquet (Schwank, ~153,878 rows)
  - /workspace/data/processed/kim_58301.parquet       (Kim, 58,301 rows)

Output:
  - /workspace/data/processed/optiprime_full_297962.parquet
"""
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

GROUND_TRUTH_CSV = '/workspace/reports/ground_truth_partitions.csv'


def load_model_group_factors():
    """Load group_factors from model weights to verify 12 groups."""
    pkl_path = '/workspace/external/optiprime/weights/model_5/log_rates/pe_on.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return set(data['group_factors'].keys())


def add_bar_idx_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add bar_idx and flag columns (PEmax, epegRNA, MLH1dn, NRCH) if missing."""
    df = df.copy()

    # Derive flags from existing columns
    if 'PEmax' not in df.columns:
        df['PEmax'] = (df['cas9_type'] == 'PEmax-Cas9').astype(int)
    if 'epegRNA' not in df.columns:
        df['epegRNA'] = (df['motif'] == 'tevoPreQ1').astype(int)
    if 'MLH1dn' not in df.columns:
        df['MLH1dn'] = (df['pe_type'] == 'PE4').astype(int)
    if 'NRCH' not in df.columns:
        df['NRCH'] = (df['cas9_pam'] == 'SpNRCH').astype(int)

    # Add bar_idx if missing
    if 'bar_idx' not in df.columns or df['bar_idx'].isna().all():
        # Derive from group + flags using ground truth mapping
        gt = pd.read_csv(GROUND_TRUTH_CSV)
        # Build lookup: (group, PEmax, epegRNA, MLH1dn, NRCH) -> bar_idx
        # But some bars share the same (group, flags) — need pe_type too
        # Use (group, PEmax, epegRNA, MLH1dn, NRCH, pe_type) as key
        gt['pe_type'] = gt['source'].apply(
            lambda s: 'PE4' if ('PE4' in s or 'dnMLH1' in s) else 'PE2'
        )
        lookup = {}
        for _, r in gt.iterrows():
            key = (r['group'], r['PEmax'], r['epegRNA'], r['MLH1dn'], r['NRCH'], r['pe_type'])
            lookup[key] = r['bar_idx']

        # Also try without pe_type (for bars that share flags)
        lookup_no_pt = {}
        for _, r in gt.iterrows():
            key = (r['group'], r['PEmax'], r['epegRNA'], r['MLH1dn'], r['NRCH'])
            if key not in lookup_no_pt:
                lookup_no_pt[key] = r['bar_idx']

        bar_indices = []
        for _, row in df.iterrows():
            key = (row['group'], row['PEmax'], row['epegRNA'], row['MLH1dn'], row['NRCH'], row['pe_type'])
            if key in lookup:
                bar_indices.append(lookup[key])
            else:
                key2 = (row['group'], row['PEmax'], row['epegRNA'], row['MLH1dn'], row['NRCH'])
                if key2 in lookup_no_pt:
                    bar_indices.append(lookup_no_pt[key2])
                else:
                    bar_indices.append(-1)
        df['bar_idx'] = bar_indices

    return df


def main():
    gt = pd.read_csv(GROUND_TRUTH_CSV)

    # ── Load all partitions ──────────────────────────────────────────────
    print("=== Loading partitions ===")
    liu = pd.read_parquet('/workspace/data/processed/hsu2026_74769.parquet')
    schwank = pd.read_parquet('/workspace/data/processed/schwank_combined.parquet')
    kim = pd.read_parquet('/workspace/data/processed/kim_58301.parquet')

    print(f"  Liu:    {len(liu):6d} rows")
    print(f"  Schwank:{len(schwank):6d} rows")
    print(f"  Kim:    {len(kim):6d} rows")
    print(f"  Total:  {len(liu) + len(schwank) + len(kim):6d} rows")
    print(f"  Target: 297962")
    print()

    # ── Add bar_idx and flags ────────────────────────────────────────────
    print("=== Adding bar_idx and flags ===")
    liu = add_bar_idx_and_flags(liu)
    schwank = add_bar_idx_and_flags(schwank)
    kim = add_bar_idx_and_flags(kim)

    # Check for unassigned bars
    for name, df in [('Liu', liu), ('Schwank', schwank), ('Kim', kim)]:
        unassigned = (df['bar_idx'] == -1).sum()
        if unassigned > 0:
            print(f"  {name}: {unassigned} rows with unassigned bar_idx")
    print()

    # ── Concatenate (use common columns) ─────────────────────────────────
    # Keep only columns that exist in all three
    common_cols = set(liu.columns) & set(schwank.columns) & set(kim.columns)
    # Ensure essential columns are included
    essential = ['spacer', 'rtt', 'pbs', 'full_unedited', 'full_edited', 'edited', 'indel',
                 'unedited', 'weight', 'scaffold_name', 'motif', 'cas9_type', 'cas9_pam',
                 'rt_name', 'pe_type', 'group', 'cell_type', 'time', 'linker', 'split',
                 'proto30', 'pegrna', 'pre_hom', 'min_edit', 'post_hom',
                 'spacer_hash', 'pegrna_hash', 'edit_hash',
                 'bar_idx', 'PEmax', 'epegRNA', 'MLH1dn', 'NRCH']
    cols_to_use = [c for c in essential if c in common_cols]
    # Add any extra common columns
    extra = common_cols - set(essential)
    cols_to_use += sorted(extra)

    combined = pd.concat([liu[cols_to_use], schwank[cols_to_use], kim[cols_to_use]],
                         ignore_index=True)
    print(f"  Combined: {len(combined)} rows, {len(cols_to_use)} columns")
    print()

    # ── Per-partition verification ───────────────────────────────────────
    print("=== Per-partition verification ===")
    all_match = True
    total_excess = 0
    total_deficit = 0
    exact_count = 0
    for _, gt_row in gt.iterrows():
        bar = gt_row['bar_idx']
        target = gt_row['size']
        actual = (combined['bar_idx'] == bar).sum()
        diff = actual - target
        if diff == 0:
            status = "OK"
            exact_count += 1
        elif diff > 0:
            status = f"excess +{diff}"
            total_excess += diff
            all_match = False
        else:
            status = f"deficit {diff}"
            total_deficit += abs(diff)
            all_match = False
        if diff != 0:
            print(f"  Bar {bar:2d}: {actual:6d} (target {target:6d}) {status}  [{gt_row['group']}]")

    print(f"\n  Exact matches: {exact_count}/42")
    print(f"  Total excess: {total_excess}")
    print(f"  Total deficit: {total_deficit}")
    print(f"  Net: {total_excess - total_deficit}")
    print(f"  Grand total: {len(combined)} (target 297962, diff {len(combined) - 297962})")
    print()

    # ── Group verification ───────────────────────────────────────────────
    print("=== Group verification ===")
    groups = combined['group'].unique()
    print(f"  Groups found: {len(groups)}")
    for g in sorted(groups):
        print(f"    {g}: {(combined['group'] == g).sum()}")
    print()

    # Cross-check with model weights
    model_groups = load_model_group_factors()
    data_groups = set(groups)
    print(f"  Model group_factors: {len(model_groups)} groups")
    print(f"  In data but not model: {data_groups - model_groups}")
    print(f"  In model but not data: {model_groups - data_groups}")
    print(f"  Match: {data_groups == model_groups}")
    print()

    # ── Context verification ─────────────────────────────────────────────
    print("=== Context verification ===")
    # Context = (cell_type, PEmax, epegRNA, MLH1dn, NRCH)
    contexts = combined.groupby(['cell_type', 'PEmax', 'epegRNA', 'MLH1dn', 'NRCH']).size()
    print(f"  Contexts (cell_type, PEmax, epeg, MLH1dn, NRCH): {len(contexts)}")

    # With scaffold
    contexts_scaffold = combined.groupby(['cell_type', 'PEmax', 'epegRNA', 'MLH1dn', 'NRCH', 'scaffold_name']).size()
    print(f"  Contexts (+scaffold): {len(contexts_scaffold)}")

    # With pe_type
    contexts_pe = combined.groupby(['cell_type', 'PEmax', 'epegRNA', 'MLH1dn', 'NRCH', 'pe_type']).size()
    print(f"  Contexts (+pe_type): {len(contexts_pe)}")

    # With scaffold + pe_type
    contexts_full = combined.groupby(['cell_type', 'PEmax', 'epegRNA', 'MLH1dn', 'NRCH', 'scaffold_name', 'pe_type']).size()
    print(f"  Contexts (+scaffold+pe_type): {len(contexts_full)}")
    print()

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path('/workspace/data/processed/optiprime_full_297962.parquet')
    combined.to_parquet(out_path, index=False)
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    subprocess.run(['cp', str(out_path), '/mnt/results/optiprime_full_297962.parquet'], check=True)
    print("Copied to /mnt/results/optiprime_full_297962.parquet")
    print("\nDONE.")


if __name__ == '__main__':
    main()
