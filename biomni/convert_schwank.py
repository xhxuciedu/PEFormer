"""
Convert Schwank lab data (PRIDICT v1 + PRIDICT2.0) to OptiPrime-format parquet.

Sources:
  PRIDICT v1:
    - Focused editing table (92,423 rows) → bar 5 (part): HEK PE2 tevo
    - Library2 Supp Table 6 (1,938 rows) → bars 4, 6-8, 10-12, 14-23
  PRIDICT2.0:
    - Supplementary Excel (22,956 rows) → bar 5 (part, HEK), bar 9 (K562 PE2), bar 13 (K562 PE4)

Output: /workspace/data/processed/schwank_combined.parquet
"""
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ── Helpers ──────────────────────────────────────────────────────────────
def revcomp_dna(seq: str) -> str:
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


def dna_to_rna(seq: str) -> str:
    return seq.replace('T', 'U')


def deterministic_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def parse_loc(val):
    """Parse location string like '[10, 29]' into (start, end) tuple."""
    if isinstance(val, str):
        loc = ast.literal_eval(val)
    else:
        loc = val
    return loc[0], loc[1]


SCAFFOLD_SP = ('GUUUAAGAGCUAAGCUGGAAACAGCAUAGCAAGUUUAAAUAAGGCUAGUCCGUUAUCAAC'
               'UUGAAAAAGUGGCACCGAGUCGGUGC')


def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add OptiPrime-derived columns (proto30, pegrna, split_edit, hashes)."""
    df = df.copy()
    df['unedited'] = 1.0 - (df['edited'] + df['indel'])
    df['proto30'] = df['full_unedited'].str.slice(0, 30)

    # T→U already done for spacer/rtt/pbs
    df['pegrna'] = df['spacer'] + SCAFFOLD_SP + df['rtt'] + df['pbs'] + df['linker']

    # split_edit
    def _split_edit(row):
        unedited = row['full_unedited']
        edited = row['full_edited']
        i = 0
        for i, (u, e) in enumerate(zip(unedited, edited)):
            if u != e:
                break
        pre_hom = unedited[:i]
        unedited_rem = unedited[i:]
        edited_rem = edited[i:]
        i = 0
        for i, (u, e) in enumerate(zip(unedited_rem[::-1], edited_rem[::-1])):
            if u != e:
                break
        else:
            i = i + 1
        if i:
            post_hom = unedited_rem[-i:]
            min_u = unedited_rem[:-i]
            min_e = edited_rem[:-i]
        else:
            post_hom = ''
            min_u = unedited_rem
            min_e = edited_rem
        return pre_hom, min_u + ':' + min_e, post_hom

    split_results = df.apply(_split_edit, axis=1, result_type='expand')
    df['pre_hom'] = split_results[0]
    df['min_edit'] = split_results[1]
    df['post_hom'] = split_results[2]

    # Hashes
    df['spacer_hash'] = df['spacer'].apply(deterministic_hash)
    df['pegrna_hash'] = df['pegrna'].apply(deterministic_hash)
    df['edit_hash'] = df['min_edit'].apply(deterministic_hash)

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply OptiPrime format_pe_df filters."""
    df = df[df['weight'] > 0]
    df = df.dropna(subset=['unedited', 'edited', 'weight'])
    df = df[df['proto30'].str.len() == 30]
    return df.reset_index(drop=True)


# ── PRIDICT2.0 conversion ────────────────────────────────────────────────

def convert_pridict2() -> pd.DataFrame:
    """Convert PRIDICT2.0 supplementary Excel to OptiPrime format."""
    xlsx_path = '/workspace/external/epridict_supp/SupplFile1_Library_Diverse_Editing_Results_with_test_splits.xlsx'
    df = pd.read_excel(xlsx_path, sheet_name='final_df_with_splits')
    print(f"  PRIDICT2.0: {len(df)} total rows")

    frames = []

    # HEK partition (bar 5 part)
    hek_mask = df['HEKaverageedited'].notna()
    hek_df = df[hek_mask].copy()
    print(f"  HEK non-null: {len(hek_df)}")
    hek_out = pd.DataFrame({
        'spacer': hek_df['spacer'].apply(dna_to_rna),
        'rtt': hek_df['RTT'].apply(dna_to_rna),
        'pbs': hek_df['PBS'].apply(dna_to_rna),
        'full_unedited': hek_df['wide_initial_target'].astype(str),
        'full_edited': hek_df['wide_mutated_target'].astype(str),
        'edited': hek_df['HEKaverageedited'].values / 100.0,
        'indel': hek_df.get('HEKaverageunintended', pd.Series(np.zeros(len(hek_df)))).fillna(0).values / 100.0,
        'weight': 1.0,
        'scaffold_name': 'SpCas9_OG',
        'motif': 'tevoPreQ1',
        'cas9_type': 'PE2-Cas9',
        'cas9_pam': 'SpNGG',
        'rt_name': 'PE2-RT',
        'pe_type': 'PE2',
        'group': 'Schwank_HEK293T',
        'cell_type': 'HEK293T',
        'time': 6.0,  # 7.0 - 1.0
        'linker': '',
        'split': hek_df['test_split_hek'].values,
        'bar_idx': 5,
        'source': 'PRIDICT2.0_HEK',
    })
    frames.append(hek_out)

    # K562 PE2 partition (bar 9 part)
    k562_mask = df['K562averageedited'].notna()
    k562_df = df[k562_mask].copy()
    print(f"  K562 non-null: {len(k562_df)}")
    # Remove duplicate spacer+PBS+RTT combinations (18 duplicates found in K562 data)
    dup_count = k562_df.duplicated(subset=['spacer', 'PBS', 'RTT']).sum()
    if dup_count > 0:
        print(f"  K562 duplicates (spacer+PBS+RTT): {dup_count}, removing")
        k562_df = k562_df.drop_duplicates(subset=['spacer', 'PBS', 'RTT'], keep='first')
        print(f"  K562 after dedup: {len(k562_df)}")
    k562_out = pd.DataFrame({
        'spacer': k562_df['spacer'].apply(dna_to_rna),
        'rtt': k562_df['RTT'].apply(dna_to_rna),
        'pbs': k562_df['PBS'].apply(dna_to_rna),
        'full_unedited': k562_df['wide_initial_target'].astype(str),
        'full_edited': k562_df['wide_mutated_target'].astype(str),
        'edited': k562_df['K562averageedited'].values / 100.0,
        'indel': k562_df.get('K562averageunintended', pd.Series(np.zeros(len(k562_df)))).fillna(0).values / 100.0,
        'weight': 1.0,
        'scaffold_name': 'SpCas9_OG',
        'motif': 'tevoPreQ1',
        'cas9_type': 'PE2-Cas9',
        'cas9_pam': 'SpNGG',
        'rt_name': 'PE2-RT',
        'pe_type': 'PE2',
        'group': 'Schwank_K562',
        'cell_type': 'K562',
        'time': 6.0,
        'linker': '',
        'split': k562_df['test_split_k562'].values,
        'bar_idx': 9,
        'source': 'PRIDICT2.0_K562',
    })
    frames.append(k562_out)

    combined = pd.concat(frames, ignore_index=True)
    combined = add_derived_cols(combined)
    combined = apply_filters(combined)
    return combined


# ── PRIDICT v1 focused editing conversion ────────────────────────────────

def convert_focused_editing() -> pd.DataFrame:
    """Convert PRIDICT v1 focused editing table to OptiPrime format (bar 5 part)."""
    csv_path = '/workspace/data/raw/pridict_extracted/focused_editing.csv'
    df = pd.read_csv(csv_path)
    print(f"  Focused editing: {len(df)} total rows")

    # All rows are HEK PE2 tevo
    # Derive sequences from wide targets and locations
    spacers = []
    pbs_list = []
    rtt_list = []

    for _, row in df.iterrows():
        wt = row['wide_initial_target']
        mt = row['wide_mutated_target']

        proto_start, proto_end = parse_loc(row['protospacerlocation_only_initial'])
        pbs_start, pbs_end = parse_loc(row['PBSlocation'])
        rtm_start, rtm_end = parse_loc(row['RT_mutated_location'])

        # Spacer: protospacer with 5' G (matching PRIDICT2.0 convention)
        proto = wt[proto_start:proto_end]
        spacer = 'G' + proto
        spacers.append(dna_to_rna(spacer))

        # PBS: revcomp of target strand upstream of nick
        pbs_seq = wt[pbs_start:pbs_end]
        pbs_list.append(dna_to_rna(revcomp_dna(pbs_seq)))

        # RTT: revcomp of edited target downstream of nick
        rtt_seq = mt[rtm_start:rtm_end]
        rtt_list.append(dna_to_rna(revcomp_dna(rtt_seq)))

    out = pd.DataFrame({
        'spacer': spacers,
        'rtt': rtt_list,
        'pbs': pbs_list,
        'full_unedited': df['wide_initial_target'].astype(str),
        'full_edited': df['wide_mutated_target'].astype(str),
        'edited': df['averageedited'].values / 100.0,
        'indel': df['averageindel'].values / 100.0,
        'weight': 1.0,
        'scaffold_name': 'SpCas9_OG',
        'motif': 'tevoPreQ1',
        'cas9_type': 'PE2-Cas9',
        'cas9_pam': 'SpNGG',
        'rt_name': 'PE2-RT',
        'pe_type': 'PE2',
        'group': 'Schwank_HEK293T',
        'cell_type': 'HEK293T',
        'time': 6.0,  # 7.0 - 1.0
        'linker': '',
        'split': np.nan,
        'bar_idx': 5,
        'source': 'PRIDICT_v1_focused',
    })

    out = add_derived_cols(out)
    out = apply_filters(out)
    return out


# ── PRIDICT v1 Library2 conversion ───────────────────────────────────────

# Library2 efficiency column → (bar_idx, group, cell_type, PEmax, epeg, MLH1dn, NRCH, pe_type, cas9_type)
LIB2_BARS = [
    # (efficiency_col, bar_idx, group, cell_type, pemax, epeg, mlh1dn, nrch, pe_type, cas9_type, is_largescreen)
    ('HEKOpti-Scaffold_PE2_averageedited',          4,  'Schwank_HEK293T', 'HEK293T', 0, 0, 0, 0, 'PE2', 'PE2-Cas9', False),
    ('HEKOpti-Scaffold_PE2-dnMLH1_averageedited',   6,  'Schwank_HEK293T', 'HEK293T', 0, 0, 1, 0, 'PE4', 'PE2-Cas9', False),
    ('HEKOpti-Scaffold_PE2-dnMLH1_averageedited',   7,  'Schwank_HEK293T', 'HEK293T', 0, 1, 1, 0, 'PE4', 'PE2-Cas9', False),
    ('K562_PE2_averageedited',                      8,  'Schwank_K562',    'K562',    0, 0, 0, 0, 'PE2', 'PE2-Cas9', False),
    ('K562_Pemax_averageedited',                   10,  'Schwank_K562',    'K562',    1, 0, 0, 0, 'PE2', 'PEmax-Cas9', False),
    ('K562_Pemax_averageedited',                   11,  'Schwank_K562',    'K562',    1, 1, 0, 0, 'PE2', 'PEmax-Cas9', False),
    ('K562_PE2-dnMLH1_averageedited',              12,  'Schwank_K562',    'K562',    0, 0, 1, 0, 'PE4', 'PE2-Cas9', False),
    ('K562_Pemax-dnMLH1_averageedited',            14,  'Schwank_K562',    'K562',    1, 0, 1, 0, 'PE4', 'PEmax-Cas9', False),
    ('K562_Pemax-dnMLH1_averageedited',            15,  'Schwank_K562',    'K562',    1, 1, 1, 0, 'PE4', 'PEmax-Cas9', False),
    ('K562_PE2-dnMLH1_averageedited',              13,  'Schwank_K562',    'K562',    0, 1, 1, 0, 'PE4', 'PE2-Cas9', False),
    ('U2OS_PE2_averageedited',                     16,  'Schwank_U2OS',    'U2OS',    0, 0, 0, 0, 'PE2', 'PE2-Cas9', False),
    ('U2OS_PE2_averageedited',                     17,  'Schwank_U2OS',    'U2OS',    0, 1, 0, 0, 'PE2', 'PE2-Cas9', False),
    ('U2OS_Pemax_averageedited',                   18,  'Schwank_U2OS',    'U2OS',    1, 0, 0, 0, 'PE2', 'PEmax-Cas9', False),
    ('U2OS_Pemax_averageedited',                   19,  'Schwank_U2OS',    'U2OS',    1, 1, 0, 0, 'PE2', 'PEmax-Cas9', False),
    ('U2OS_PE2-dnMLH1_averageedited',              20,  'Schwank_U2OS',    'U2OS',    0, 0, 1, 0, 'PE4', 'PE2-Cas9', False),
    ('U2OS_PE2-dnMLH1_averageedited',              21,  'Schwank_U2OS',    'U2OS',    0, 1, 1, 0, 'PE4', 'PE2-Cas9', False),
    ('U2OS_Pemax-dnMLH1_averageedited',            22,  'Schwank_U2OS',    'U2OS',    1, 0, 1, 0, 'PE4', 'PEmax-Cas9', False),
    ('U2OS_Pemax-dnMLH1_averageedited',            23,  'Schwank_U2OS',    'U2OS',    1, 1, 1, 0, 'PE4', 'PEmax-Cas9', False),
    # HEK PE2 tevo from Library2 (bar 5 part)
    ('HEKOpti-Scaffold_PE2_averageedited',          5,  'Schwank_HEK293T', 'HEK293T', 0, 1, 0, 0, 'PE2', 'PE2-Cas9', False),
    # K562 PE2 tevo from Library2 (bar 9 part)
    ('K562_PE2_averageedited',                      9,  'Schwank_K562',    'K562',    0, 1, 0, 0, 'PE2', 'PE2-Cas9', False),
]


def convert_library2() -> pd.DataFrame:
    """Convert PRIDICT v1 Library2 Supp Table 6 to OptiPrime format."""
    csv_path = '/workspace/external/pridict_supp/tables/Supplementary_Table_6_Editing_Efficiencies_Library2.csv'
    df = pd.read_csv(csv_path)
    print(f"  Library2: {len(df)} total rows, {df['tevopreq'].notna().sum()} with tevopreq")

    frames = []

    for eff_col, bar_idx, group, cell_type, pemax, epeg, mlh1dn, nrch, pe_type, cas9_type, is_largescreen in LIB2_BARS:
        if eff_col not in df.columns:
            print(f"    WARNING: {eff_col} not in Library2 columns, skipping bar {bar_idx}")
            continue

        # Determine tevo filter
        if epeg == 1:
            mask = df['tevopreq'] == True
        else:
            mask = df['tevopreq'] == False

        # Filter by non-null efficiency
        mask = mask & df[eff_col].notna()

        # Filter by uniqueindex_largescreen non-null — this removes library2-only designs
        # (not in library1/largescreen), giving exact matches for 16/18 small bars
        mask = mask & df['uniqueindex_largescreen'].notna()

        sub = df[mask].copy()
        if len(sub) == 0:
            print(f"    Bar {bar_idx}: 0 rows (mask empty)")
            continue

        # For largescreen, use largescreen efficiency (library1 HEK PE2 for overlapping designs)
        # The largescreen_averageedited column is present for all rows, but only non-null for overlapping designs

        # Derive sequences
        spacers = sub['protospacer'].apply(lambda s: dna_to_rna('G' + s) if s[0] != 'G' else dna_to_rna(s))
        pbs_list = sub['PBS'].apply(dna_to_rna)
        rtt_list = sub['RTT'].apply(dna_to_rna)

        # For indel, use the corresponding averageindel column if available
        indel_col = eff_col.replace('averageedited', 'averageindel')
        if indel_col in sub.columns:
            indel_vals = sub[indel_col].fillna(0).values / 100.0
        else:
            indel_vals = np.zeros(len(sub))

        out = pd.DataFrame({
            'spacer': spacers.values,
            'rtt': rtt_list.values,
            'pbs': pbs_list.values,
            'full_unedited': sub['wide_initial_target'].astype(str).values,
            'full_edited': sub['wide_mutated_target'].astype(str).values,
            'edited': sub[eff_col].values / 100.0,
            'indel': indel_vals,
            'weight': 1.0,
            'scaffold_name': 'SpCas9_OG',
            'motif': 'tevoPreQ1' if epeg else 'none',
            'cas9_type': cas9_type,
            'cas9_pam': 'SpNRCH' if nrch else 'SpNGG',
            'rt_name': 'PE2-RT',
            'pe_type': pe_type,
            'group': group,
            'cell_type': cell_type,
            'time': 6.0,  # 7.0 - 1.0
            'linker': '',
            'split': np.nan,
            'bar_idx': bar_idx,
            'source': f'Library2_{eff_col}',
        })
        frames.append(out)
        print(f"    Bar {bar_idx}: {len(sub)} rows ({'tevo' if epeg else 'non-tevo'}, {eff_col})")

    combined = pd.concat(frames, ignore_index=True)
    combined = add_derived_cols(combined)
    combined = apply_filters(combined)
    return combined


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    gt = pd.read_csv('/workspace/reports/ground_truth_partitions.csv')

    print("=== Converting PRIDICT v1 focused editing ===")
    fe_df = convert_focused_editing()
    print(f"  Focused editing output: {len(fe_df)} rows")

    print("\n=== Converting PRIDICT v1 Library2 ===")
    lib2_df = convert_library2()
    print(f"  Library2 output: {len(lib2_df)} rows")

    print("\n=== Converting PRIDICT2.0 ===")
    v2_df = convert_pridict2()
    print(f"  PRIDICT2.0 output: {len(v2_df)} rows")

    # Combine all Schwank data
    combined = pd.concat([fe_df, lib2_df, v2_df], ignore_index=True)
    print(f"\n=== Combined Schwank: {len(combined)} rows ===")

    # Per-bar verification
    print("\nPer-bar counts:")
    for bar_idx, grp in combined.groupby('bar_idx'):
        gt_size = gt.loc[gt['bar_idx'] == bar_idx, 'size'].values[0]
        status = "OK" if len(grp) == gt_size else f"excess {len(grp) - gt_size}"
        print(f"  Bar {bar_idx:2d}: {len(grp):6d} (target {gt_size:6d}) {status}")

    # Per-group counts
    print("\nPer-group counts:")
    for group, grp in combined.groupby('group'):
        print(f"  {group}: {len(grp)}")

    # Save
    out_path = Path('/workspace/data/processed/schwank_combined.parquet')
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    import subprocess
    subprocess.run(['cp', str(out_path), '/mnt/results/schwank_combined.parquet'], check=True)
    print("Copied to /mnt/results/schwank_combined.parquet")
    print("\nDONE.")


if __name__ == '__main__':
    main()
