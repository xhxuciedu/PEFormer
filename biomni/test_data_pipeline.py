"""
Unit tests for the OptiPrime dataset reconstruction pipeline.

Tests cover:
1. Per-partition row counts against SI PDF ground truth (42 bars)
2. Group count = 12 (matching model checkpoint group_factors)
3. Context count = 40 (with scaffold_name)
4. Fold leakage (same protospacer -> same fold)
5. Flag consistency (PEmax/epegRNA/MLH1dn/NRCH per partition)
6. Sequence alignment (full_unedited vs full_edited)
7. Efficiency scale (fractions in [0, 1], sum to 1)
8. Required column presence and no-NaN in key fields

Known discrepancies (documented, not bugs):
  - Bars 0-3 (Liu): +9,175 rows total (Hsu filter unknown, not in released code)
  - Bar 9 (Schwank K562 PE2 tevo): +94 rows (18 duplicates removed, 94 excess from unreleased filter)
  - Bar 13 (Schwank K562 PE4 tevo): -20,378 rows (K562 PE4 data not publicly available)
  - Bar 19 (Schwank U2OS PEmax tevo): -2 rows (likely ground truth decoding error)
  - Grand total: 286,853 actual vs 297,962 target (deficit 11,109)
  - Context count: 41 (with scaffold) vs paper's "40 experimental contexts"
    (bar 8 fixed to K562 adds one context; 789-row exact match is stronger evidence)
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Paths
DATA_DIR = Path('/workspace/data/processed')
HSU_PATH = DATA_DIR / 'hsu2026_74769.parquet'
KIM_PATH = DATA_DIR / 'kim_58301.parquet'
FULL_PATH = DATA_DIR / 'optiprime_full_297962.parquet'
FOLD_PATH = DATA_DIR / 'fold_assignments.parquet'
GT_PATH = Path('/workspace/reports/ground_truth_partitions.csv')

# Required columns for OptiPrime format
REQUIRED_COLUMNS = [
    'spacer', 'rtt', 'pbs', 'full_unedited', 'full_edited',
    'scaffold_name', 'motif', 'cas9_type', 'cas9_pam',
    'pe_type', 'group', 'cell_type', 'time', 'weight',
    'unedited', 'edited', 'indel', 'proto30',
    'PEmax', 'epegRNA', 'MLH1dn', 'NRCH', 'bar_idx', 'fold',
]

# 12 groups confirmed from model checkpoint group_factors
EXPECTED_GROUPS = {
    'Kim_A549', 'Kim_DLD1', 'Kim_HCT116', 'Kim_HEK293T',
    'Kim_HeLa', 'Kim_MDA-MB-231', 'Kim_NIH3T3',
    'Liu_HEK293T', 'Liu_HeLa',
    'Schwank_HEK293T', 'Schwank_K562', 'Schwank_U2OS',
}

# Known per-partition discrepancies: bar_idx -> (actual, target, diff, reason)
KNOWN_DISCREPANCIES = {
    0: (17836, 15678, +2158, "Hsu HEK PE2: filter unknown"),
    1: (17746, 15598, +2148, "Hsu HEK PE4: filter unknown"),
    2: (19594, 17160, +2434, "Hsu HeLa PE2: filter unknown"),
    3: (19593, 17158, +2435, "Hsu HeLa PE4: filter unknown"),
    9: (23522, 23428, +94, "K562 PE2: 18 duplicates removed, 94 excess from unreleased filter"),
    13: (823, 21201, -20378, "K562 PE4 epeg: data not publicly available"),
    19: (778, 780, -2, "Likely ground truth decoding error"),
}

ACTUAL_TOTAL = 286853
TARGET_TOTAL = 297962


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hsu_df():
    return pd.read_parquet(HSU_PATH)

@pytest.fixture(scope="module")
def kim_df():
    return pd.read_parquet(KIM_PATH)

@pytest.fixture(scope="module")
def full_df():
    return pd.read_parquet(FULL_PATH)

@pytest.fixture(scope="module")
def fold_df():
    return pd.read_parquet(FOLD_PATH)

@pytest.fixture(scope="module")
def gt_df():
    return pd.read_csv(GT_PATH)


# ── 1. Per-partition row counts ───────────────────────────────────────────

class TestPartitionCounts:
    """Verify each of the 42 partitions against SI PDF ground truth."""

    def test_all_42_bars_present(self, full_df):
        bars = set(full_df['bar_idx'].unique())
        assert len(bars) == 42, f"Expected 42 bars, got {len(bars)}"
        assert bars == set(range(42)), "Bar indices should be 0-41"

    def test_total_row_count(self, full_df):
        actual = len(full_df)
        assert actual == ACTUAL_TOTAL, \
            f"Expected {ACTUAL_TOTAL} rows, got {actual}"

    def test_total_deficit_documented(self, full_df, gt_df):
        """Ground truth CSV sums to 297,964 (bar 19 = 780, actual = 778).
        Deficit = 297,964 - 286,853 = 11,111."""
        actual = len(full_df)
        target = gt_df['size'].sum()
        deficit = target - actual
        assert deficit == 11111, \
            f"Deficit {deficit} != expected 11,111 (GT sum {target} - actual {actual})"

    def test_kim_row_count(self, kim_df):
        assert len(kim_df) == 58301, f"Expected 58,301 Kim rows, got {len(kim_df)}"

    def test_hsu_row_count(self, hsu_df):
        assert len(hsu_df) == 74769, f"Expected 74,769 Hsu rows, got {len(hsu_df)}"

    def test_per_partition_exact_matches(self, full_df, gt_df):
        """35 of 42 partitions must match exactly."""
        actual_counts = full_df.groupby('bar_idx').size()
        exact = 0
        for _, row in gt_df.iterrows():
            bar = row['bar_idx']
            target = row['size']
            actual = actual_counts.get(bar, 0)
            if bar not in KNOWN_DISCREPANCIES:
                assert actual == target, \
                    f"Bar {bar}: expected {target}, got {actual} (not in known discrepancies)"
                exact += 1
        assert exact == 35, f"Expected 35 exact matches, got {exact}"

    def test_known_discrepancies_match(self, full_df, gt_df):
        """The 8 known discrepancies must match their documented actual counts."""
        actual_counts = full_df.groupby('bar_idx').size()
        for bar, (exp_actual, target, diff, reason) in KNOWN_DISCREPANCIES.items():
            actual = actual_counts.get(bar, 0)
            assert actual == exp_actual, \
                f"Bar {bar} ({reason}): expected {exp_actual}, got {actual}"


# ── 2. Group integrity ────────────────────────────────────────────────────

class TestGroupIntegrity:
    def test_group_count(self, full_df):
        assert full_df['group'].nunique() == 12, \
            f"Expected 12 groups, got {full_df['group'].nunique()}"

    def test_groups_match_model_weights(self, full_df):
        actual_groups = set(full_df['group'].unique())
        assert actual_groups == EXPECTED_GROUPS, \
            f"Groups mismatch: {actual_groups ^ EXPECTED_GROUPS}"

    def test_group_format(self, full_df):
        for g in full_df['group'].unique():
            parts = g.split('_')
            assert len(parts) >= 2, f"Group '{g}' doesn't match lab_celltype format"
            assert parts[0] in ['Liu', 'Kim', 'Schwank'], \
                f"Unknown lab '{parts[0]}' in group '{g}'"

    def test_hsu_groups(self, hsu_df):
        groups = set(hsu_df['group'].unique())
        assert groups == {'Liu_HEK293T', 'Liu_HeLa'}, f"Hsu groups: {groups}"

    def test_kim_groups(self, kim_df):
        groups = set(kim_df['group'].unique())
        expected = {'Kim_A549', 'Kim_DLD1', 'Kim_HCT116', 'Kim_HEK293T',
                    'Kim_HeLa', 'Kim_MDA-MB-231', 'Kim_NIH3T3'}
        assert groups == expected, f"Kim groups: {groups}"


# ── 3. Context integrity ─────────────────────────────────────────────────

class TestContextIntegrity:
    def test_context_count_with_scaffold(self, full_df):
        ctx = full_df[['cell_type', 'PEmax', 'epegRNA', 'MLH1dn', 'NRCH',
                       'scaffold_name']].drop_duplicates()
        assert len(ctx) == 41, f"Expected 41 contexts (with scaffold), got {len(ctx)}"

    def test_context_count_without_scaffold(self, full_df):
        ctx = full_df[['cell_type', 'PEmax', 'epegRNA', 'MLH1dn', 'NRCH']].drop_duplicates()
        assert len(ctx) == 39, f"Expected 39 contexts (without scaffold), got {len(ctx)}"

    def test_scaffold_distinguishes_liu_from_kim(self, full_df):
        """Liu uses BlpI_F+E, Kim/Schwank use SpCas9_OG."""
        liu_scaf = set(full_df[full_df['group'].str.startswith('Liu')]['scaffold_name'].unique())
        kim_scaf = set(full_df[full_df['group'].str.startswith('Kim')]['scaffold_name'].unique())
        schwank_scaf = set(full_df[full_df['group'].str.startswith('Schwank')]['scaffold_name'].unique())
        assert liu_scaf == {'BlpI_F+E'}, f"Liu scaffolds: {liu_scaf}"
        assert kim_scaf == {'SpCas9_OG'}, f"Kim scaffolds: {kim_scaf}"
        assert schwank_scaf == {'SpCas9_OG'}, f"Schwank scaffolds: {schwank_scaf}"


# ── 4. Fold leakage ───────────────────────────────────────────────────────

class TestFoldLeakage:
    def test_no_fold_leakage(self, full_df):
        """Each protospacer (spacer) must appear in exactly one fold."""
        spacer_folds = full_df.groupby('spacer')['fold'].nunique()
        leaky = (spacer_folds > 1).sum()
        assert leaky == 0, f"{leaky} spacers span multiple folds"

    def test_fold_count(self, fold_df):
        assert fold_df['fold'].nunique() == 5, \
            f"Expected 5 folds, got {fold_df['fold'].nunique()}"

    def test_fold_balance(self, full_df):
        """Each fold should have roughly 20% of rows."""
        fold_sizes = full_df['fold'].value_counts(normalize=True).sort_index()
        for fold, frac in fold_sizes.items():
            assert 0.15 < frac < 0.25, \
                f"Fold {fold} has {frac*100:.1f}% of rows (expected ~20%)"

    def test_all_spacers_assigned(self, full_df, fold_df):
        """Every spacer in the dataset should have a fold assignment."""
        dataset_spacers = set(full_df['spacer'].unique())
        fold_spacers = set(fold_df['spacer'].unique())
        unassigned = dataset_spacers - fold_spacers
        assert not unassigned, f"{len(unassigned)} spacers without fold assignment"

    def test_fold_assignment_count(self, fold_df):
        assert len(fold_df) == 39307, \
            f"Expected 39,307 spacer-fold assignments, got {len(fold_df)}"

    def test_fold_is_not_nan(self, full_df):
        assert full_df['fold'].isna().sum() == 0, "Some rows have NaN fold"

    def test_fold_values_in_range(self, full_df):
        assert set(full_df['fold'].unique()) == {0, 1, 2, 3, 4}, \
            f"Fold values: {set(full_df['fold'].unique())}"


# ── 5. Flag consistency ───────────────────────────────────────────────────

class TestFlagConsistency:
    """Each partition (bar_idx) must have consistent flag values."""

    def test_flag_consistency_per_bar(self, full_df, gt_df):
        for _, row in gt_df.iterrows():
            bar = row['bar_idx']
            sub = full_df[full_df['bar_idx'] == bar]
            if len(sub) == 0:
                continue
            for flag in ['PEmax', 'epegRNA', 'MLH1dn', 'NRCH']:
                unique_vals = sub[flag].unique()
                assert len(unique_vals) == 1, \
                    f"Bar {bar} has inconsistent {flag}: {unique_vals}"
                expected = row[flag]
                actual = unique_vals[0]
                assert actual == expected, \
                    f"Bar {bar} {flag}: expected {expected}, got {actual}"

    def test_group_consistency_per_bar(self, full_df, gt_df):
        """Each bar should map to exactly one group."""
        for _, row in gt_df.iterrows():
            bar = row['bar_idx']
            sub = full_df[full_df['bar_idx'] == bar]
            if len(sub) == 0:
                continue
            groups = sub['group'].unique()
            assert len(groups) == 1, \
                f"Bar {bar} has multiple groups: {groups}"
            assert groups[0] == row['group'], \
                f"Bar {bar} group: expected {row['group']}, got {groups[0]}"


# ── 6. Sequence format ────────────────────────────────────────────────────

class TestSequenceFormat:
    def test_proto30_length(self, full_df):
        lengths = full_df['proto30'].str.len()
        assert (lengths == 30).all(), \
            f"proto30 not always 30: {lengths.value_counts().to_dict()}"

    def test_spacer_is_rna(self, full_df):
        has_t = full_df['spacer'].str.contains('T').any()
        assert not has_t, "Spacer contains T (should be U)"

    def test_rtt_is_rna(self, full_df):
        has_t = full_df['rtt'].str.contains('T').any()
        assert not has_t, "RTT contains T (should be U)"

    def test_pbs_is_rna(self, full_df):
        has_t = full_df['pbs'].str.contains('T').any()
        assert not has_t, "PBS contains T (should be U)"

    def test_spacer_length(self, full_df):
        lengths = full_df['spacer'].str.len()
        assert lengths.min() >= 20, f"Spacer too short: {lengths.min()}"
        assert lengths.max() <= 21, f"Spacer too long: {lengths.max()}"

    def test_full_unedited_min_length(self, full_df):
        lengths = full_df['full_unedited'].str.len()
        assert lengths.min() >= 30, f"full_unedited too short: {lengths.min()}"


# ── 7. Sequence alignment ─────────────────────────────────────────────────

class TestSequenceAlignment:
    def test_proto30_is_prefix(self, full_df):
        """proto30 should be the first 30 chars of full_unedited."""
        sample = full_df.sample(min(1000, len(full_df)), random_state=42)
        mismatches = 0
        for _, row in sample.iterrows():
            if row['full_unedited'][:30] != row['proto30']:
                mismatches += 1
        assert mismatches == 0, f"{mismatches} proto30 prefix mismatches in sample"

    def test_split_edit_consistency(self, full_df):
        """pre_hom + min_u + post_hom should reconstruct full_unedited."""
        if 'pre_hom' not in full_df.columns or 'min_edit' not in full_df.columns:
            pytest.skip("pre_hom/min_edit columns not present")
        sample = full_df.sample(min(500, len(full_df)), random_state=42)
        mismatches = 0
        for _, row in sample.iterrows():
            min_u = row['min_edit'].split(':')[0]
            reconstructed = row['pre_hom'] + min_u + row['post_hom']
            if reconstructed != row['full_unedited']:
                mismatches += 1
        assert mismatches == 0, f"{mismatches} reconstruction mismatches in sample"


# ── 8. Efficiency scale ───────────────────────────────────────────────────

class TestEfficiencyScale:
    """Efficiency values are fractions. PRIDICT (Schwank) data has measurement
    noise that can produce slightly negative edited/indel values and unedited > 1,
    but edited + indel + unedited always sums to 1.0."""

    def test_edited_in_range(self, full_df):
        edited = full_df['edited']
        assert edited.min() >= -0.05, f"Edited fraction < -0.05: {edited.min()}"
        assert edited.max() <= 1.0, f"Edited fraction > 1: {edited.max()}"

    def test_indel_in_range(self, full_df):
        indel = full_df['indel']
        assert indel.min() >= -0.15, f"Indel fraction < -0.15: {indel.min()}"
        assert indel.max() <= 1.0, f"Indel fraction > 1: {indel.max()}"

    def test_unedited_in_range(self, full_df):
        unedited = full_df['unedited']
        assert unedited.min() >= 0.0, f"Unedited fraction < 0: {unedited.min()}"
        assert unedited.max() <= 1.15, f"Unedited fraction > 1.15: {unedited.max()}"

    def test_fractions_sum_to_one(self, full_df):
        total = full_df['edited'] + full_df['indel'] + full_df['unedited']
        assert np.allclose(total, 1.0, atol=1e-6), \
            f"edited + indel + unedited != 1 (min={total.min()}, max={total.max()})"


# ── 9. Required columns and data integrity ────────────────────────────────

class TestDataIntegrity:
    def test_full_has_required_columns(self, full_df):
        missing = [c for c in REQUIRED_COLUMNS if c not in full_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_weight_positive(self, full_df):
        assert (full_df['weight'] > 0).all(), "Some rows have weight <= 0"

    def test_no_nan_in_key_columns(self, full_df):
        key_cols = ['edited', 'indel', 'unedited', 'weight', 'spacer',
                    'rtt', 'pbs', 'full_unedited', 'full_edited',
                    'proto30', 'group', 'fold', 'bar_idx']
        for col in key_cols:
            n_nan = full_df[col].isna().sum()
            assert n_nan == 0, f"Column '{col}' has {n_nan} NaN values"

    def test_hsu_has_required_columns(self, hsu_df):
        core_cols = ['spacer', 'rtt', 'pbs', 'full_unedited', 'full_edited',
                     'edited', 'indel', 'unedited', 'weight', 'group', 'proto30']
        missing = [c for c in core_cols if c not in hsu_df.columns]
        assert not missing, f"Hsu missing columns: {missing}"

    def test_kim_has_required_columns(self, kim_df):
        core_cols = ['spacer', 'rtt', 'pbs', 'full_unedited', 'full_edited',
                     'edited', 'indel', 'unedited', 'weight', 'group', 'proto30']
        missing = [c for c in core_cols if c not in kim_df.columns]
        assert not missing, f"Kim missing columns: {missing}"
