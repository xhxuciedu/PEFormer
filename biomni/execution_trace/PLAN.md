# Plan: Fix 5 Dataset Reconstruction Issues (v3)

## Summary

Fix 5 issues identified in the independent audit of the OptiPrime dataset reconstruction. The fixes target: (1) bar 8 group misassignment, (2) bar 9 excess rows, (3) bar 13 missing data investigation, (4) Hsu 9,175-row excess investigation, and (5) report QA regression. After fixes, rebuild the combined dataset, fold assignments, unit tests, and validation report.

---

## Issue 1: Bar 8 — K562 PE2 non-tevo (target 789) misassigned as HEK293T

### Root Cause
`LIB2_BARS` in `convert_schwank.py` maps bar 8 to `largescreen_averageedited` (866 rows, HEK293T). The correct source is `K562_PE2_averageedited` with `tevopreq=False` and `uniqueindex_largescreen` non-null = exactly 789 rows, group `Schwank_K562`.

### Fix
- Change the bar 8 entry in `LIB2_BARS` from:
  ```python
  ('largescreen_averageedited', 8, 'Schwank_HEK293T', 'HEK293T', 0, 0, 0, 0, 'PE2', 'PE2-Cas9', True)
  ```
  to:
  ```python
  ('K562_PE2_averageedited', 8, 'Schwank_K562', 'K562', 0, 0, 0, 0, 'PE2', 'PE2-Cas9', False)
  ```
- Re-run `convert_schwank.py` to rebuild `schwank_combined.parquet`
- **Context count implication**: 41 contexts (with scaffold) vs paper's "40 experimental contexts." Documented as unresolved — the 789-row exact match and the pattern (3/4 sibling K562 non-tevo bars already match with the same filter) are stronger evidence than the paper's context count claim.

### Verification
- Bar 8 count = 789 (exact match)
- Bar 8 group = Schwank_K562, cell_type = K562
- 3 sibling K562 non-tevo bars (10, 12, 14) still match exactly
- `largescreen_averageedited` column no longer used for any bar

---

## Issue 2: Bar 9 — K562 PRIDICT2.0 112-row excess

### Findings
- PRIDICT2.0 K562 has 22,752 non-null `K562averageedited` (from Excel)
- `k562_indices_nan.pkl` = 204 indices = exactly the 204 NaN rows (not additional QC)
- 18 duplicate spacer+PBS+RTT combinations in K562 data
- Needed from PRIDICT2.0: 23,428 - 788 (Library2 K562 PE2 tevo) = 22,640
- Excess: 22,752 - 22,640 = 112
- Removing 18 duplicates: 22,752 - 18 = 22,734, still 94 excess

### Fix
- Apply duplicate removal (same spacer+PBS+RTT) to PRIDICT2.0 K562 data: removes 18 rows
- Remaining 94-row excess is from an unreleased filter — document and accept
- Update `convert_schwank.py` to drop duplicates before counting

### Verification
- Bar 9 count = 23,540 - 18 = 23,522 (was 23,540, now 94 excess vs 112)
- Document remaining 94-row excess as unreleased filter

---

## Issue 3: Bar 13 — K562 PE4 epegRNA (target 21,201) only 823 rows

### Investigation Results (all leads checked)
- **4 unopened PRIDICT zip files**: Extracted and inspected columns — none have K562 or PE4/dnMLH1 columns
- **Library2 alone**: Only 1,938 rows total; `K562_PE2-dnMLH1_averageedited` with `tevopreq=True` + largescreen index = 823 rows (current source)
- **PRIDICT2.0 Ranking_Percentile.csv**: 24,134 rows, only HEK/K562 averageedited + percentile columns, no PE4 data
- **PRIDICT2.0 supplementary Excel**: All 22,956 rows are "PE2-NGG" (no PE4)
- **Main paper + SI PDF**: No mention of coverage filter, QC threshold, or additional K562 PE4 data

### Conclusion
K562 PE4 epegRNA data (21,201 rows) is not in any publicly available source. The OptiPrime authors likely conducted an unpublished K562 PE4 experiment. Keep 823 rows and document the 20,378-row shortfall as "not found despite exhausting all available sources."

### Verification
- Bar 13 count = 823 (unchanged)
- Document specific files checked

---

## Issue 4: Hsu 9,175-row excess (bars 0-3)

### Investigation Results
- **pe_constants.py bounds** (MAX_PRE_HOM=25, MAX_POST_HOM=25, MAX_LARGE_SUB=20): Tested on actual homology arm lengths — would remove 72,135 of 74,769 rows. Far too aggressive; these constants are for synthetic data generation, not real data filtering.
- **Design category grid search** (LibMMR only, 10,000 rows):
  - Excluding Indel + Endogenous: 67,393 total (excess 1,799)
  - Excluding Indel + Endogenous + PAM SNP+indel: 64,465 (deficit 1,129)
  - Target 65,594 is between these — no clean Design category combination works
- **Per-partition excess**: ~12.1% for HEK293T, ~12.4% for HeLa — proportional, not fixed count
- **Additional MOESM files**: Only MOESM1 (PDF) and MOESM3 (Excel) available; no MOESM2/4+
- **Source code**: `format_pe_df` only filters weight>0 and dropna; `process_liu` has no filter
- **SI PDF + paper**: No mention of coverage filter, QC threshold, or inclusion criteria

### Conclusion
The 9,175-row excess is from an unpublished filter not reproducible from public data. Keep all 74,769 rows and document the gap.

### Verification
- Bars 0-3 counts unchanged (17,836 / 17,746 / 19,594 / 19,593)
- Document all investigated filters and why they don't work

---

## Issue 5: Report QA — auto-generate summary tables from parquet

### Fix
- Write a `generate_report.py` script that:
  1. Loads the final `optiprime_full_297962.parquet` and `ground_truth_partitions.csv`
  2. Auto-generates the per-partition reconciliation table (group, flags, target n, actual n, diff) by joining ground truth against the dataframe
  3. Auto-generates the group count table via `df.groupby('group').size()`
  4. Auto-generates the context count via `df[['cell_type','PEmax','epegRNA','MLH1dn','NRCH','scaffold_name']].drop_duplicates()`
  5. Writes `report_dataset_reconstruction.md` with all tables generated from the actual data
- Add a regression test that verifies report numbers match the parquet (e.g., group counts in report == `df.groupby('group').size()`)

### Verification
- All numbers in the report match the actual parquet data
- No stale/hand-computed values

---

## Execution Order

1. **Fix bar 8** in `convert_schwank.py` → rebuild `schwank_combined.parquet`
2. **Fix bar 9** duplicate removal in `convert_schwank.py` → rebuild `schwank_combined.parquet`
3. **Rebuild combined dataset** → `optiprime_full_297962.parquet` (update `combine_dataset.py` if needed)
4. **Rebuild fold assignments** → `fold_assignments.parquet` (re-run `construct_folds.py`)
5. **Update unit tests** → fix expected counts for bars 8 and 9, context count (41), group counts
6. **Auto-generate report** → `report_dataset_reconstruction.md` from actual parquet data
7. **Run tests** → verify all pass
8. **Copy all deliverables** to `/mnt/results/`

## Expected Final State

| Bar | Target | Expected Actual | Diff | Notes |
|-----|--------|----------------|------|-------|
| 0-3 | 65,594 | 74,769 | +9,175 | Hsu filter unknown |
| 4-7 | 117,887 | 117,887 | 0 | Exact |
| 8 | 789 | 789 | 0 | Fixed: K562 PE2 non-tevo |
| 9 | 23,428 | 23,522 | +94 | Fixed: removed 18 duplicates, 94 excess remains |
| 10-12 | 2,452 | 2,452 | 0 | Exact |
| 13 | 21,201 | 823 | -20,378 | Data not publicly available |
| 14-23 | 7,844 | 7,842 | -2 | Bar 19 decoding error |
| 24-41 | 58,301 | 58,301 | 0 | Exact |

**Expected total**: 286,948 - 866 (old bar 8) + 789 (new bar 8) - 18 (bar 9 duplicates) = 286,853
**Contexts**: 41 (with scaffold) — documented discrepancy with paper's "40"
**Groups**: 12 (unchanged)
