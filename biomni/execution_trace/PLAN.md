# Plan: Reconstruct the OptiPrime 297,962-Example PE Training Dataset (v2)

## Objective

Reconstruct the exact 297,962-example prime-editing training dataset used by Hsu et al. for OptiPrime, with **partition-level verification** against ground truth decoded from the Supplementary Information PDF (page 11 figure), not just grand-total matching.

---

## Ground Truth: 42 Partitions Decoded from SI PDF

### Method
The SI PDF (page 11, 0-indexed page 10) contains a bar chart showing all 42 data partitions. Using PyMuPDF's `get_text("rawdict")`, we extracted the n-value labels from embedded font subsets (F34 for digits, F5 for commas). Using the known partition size 115,861 as a Rosetta Stone, we decoded all 10 digit glyphs and brute-forced the remaining 6 unknowns against all 720 permutations of the remaining digits {0,2,3,4,7,9}. Exactly **one** permutation yields a sum of 297,962.

### Complete Partition Table

The figure has 3 colored bar sections (blue=Liu/Hsu, green=Schwank, purple=Kim) and a 4-row flag matrix (PEmax, epegRNA, MLH1dn, NRCH) parsed from filled/unfilled circles:

| Bar | Size | Group | PEmax | epeg | MLH1dn | NRCH | Source File |
|-----|------|-------|-------|------|--------|------|-------------|
| 0 | 15,678 | Liu_HEK293T | 1 | 1 | 0 | 0 | Hsu Lib-MMR/CV HEK293T PE2 |
| 1 | 15,598 | Liu_HEK293T | 1 | 1 | 1 | 0 | Hsu Lib-MMR/CV HEK293T PE4 |
| 2 | 17,160 | Liu_HeLa | 1 | 1 | 0 | 0 | Hsu Lib-MMR/CV HeLa PE2 |
| 3 | 17,158 | Liu_HeLa | 1 | 1 | 1 | 0 | Hsu Lib-MMR/CV HeLa PE4 |
| 4 | 824 | Schwank_HEK293T | 0 | 0 | 0 | 0 | PRIDICT v1 subscreen HEK PE2 non-tevo |
| 5 | 115,861 | Schwank_HEK293T | 0 | 1 | 0 | 0 | PRIDICT v1 focused + PRIDICT2.0 HEK PE2 |
| 6 | 822 | Schwank_HEK293T | 0 | 0 | 1 | 0 | PRIDICT v1 subscreen HEK PE2-dnMLH1 non-tevo |
| 7 | 820 | Schwank_HEK293T | 0 | 1 | 1 | 0 | PRIDICT v1 subscreen HEK PE2-dnMLH1 tevo |
| 8 | 789 | Schwank_HEK293T | 0 | 0 | 0 | 0 | PRIDICT v1 subscreen/library2 HEK PE2 non-tevo |
| 9 | 23,428 | Schwank_K562 | 0 | 1 | 0 | 0 | PRIDICT2.0 K562 PE2 + subscreen K562 PE2 tevo |
| 10 | 816 | Schwank_K562 | 1 | 0 | 0 | 0 | PRIDICT v1 subscreen K562 Pemax non-tevo |
| 11 | 819 | Schwank_K562 | 1 | 1 | 0 | 0 | PRIDICT v1 subscreen K562 Pemax tevo |
| 12 | 817 | Schwank_K562 | 0 | 0 | 1 | 0 | PRIDICT v1 subscreen K562 PE2-dnMLH1 non-tevo |
| 13 | 21,201 | Schwank_K562 | 0 | 1 | 1 | 0 | PRIDICT2.0 K562 PE4 + subscreen K562 PE4 tevo |
| 14 | 816 | Schwank_K562 | 1 | 0 | 1 | 0 | PRIDICT v1 subscreen K562 Pemax-dnMLH1 non-tevo |
| 15 | 823 | Schwank_K562 | 1 | 1 | 1 | 0 | PRIDICT v1 subscreen K562 Pemax-dnMLH1 tevo |
| 16 | 781 | Schwank_U2OS | 0 | 0 | 0 | 0 | subscreen U2OS PE2 non-tevo |
| 17 | 774 | Schwank_U2OS | 0 | 1 | 0 | 0 | subscreen U2OS PE2 tevo |
| 18 | 785 | Schwank_U2OS | 1 | 0 | 0 | 0 | subscreen U2OS Pemax non-tevo |
| 19 | 780 | Schwank_U2OS | 1 | 1 | 0 | 0 | subscreen U2OS Pemax tevo |
| 20 | 780 | Schwank_U2OS | 0 | 0 | 1 | 0 | subscreen U2OS PE2-dnMLH1 non-tevo |
| 21 | 774 | Schwank_U2OS | 0 | 1 | 1 | 0 | subscreen U2OS PE2-dnMLH1 tevo |
| 22 | 784 | Schwank_U2OS | 1 | 0 | 1 | 0 | subscreen U2OS Pemax-dnMLH1 non-tevo |
| 23 | 775 | Schwank_U2OS | 1 | 1 | 1 | 0 | subscreen U2OS Pemax-dnMLH1 tevo |
| 24 | 3,409 | Kim_HEK293T | 0 | 0 | 0 | 0 | DP_variant_293T_PE2_Conv |
| 25 | 3,418 | Kim_HEK293T | 0 | 0 | 0 | 1 | DP_variant_293T_NRCH_PE2 |
| 26 | 3,277 | Kim_HEK293T | 1 | 0 | 0 | 0 | DP_variant_293T_PE2max |
| 27 | 3,034 | Kim_HEK293T | 1 | 1 | 0 | 0 | DP_variant_293T_PE2max_epegRNA |
| 28 | 3,347 | Kim_HEK293T | 1 | 0 | 0 | 1 | DP_variant_293T_NRCH-PE2max |
| 29 | 2,531 | Kim_HEK293T | 1 | 0 | 1 | 0 | DP_variant_293T_PE4max |
| 30 | 3,331 | Kim_HEK293T | 1 | 1 | 1 | 0 | DP_variant_293T_PE4max_epegRNA |
| 31 | 3,436 | Kim_A549 | 1 | 0 | 0 | 0 | DP_variant_A549_PE2max_221114 |
| 32 | 3,229 | Kim_A549 | 1 | 1 | 0 | 0 | DP_variant_A549_PE2max_epegRNA |
| 33 | 3,392 | Kim_A549 | 1 | 0 | 1 | 0 | DP_variant_A549_PE4max |
| 34 | 3,191 | Kim_A549 | 1 | 1 | 1 | 0 | DP_variant_A549_PE4max_epegRNA |
| 35 | 3,423 | Kim_DLD1 | 1 | 0 | 0 | 0 | DP_variant_DLD1_PE2max_221114 |
| 36 | 3,057 | Kim_DLD1 | 1 | 0 | 1 | 0 | DP_variant_DLD1_PE4max |
| 37 | 3,063 | Kim_DLD1 | 1 | 0 | 1 | 1 | DP_variant_DLD1_NRCHPE4max |
| 38 | 3,386 | Kim_HeLa | 1 | 0 | 0 | 0 | DP_variant_HeLa_PE2max |
| 39 | 3,145 | Kim_MDA-MB-231 | 0 | 0 | 0 | 0 | DP_variant_MDA_PE2 |
| 40 | 3,298 | Kim_NIH3T3 | 1 | 0 | 1 | 1 | DP_variant_NIH_NRCHPE4max |
| 41 | 3,334 | Kim_HCT116 | 0 | 0 | 0 | 0 | DP_variant_HCT116_PE2 |

**Sum = 297,962** (verified)

### Lab Totals
- Liu (Hsu, bars 0-3): **65,594** (blue)
- Schwank (PRIDICT v1 + v2.0, bars 4-23): **174,067** (green)
- Kim (DeepPrime, bars 24-41): **58,301** (purple)

### 40 Experimental Contexts

Context = (cell_type, PEmax, epegRNA, MLH1dn, NRCH) — excluding lab/group and pe_type.

- HEK293T: 12 contexts (14 bars; bars 4, 8, 24 share context (0,0,0,0))
- HeLa: 3 contexts (3 bars)
- K562: 7 contexts (7 bars)
- U2OS: 8 contexts (8 bars)
- A549: 4 contexts (4 bars)
- DLD1: 3 contexts (3 bars)
- MDA-MB-231: 1, NIH3T3: 1, HCT116: 1
- **Total: 40 contexts** (matches paper)

### 12 Groups (confirmed from model checkpoint `group_factors`)

Kim_A549, Kim_DLD1, Kim_HCT116, Kim_HEK293T, Kim_HeLa, Kim_MDA-MB-231, Kim_NIH3T3, Liu_HEK293T, Liu_HeLa, Schwank_HEK293T, Schwank_K562, Schwank_U2OS

---

## Key Corrections from Previous Run

1. **Hsu true count is 65,594, not 74,769.** The previous run used raw nonmissing values from the Excel. The true count after OptiPrime's filtering (proto30==30, weight>0, dropna) is 65,594 (bars 0-3 sum). The ~12% reduction (9,175 rows) comes from the `proto30.str.len() == 30` filter in `format_pe_df`.

2. **No Schwank_HEKOpti or Schwank_Liver groups.** These don't exist in the 12 `group_factors` groups. The HEK subscreen data ("HEKOpti-Scaffold") is assigned to Schwank_HEK293T. Liver data is excluded entirely.

3. **PRIDICT2.0 uses PE2-NGG, not PE2max.** The `Editor_Variant` column in the PRIDICT2.0 supplementary Excel shows all 22,956 rows are "PE2-NGG". This means PEmax=0 (not PEmax=1 as the previous run assumed). PRIDICT2.0 HEK data goes to bar 5 (PEmax=0, epeg=1), and K562 data goes to bars 9 (PE2) and 13 (PE4/dnMLH1).

4. **DeepPrime variant files map perfectly to bars 24-41.** 18 of 19 variant files are used (the duplicate DLD1_PE2max_220428 is excluded in favor of the 221114 date). The filter rate is consistently ~83.7% (proto30 filter removes ~16.3% of rows).

5. **42 partitions, not 41 contexts.** The 40 experimental contexts arise because 3 HEK293T bars (4, 8, 24) from different labs share the same (cell_type, PEmax, epegRNA, MLH1dn, NRCH) = (HEK293T, 0, 0, 0, 0).

6. **Partition-level verification, not grand-total matching.** Each of the 42 partitions must match its decoded size exactly. No combinatorial search against the grand total.

---

## Source Code Analysis (from github.com/alvin-hsu/optiprime-src)

### Data Loading Pipeline
1. `RxDataset.load_dir()` globs `*.csv` from a directory
2. Each CSV is processed by `preprocess_fn()`:
   - `group = f'{lab_name}_{cell_type}'` from filename parts[0] and parts[1]
   - `process_fname()` dispatches by lab_name: Liu→process_liu, Schwank→process_schwank, Kim→process_hkim
   - `format_pe_df()` applies filters and computes derived columns
   - `time -= 1` (approximate protein expression time)

### Lab Processors (pe_datasets.py)
- **process_liu**: scaffold='BlpI_F+E', motif='tevoPreQ1', cas9='PEmax-Cas9', pe_type from filename, time=3.0(HEK)/5.0(HeLa). Does NOT set cas9_pam (defaults SpNGG).
- **process_schwank**: cas9 from filename (PEmax if 'max' suffix), pe_type from filename, time=7.0. Does NOT set motif (defaults tevoPreQ1) or cas9_pam (defaults SpNGG). MLH1dn is encoded via pe_type (PE4=PE2+dnMLH1).
- **process_hkim**: cas9 from 'max' in details, pe_type from details, motif from '-e' (epeg), cas9_pam from 'NRCH', time=8.0(LibClinvar)/7.0.

### Filters (format_pe_df in pe_utils.py)
- Required columns: `spacer`, `rtt`, `pbs`, `full_unedited`, `full_edited`
- `weight > 0` (default weight=1.0)
- `dropna(subset=['unedited', 'edited', 'weight'])`
- `proto30.str.len() == 30` where `proto30 = full_unedited.str.slice(0, 30)`
- T→U conversion in spacer/rtt/pbs

### Schwank Filename Convention (inferred)
The PRIDICT v1 subscreen files need renaming to match process_schwank's parser:
- PE2 → `Schwank_{cell}_subscreen_PE2.csv` (pe_type=PE2, is_max=False, PEmax=0, MLH1=0)
- PE2-dnMLH1 → `Schwank_{cell}_subscreen_PE4.csv` (pe_type=PE4, is_max=False, PEmax=0, MLH1=1)
- Pemax → `Schwank_{cell}_subscreen_PE2max.csv` (pe_type=PE2, is_max=True, PEmax=1, MLH1=0)
- Pemax-dnMLH1 → `Schwank_{cell}_subscreen_PE4max.csv` (pe_type=PE4, is_max=True, PEmax=1, MLH1=1)

The epeg (tevo/non-tevo) flag must be set via a `motif` column in the CSV ('tevoPreQ1' or 'none'), since process_schwank doesn't parse it from the filename.

### PRIDICT2.0 Data
- `Editor_Variant` = "PE2-NGG" for all 22,956 rows
- Has `spacer`, `PBS`, `RTT` columns in the supplementary Excel
- HEKaverageedited: 22,619 non-null (337 missing), K562averageedited: 22,752 non-null (204 missing)
- Efficiency values in percentages (0-93.39 for HEK, 0-90.15 for K562)
- data_23k_v1.csv has clamped values (all 22,956 non-null) as fractions (0-0.934)
- wide_initial_target: all 99 chars (no proto30 filter issues)

---

## Execution Plan

### Step 1: Project Setup & Environment
1. Verify existing project structure and venv from previous run
2. Install any missing dependencies
3. Create `reports/ground_truth_partitions.csv` with the 42-partition table above

### Step 2: Extract & Filter Hsu Data → 65,594 rows (bars 0-3)
1. Read the Hsu Excel (Lib-MMR + Lib-CV sheets)
2. Reshape to long format: one row per (pegRNA, cell_type, PE_version)
3. Derive required fields: spacer, rtt, pbs, full_unedited, full_edited, edited_frac, indel_frac
4. Apply OptiPrime filters: weight>0, dropna, proto30==30
5. **Verify per-partition**: Liu_HEK293T PE2 = 15,678, PE4 = 15,598, Liu_HeLa PE2 = 17,160, PE4 = 17,158
6. Save to `data/processed/liu_65594.parquet`

### Step 3: Convert DeepPrime Variant Files → 58,301 rows (bars 24-41)
1. For each of the 18 selected DeepPrime variant files:
   - Read WT74_On, Edited74_On, PBSlen, RTlen, Measured_PE_efficiency
   - Derive: full_unedited=WT74_On, full_edited=Edited74_On (remove 'x' deletions), spacer/rtt/pbs from sequence positions, edited_frac=efficiency/100, indel_frac=0
   - Set filename: `Kim_{celltype}_{libname}_{details}.csv` with appropriate flags
2. Apply OptiPrime filters
3. **Verify per-partition**: each file's filtered count must match the corresponding bar size (±1 tolerance for rounding)
4. Save to `data/processed/kim_58301.parquet`

### Step 4: Convert PRIDICT v1 Data → Schwank bars 4-8, 10-12, 14-23
1. **Full focused editing table (119,701 rows)** → bar 5 (115,861):
   - Extract wide_initial_target, wide_mutated_target, location columns, PE2df_percentageedited
   - Derive spacer/rtt/pbs from locations, edited_frac=PE2df_percentageedited/100
   - Set motif='tevoPreQ1' (epeg=1), pe_type='PE2', cas9_type='PE2-Cas9'
   - Apply filters and verify count ≈ 115,861 (may include PRIDICT2.0 HEK data)
2. **Subscreen files (16 files, excluding Liver)** → bars 4, 6, 7, 8, 10-12, 14-23:
   - Rename to Schwank convention (PE2→PE2, PE2-dnMLH1→PE4, Pemax→PE2max, Pemax-dnMLH1→PE4max)
   - Set motif column ('tevoPreQ1' for tevo, 'none' for non-tevo)
   - Derive required columns from wide targets and locations
   - Apply filters and verify each count matches bar size
3. **Library2 file (1,938 rows)** → may contribute to bar 8 or other small partitions
4. Save to `data/processed/schwank_v1.parquet`

### Step 5: Convert PRIDICT2.0 Data → Schwank bars 5 (HEK), 9 (K562 PE2), 13 (K562 PE4)
1. Read supplementary Excel (has spacer, PBS, RTT directly)
2. **HEK partition** (bar 5, part): filter HEKaverageedited non-null, set pe_type='PE2', motif='tevoPreQ1', cas9_type='PE2-Cas9'
3. **K562 PE2 partition** (bar 9): filter K562averageedited non-null, set pe_type='PE2', motif='tevoPreQ1', cas9_type='PE2-Cas9'
4. **K562 PE4 partition** (bar 13): filter K562averageedited non-null, set pe_type='PE4', motif='tevoPreQ1', cas9_type='PE2-Cas9'
   - NOTE: PRIDICT2.0 tested PE2-NGG, not PE4. The PE4 (dnMLH1) K562 data may come from a separate experiment by the OptiPrime authors, or from re-processing. This partition's source is the primary unresolved question.
5. Apply filters and verify counts: bar 9 ≈ 23,428, bar 13 ≈ 21,201
6. Save to `data/processed/schwank_v2.parquet`

### Step 6: Combine & Validate Full Dataset
1. Concatenate all partitions: Liu (65,594) + Schwank (174,067) + Kim (58,301) = 297,962
2. **Partition-level verification**: for each of the 42 partitions, assert the row count matches the ground truth from the SI figure
3. Verify 12 unique groups
4. Verify 40 unique (cell_type, PEmax, epegRNA, MLH1dn, NRCH) contexts
5. Cross-check group assignments against `group_factors` in model weights
6. Save to `data/processed/optiprime_full_297962.parquet`
7. Write `reports/dataset_reconstruction_status.md` with per-partition verification table

### Step 7: Fold Construction
1. Implement protospacer-disjoint fivefold split using `split_preset` logic from source code
2. The `split` column must come from the CSV files (if present) or be constructed
3. Verify: all records for the same protospacer in one fold only
4. Save to `data/processed/fold_assignments.parquet`

### Step 8: Unit Tests
1. Test per-partition row counts against ground truth (42 assertions)
2. Test group count = 12
3. Test context count = 40
4. Test sequence alignment (full_unedited vs full_edited)
5. Test fold leakage (same protospacer → same fold)
6. Test flag consistency (PEmax/epegRNA/MLH1dn/NRCH per partition)

---

## Unresolved Questions (to be investigated during execution)

1. **Bar 13 source (21,201 K562 PE4 epeg)**: PRIDICT2.0 uses PE2-NGG, not PE4. The K562 PE4 (dnMLH1) data may come from a separate experiment by the Hsu/Schwank labs not in public supplementary files. If the source cannot be found, this partition will be flagged as "unresolved" and the closest available data (PRIDICT2.0 K562 PE2 re-labeled as PE4, or subscreen data) will be used as a proxy.

2. **Bar 5 exact composition (115,861 HEK PE2 epeg)**: Likely = PRIDICT v1 focused/full table + PRIDICT2.0 HEK + subscreen HEK tevo. The exact combination and filter rate will be determined empirically.

3. **Bar 8 source (789 HEK PE2 non-tevo)**: Same flags as bar 4 (824) but different size. May come from library2 file or a different subscreen experiment. Differs from bar 4 in pe_type or another parameter not captured in the flag matrix.

4. **The proto30 filter**: `full_unedited.str.slice(0, 30)` must be exactly 30 chars. This is the primary filter causing row loss. For DeepPrime data (WT74_On, 74 chars), the filter removes rows where the first 30 chars are shorter (shouldn't happen with 74-char sequences, so the ~16% loss may come from a different issue — possibly the 'x' deletion handling in Edited74_On affecting full_unedited derivation). For PRIDICT data (99-char targets), no rows should be lost to this filter.

5. **Efficiency scale**: PRIDICT data uses percentages (0-100), PRIDICT2.0 uses percentages in Excel / fractions in CSV, DeepPrime uses percentages. All must be converted to fractions (0-1) for `edited_frac`.

---

## Deliverables

- `data/processed/liu_65594.parquet` — Hsu data (4 partitions)
- `data/processed/kim_58301.parquet` — DeepPrime data (18 partitions)
- `data/processed/schwank_v1.parquet` — PRIDICT v1 data (17 partitions)
- `data/processed/schwank_v2.parquet` — PRIDICT2.0 data (3 partitions)
- `data/processed/optiprime_full_297962.parquet` — Full dataset (42 partitions)
- `data/processed/fold_assignments.parquet` — Fivefold CV assignments
- `reports/ground_truth_partitions.csv` — The 42-partition ground truth table
- `reports/dataset_reconstruction_status.md` — Per-partition verification report
- `tests/test_data_pipeline.py` — Unit tests

---

## Execution Order

1. Step 1: Setup (quick, reuse existing venv)
2. Step 2: Hsu data extraction (medium, need to derive sequences from Excel)
3. Step 3: DeepPrime conversion (medium, 18 files with known mapping)
4. Steps 4-5: PRIDICT conversion (hard, need to derive sequences and resolve unknowns)
5. Step 6: Combine & validate (critical — partition-level verification)
6. Step 7: Folds (medium)
7. Step 8: Tests (quick)

Steps 2 and 3 can proceed in parallel. Steps 4-5 are the critical path and may require iterative debugging.
