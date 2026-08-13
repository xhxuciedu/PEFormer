# Plan: Reconstruct the OptiPrime 297,962-Example PE Training Dataset

## Objective

Reconstruct, as closely as possible, the exact 297,962-example prime-editing training dataset used by Hsu et al. for OptiPrime, then set up the project infrastructure for the broader PE-RankFormer pilot study.

**Arithmetic:** 74,769 (Hsu Lib-MMR + Lib-CV) + 223,193 (historical: PRIDICT v1, DeepPrime, PRIDICT2.0) = 297,962.

---

## What We Know (Verified During Investigation)

### Hsu Data (74,769) — VERIFIED EXACTLY
- Source: `/mnt/user-uploads/41587_2026_3261_MOESM3_ESM.xlsx`
- Lib-MMR sheet: 10,000 data rows, 4 editing columns → 36,560 nonmissing values
- Lib-CV sheet: 10,406 data rows, 4 editing columns → 38,209 nonmissing values
- Total: 74,769 (verified per-column: HEK293T_PE2=17,836, HEK293T_PE4=17,746, HeLa_PE2=19,594, HeLa_PE4=19,593)
- 4 experimental contexts: HEK293T PE2, HEK293T PE4, HeLa PE2, HeLa PE4

### OptiPrime Data Loading Pipeline (from source code)
- `RxDataset.load_dir()` globs `*.csv` from a directory, reads each, applies `preprocess_fn`, concatenates
- `preprocess_fn()` (scripts/pe/1_train.py): sets `group = f'{lab_name}_{cell_type}'` from filename, calls `process_fname()`, then `format_pe_df()`, sets time-1, computes hashes
- **Filename convention**: `{lab_name}_{cell_type}_{...}_{pe_type_or_editor}.csv`
- **Lab processors** (scripts/pe/pe_datasets.py):
  - `process_liu` (Hsu): cell_type from parts[1], pe_type from parts[3]. Scaffold='BlpI_F+E', motif='tevoPreQ1', cas9='PEmax-Cas9', time=3.0 (HEK293T) or 5.0 (HeLa)
  - `process_schwank` (PRIDICT/PRIDICT2.0): type_editor from parts[3]. pe_type=type_editor[0:3], is_max=(type_editor[3:]=='max'), cas9='PEmax-Cas9' if max else 'PE2-Cas9', time=7.0
  - `process_hkim` (DeepPrime): lib_name from parts[2], details from parts[3]. pe_type=details[0:3], is_max=('max' in details), is_epeg=('-e' in details), is_nrch=('NRCH' in details), time=8.0 (LibClinvar) or 7.0
  - `process_ykim` (DeepPE/Kim 2021): pam_var from parts[2]. cas9='PE2-Cas9', pe_type='PE2', motif='none', time=3.0
- **`format_pe_df()`** (scripts/pe/pe_utils.py):
  - Required columns: `spacer`, `rtt`, `pbs`, `full_unedited`, `full_edited`
  - Default values for missing: scaffold_name='SpCas9_OG', motif='tevoPreQ1', cas9_type='PE2-Cas9', cas9_pam='SpNGG', pe_type='PE2', weight=1.0, edited_frac=0.0, indel_frac=0.0
  - Computes `unedited = 1 - (edited_frac + indel_frac)`
  - Renames edited_frac→edited, indel_frac→indel
  - **Filters**: `weight > 0`, `dropna(subset=['unedited', 'edited', 'weight'])`, `proto30.str.len() == 30`
  - Converts T→U in spacer/rtt/pbs
  - proto30 = full_unedited.str.slice(0, 30) (PS20_OFFSET=4, so protospacer starts at position 4)

### Historical Data Sources Identified

#### 1. DeepPrime (Yu et al. 2023, Cell — ref 55)
- **Location**: `/workspace/external/deepprime_official/data/`
- **Main dataset**: `DeepPrime_dataset_final_Feat8.csv` — 288,793 rows (259,910 train folds 0-4 + 28,883 test)
  - Columns: WT74_On (74bp), Edited74_On (74bp, 'x' for deletions), PBSlen, RTlen, Measured_PE_efficiency (percentages 0-62.35), Fold, Edit_pos, Edit_len, type_sub/ins/del
  - Edit lengths: 1-3 only. Types: 153,974 sub, 81,503 ins, 53,316 del
  - All HEK293T PE2 conventional (ClinVar library)
- **19 variant files**: `DP_variant_*.csv` — 72,681 rows total (60,857 train + 11,824 test)
  - Same 34-column format, all non-null efficiencies
  - Contexts: HEK293T (7: PE2_Conv, PE2max, PE2max-e, PE4max, PE4max-e, NRCH-PE2, NRCH-PE2max), A549 (4), DLD1 (4, two PE2max dates), HCT116 (1), HeLa (1), MDA-MB-231 (1), NIH3T3 (1)
  - **18 unique experimental contexts** (two DLD1_PE2max dates = same context)

#### 2. PRIDICT2.0 (Mathis et al. 2024 — ref 56)
- **Location**: `/workspace/external/pridict2/dataset/proc_v2/data_23k_v1.csv` (22,956 rows, 35 cols)
  - Has `wide_initial_target`, `wide_mutated_target`, PBSlength, RTlength
  - HEKaverageedited_clamped: 22,956 non-null (all, range 0-0.934, fractions)
  - K562averageedited_clamped: 22,956 non-null (all, range 0-0.902, fractions)
  - Also has unedited and unintended columns for both cell types
- **Supplementary Excel** (epridict supplementary branch, cloned to `/workspace/external/epridict_supp/`):
  - `SupplFile1_Library_Diverse_Editing_Results_with_test_splits.xlsx`: 22,956 rows, 141 cols
  - **Has `spacer`, `PBS`, `RTT` columns directly** — no need to derive from target sequences
  - HEKaverageedited: 22,619 non-null (337 missing), K562averageedited: 22,752 non-null (204 missing)
  - Has `test_split_hek` and `test_split_k562` for train/test splits
  - Efficiency values in percentages (0-100), not fractions
  - 2 experimental contexts: HEK293T PE2max, K562 PE2max

#### 3. PRIDICT v1 (Mathis et al. 2023, Nat Biotechnol — ref 54)
- **Location**: `/workspace/external/pridict_supp/` (supplementary_files branch)
- **Main editing table**: `03_jupyter_notebooks/20220719_FINAL_Editingtable_focused_NM_withindex.zip`
  - 92,423 rows, 62 columns
  - Has `wide_initial_target`, `wide_mutated_target`, location columns (protospacerlocation, PBSlocation, RT_initial_location, RT_mutated_location)
  - `averageedited`: 92,423 non-null (percentages, range -3.38 to 93.79 — has negative values from background subtraction)
  - `averageedited_original`: 92,423 non-null (range 0-93.79, cleaner)
  - `averageindel`: 92,423 non-null (percentages, range -14.45 to 96.93)
  - Does NOT have explicit spacer/PBS/RTT columns — must derive from wide targets + locations
  - 1 experimental context: HEK293T PE2
- **20 subscreen files**: `03_jupyter_notebooks/subscreen_featuredf_files/`
  - ~964-974 rows each, ~19,380 total
  - Cell types: HEK (Opti-Scaffold), K562, U2OS, Liver-GFPplus
  - PE versions: PE2, PE2-dnMLH1, Pemax, Pemax-dnMLH1, PE2Adeno
  - epeg status: tevopreq vs nontevopreq
  - Up to 20 potential contexts, but process_schwank only handles PE2/PE2max/PE4/PE4max naming

#### 4. YKim / DeepPE (Kim et al. 2021, Nat Biotechnol — ref 59)
- Code has `process_ykim` but paper says training data from refs 54-56 only
- Ref 59 is cited alongside ref 55 as "DeepPrime55,59" in the intro, but the training data section says "previous studies54–56"
- **YKim data NOT available in any cloned repo** — would need to be obtained separately
- **Working assumption: YKim data is NOT part of the 223,193** (paper explicitly says refs 54-56)

### The 40 Experimental Contexts
- 4 from Hsu (Liu): HEK293T PE2, HEK293T PE4, HeLa PE2, HeLa PE4
- 36 from historical — exact breakdown unclear
- DeepPrime variants: 18 unique contexts
- PRIDICT2.0: 2 contexts (HEK, K562)
- PRIDICT v1 main: 1 context (HEK293T PE2)
- Subtotal: 21 — gap of 15 to reach 36
- **Resolution**: PRIDICT v1 subscreens may contribute additional contexts. The subscreen files cover K562 (PE2, PE2-dnMLH1, Pemax, Pemax-dnMLH1 × tevo/non-tevo = 8), U2OS (same = 8), HEK (PE2 tevo/non-tevo = 2), Liver (PE2Adeno tevo/non-tevo = 2) = 20 potential contexts. If 15 of these 20 are used, total = 4 + 18 + 2 + 1 + 15 = 40. This will be verified during execution.

### The 223,193 Arithmetic Challenge
Available raw data far exceeds 223,193. Key combinations tested:

| Combination | Count | Delta |
|---|---|---|
| DP variants (all) + PRIDICT2.0 (non-null both) + PRIDICT v1 main | 210,475 | -12,718 |
| DP variants (all) + PRIDICT2.0 (all both) + PRIDICT v1 main | 211,016 | -12,177 |
| DP variants (all) + PRIDICT2.0 (non-null) + PRIDICT v1 main + 13 subscreens | ~223,193 | ~0 |
| DP variants (train) + PRIDICT2.0 (non-null) + PRIDICT v1 main + all subscreens | 218,031 | -5,162 |

**No combination yields exactly 223,193 without including subscreen data or YKim data.** The exact filtering will be resolved systematically during execution (see Step 5 below).

---

## Execution Plan

### Step 1: Project Setup & Environment
1. Create project directory structure per claude.md (data/raw, data/interim, data/processed, data/manifests, src/pe_rankformer, scripts, configs, tests, results, reports, etc.)
2. Create Python 3.11 venv, install dependencies (pandas, pyarrow, openpyxl, etc. — full ML stack for later steps)
3. Save `reports/environment.txt` and `reports/project_inventory.md`
4. Initialize git repo

### Step 2: Extract Hsu Data (74,769) → Parquet
1. Read the Excel workbook, extract Lib-MMR and Lib-CV sheets
2. Reshape to long format: one row per (pegRNA, cell_type, PE_version) combination
3. Derive required fields:
   - `spacer`, `rtt`, `pbs` from the pegRNA design columns in the Excel
   - `full_unedited`, `full_edited` from the target site sequences
   - `edited_frac` = editing_efficiency / 100 (convert from percentage to fraction)
   - `indel_frac` = indel_rate / 100
4. Verify: `assert len(hsu_long) == 74769`
5. Save to `data/processed/hsu2026_74769.parquet`
6. Write `reports/hsu_data_validation.md`

**Key challenge**: The Hsu Excel has target site sequences and pegRNA designs, but the exact column mapping to OptiPrime's required format needs careful reverse-engineering. The Supp Table 4/5 columns include: spacer, scaffold, rtt, pbs, linker, motif, full_unedited, full_edited, cas9_pam, and the 4 editing columns. Need to verify which columns are present.

### Step 3: Download & Catalog Historical Data
1. Copy DeepPrime data from `/workspace/external/deepprime_official/data/` to `data/raw/deepprime/`
2. Copy PRIDICT2.0 data from `/workspace/external/pridict2/dataset/` and `/workspace/external/epridict_supp/` to `data/raw/pridict2/`
3. Extract PRIDICT v1 data from `/workspace/external/pridict_supp/` zip files to `data/raw/pridict/`
4. Calculate SHA256 checksums for all source files
5. Create `data/manifests/data_sources.csv` with provenance, checksums, row counts

### Step 4: Reverse-Engineer OptiPrime Data Loader
1. Document the complete loading pipeline (already analyzed above)
2. Write `reports/optiprime_data_loader_reverse_engineering.md` covering:
   - Filename conventions for each lab
   - Required vs optional columns
   - Filtering rules (weight>0, dropna, proto30==30)
   - Efficiency scale conversion (percentage → fraction)
   - T→U conversion in spacer/rtt/pbs
   - Lab-specific metadata assignment (scaffold, motif, cas9_type, time)
   - The `split_preset` function for fivefold CV (stratified by protospacer)

### Step 5: Convert Historical Data to OptiPrime Format & Reconcile to 223,193

This is the core challenge. Approach:

#### 5a. DeepPrime Conversion
- **Input**: WT74_On (74bp), Edited74_On (74bp, 'x'=deletion), PBSlen, RTlen, Measured_PE_efficiency (percentage)
- **Derivation**:
  - `full_unedited` = WT74_On (already ≥30 chars)
  - `full_edited` = Edited74_On with 'x' removed (compress deletions)
  - `spacer` = dna_to_rna(WT74_On[4:24]) (protospacer at positions 4-23, 0-indexed)
  - `rtt` = dna_to_rna(revcomp(Edited74_On[24:24+RTlen])) (downstream of nick, from edited sequence)
  - `pbs` = dna_to_rna(revcomp(WT74_On[24-PBSlen:24])) (upstream of nick, from WT sequence)
  - `edited_frac` = Measured_PE_efficiency / 100
  - `indel_frac` = 0.0 (DeepPrime doesn't provide indel rates separately)
- **Filename convention**: `Kim_{cell_type}_{lib_name}_{details}.csv`
  - e.g., `Kim_HEK293T_LibClinvar_PE2.csv` for main dataset
  - e.g., `Kim_HEK293T_LibVariant_PE2max.csv` for variants
- **Apply filtering**: weight>0, dropna, proto30==30
- **Count retained rows per file**

#### 5b. PRIDICT2.0 Conversion
- **Input (from supplementary Excel)**: spacer, PBS, RTT (already available!), wide_initial_target, wide_mutated_target, HEKaverageedited, K562averageedited (percentages)
- **Derivation**:
  - `full_unedited` = wide_initial_target (need to verify length ≥30)
  - `full_edited` = wide_mutated_target
  - `spacer` = dna_to_rna(spacer) (already provided)
  - `rtt` = dna_to_rna(RTT) (already provided)
  - `pbs` = dna_to_rna(PBS) (already provided)
  - `edited_frac` = HEKaverageedited / 100 or K562averageedited / 100
  - `indel_frac` = from HEKaverageunintended / 100 or K562averageunintended / 100
- **Two output files**: `Schwank_HEK293T_Library_PE2max.csv` and `Schwank_K562_Library_PE2max.csv`
- **Apply filtering**: dropna on edited (removes 337 HEK rows, 204 K562 rows with missing efficiencies)

#### 5c. PRIDICT v1 Conversion
- **Input**: wide_initial_target, wide_mutated_target, location columns, averageedited_original (percentage), averageindel_original (percentage)
- **Derivation**:
  - `full_unedited` = wide_initial_target
  - `full_edited` = wide_mutated_target
  - `spacer` = dna_to_rna(wide_initial_target[protospacer_start:protospacer_end]) using protospacerlocation
  - `rtt` = dna_to_rna(revcomp(wide_mutated_target[RT_start:RT_end])) using RT_mutated_location
  - `pbs` = dna_to_rna(revcomp(wide_initial_target[PBS_start:PBS_end])) using PBSlocation
  - `edited_frac` = averageedited_original / 100
  - `indel_frac` = averageindel_original / 100
- **Filename**: `Schwank_HEK293T_Library1_PE2.csv`
- **Apply filtering**: weight>0, dropna, proto30==30
- **Also convert subscreen files** (20 files, each to separate Schwank CSV)

#### 5d. Reconciliation
1. After converting all sources, count retained rows per file
2. Build a source-level reconciliation table
3. Systematically test combinations:
   - With/without DeepPrime test folds
   - With/without DeepPrime main dataset
   - With/without PRIDICT v1 subscreens
   - With/without PRIDICT2.0 missing-efficiency rows
   - Check for duplicate removal (same protospacer+edit across datasets)
4. Use the 40-context constraint to narrow down
5. Target: `assert len(historical_table) == 223193`
6. If exact match found: document the exact filtering
7. If no exact match: produce discrepancy table, preserve maximally verified dataset, document honestly in `reports/dataset_reconstruction_status.md`

### Step 6: Combine & Validate Full Dataset
1. Concatenate Hsu (74,769) + historical (223,193) → 297,962
2. `assert len(full_training_table) == 297962`
3. Verify unique experimental contexts = 40
4. Save to `data/processed/optiprime_full_297962.parquet`
5. Write `reports/dataset_reconstruction_status.md` with reconciliation table

### Step 7: Fold Construction
1. Implement protospacer-disjoint fivefold split (stratified by protospacer sequence)
2. Verify: all records for the same protospacer in one fold only
3. Save to `data/processed/fold_assignments.parquet`

### Step 8: Unit Tests for Data Pipeline
1. Test sequence alignment (full_unedited vs full_edited)
2. Test PBS/RTT segmentation
3. Test efficiency scale conversion (percentage → fraction)
4. Test target/protospacer grouping
5. Test fold leakage (same protospacer → same fold)
6. Test dataset row counts (74,769, 223,193, 297,962)

---

## Key Risks & Mitigations

1. **Exact 223,193 may not be reproducible**: The paper doesn't specify the exact filtering applied to historical data. Mitigation: systematic trial of all combinations, document discrepancy if exact match impossible.

2. **DeepPrime sequence derivation may have edge cases**: The 'x' deletion representation in Edited74_On needs careful handling. Mitigation: validate against known pegRNA designs, check proto30 length for all rows.

3. **PRIDICT v1 negative efficiency values**: The `averageedited` column has negative values from background subtraction. Mitigation: use `averageedited_original` (range 0-93.79) or clip to [0, 1] after conversion.

4. **PRIDICT v1 subscreen inclusion uncertain**: The process_schwank function only handles PE2/PE2max/PE4/PE4max naming, but subscreens have PE2-dnMLH1, Pemax, etc. Mitigation: test both with and without subscreens; the 40-context constraint will help determine inclusion.

5. **YKim data availability**: If YKim data is needed for the count to work, it's not available in any cloned repo. Mitigation: first try without YKim (per paper's refs 54-56 statement); if needed, attempt to obtain from ref 59's supplementary data.

---

## Deliverables (Steps 3-8)

- `data/processed/hsu2026_74769.parquet`
- `data/processed/optiprime_full_297962.parquet`
- `data/processed/fold_assignments.parquet`
- `data/manifests/data_sources.csv`
- `reports/project_inventory.md`
- `reports/environment.txt`
- `reports/hsu_data_validation.md`
- `reports/optiprime_data_loader_reverse_engineering.md`
- `reports/dataset_reconstruction_status.md`
- `reports/research_log.md`
- Unit tests in `tests/`

---

## Execution Order

1. Project setup & environment (Step 1)
2. Hsu data extraction & validation (Step 2)
3. Historical data download & cataloging (Step 3)
4. OptiPrime loader reverse-engineering report (Step 4)
5. Historical data conversion & reconciliation (Step 5) — **core challenge**
6. Full dataset combination & validation (Step 6)
7. Fold construction (Step 7)
8. Unit tests (Step 8)

Steps 1-4 can proceed in parallel with data investigation. Step 5 is the critical path and may require iterative debugging. Steps 6-8 follow once Step 5 succeeds.
