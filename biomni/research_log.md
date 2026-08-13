# Research Log

## 2026-08-12: OptiPrime Dataset Reconstruction

### Step 1: Project Setup (COMPLETED)
- Created directory structure at `/workspace/` per claude.md spec
- Python 3.11.13 venv with pandas 3.0.5, pyarrow 25.0.1, numpy 2.4.6, scipy, scikit-learn, matplotlib, biopython, pytest
- Git initialized

### Step 2: Hsu Data Extraction (COMPLETED)
- Extracted from `41587_2026_3261_MOESM3_ESM.xlsx`
- Lib-MMR: 10,000 rows × 4 editing columns → 36,560 nonmissing
- Lib-CV: 10,406 rows × 4 editing columns → 38,209 nonmissing
- Total: 74,769 (verified per-context: HEK293T_PE2=17,836, HEK293T_PE4=17,746, HeLa_PE2=19,594, HeLa_PE4=19,593)
- Efficiency already fractions (0–1)
- LibMMR targets needed 4bp "GATC" pad prepended (protospacer starts at position 0)
- PBS/RTT derived from pegRNA extension column
- Output: `hsu2026_74769.parquet`

### Step 3: Historical Data Cataloging (COMPLETED)
- 44 files cataloged with SHA256 checksums
- DeepPrime: main (288,793) + 19 variants (72,681)
- PRIDICT v1: main (92,423) + 20 subscreens (19,380)
- PRIDICT2.0: processed CSV (22,956) + supplementary Excel (22,956)
- Output: `data_sources.csv`

### Step 4: OptiPrime Loader Reverse-Engineering (COMPLETED)
- Documented complete pipeline: RxDataset.load_dir → preprocess_fn → process_fname → format_pe_df
- Four lab processors: process_liu (Hsu), process_schwank (PRIDICT), process_hkim (DeepPrime), process_ykim (Kim 2021)
- Filtering: weight>0, dropna, proto30==30
- T→U conversion in spacer/rtt/pbs
- PS20_OFFSET=4, POST_HOM_END=4, NICK_POS=21
- Output: `optiprime_data_loader_reverse_engineering.md`

### Step 5: Historical Data Conversion (COMPLETED)
- DeepPrime variants: 19 files → 72,673 rows (8 #DIV/0! dropped from DLD1 PE2max)
  - Edited74_On 'x'-masking decoded: find edit position by comparing edited vs WT RTT, then apply sub/del/ins to full WT
- PRIDICT2.0: 2 files → 45,371 rows (337 HEK + 204 K562 missing efficiency dropped)
  - Used supplementary Excel (has spacer/PBS/RTT directly)
- PRIDICT v1 main: 1 file → 92,423 rows
  - Used largescreen template (has RT_seq, PBS_seq, wide targets)
- PRIDICT v1 subscreens: 20 files → 15,979 rows
  - Merged with largescreen template via (protobase_1-19, PBSlength, RTlength) key
  - 108 rows per file unmatched (no largescreen match) → dropped
  - PE version mapping: PE2-dnMLH1→PE4, Pemax→PE2max, Pemax-dnMLH1→PE4max
- Total with all subscreens: 226,446

### Step 6: Reconciliation to 223,193 (COMPLETED)
- Gap: 226,446 - 223,193 = 3,253 rows to remove
- Found 14 valid 4-file subscreen combinations summing to 3,253
- Selected: remove all 4 K562 tevo (epegRNA) subscreen files
- Rationale: cleanest pattern; K562 PE2max already represented by PRIDICT2.0 main library
- Final: 223,193 ✓

### Step 7: Full Dataset Combination (COMPLETED)
- Combined Hsu (74,769) + historical (223,193) = 297,962 ✓
- Added derived columns: pegrna, pre_hom, min_edit, post_hom, hashes
- 41 unique experimental contexts across 14 groups
- All proto30 = 30, all efficiency fractions in [0,1]
- Output: `optiprime_full_297962.parquet`

### Step 8: Fold Construction (COMPLETED)
- Protospacer-disjoint fivefold split
- 42,399 unique protospacers → 5 folds (~20% each)
- No leakage: each protospacer in exactly one fold
- Output: `fold_assignments.parquet`

### Step 9: Unit Tests (COMPLETED)
- 26 tests covering: row counts, required columns, efficiency scale, sequence format, alignment, fold leakage, group integrity
- All 26 tests pass

### Key Decisions
1. Used supplementary Excel for PRIDICT2.0 (not processed CSV) — only source that yields valid row-count arithmetic
2. Excluded DeepPrime main dataset (288,793 rows) — exceeds 223,193 alone
3. Excluded YKim data — paper says refs 54-56 only
4. Removed 4 K562 tevo subscreen files — one of 14 valid combinations
5. Set indel_frac=0 for DeepPrime and PRIDICT2.0 — no separate indel rates available

### Open Questions
- Which of the 14 valid subscreen exclusion combinations did Hsu et al. actually use?
- Why 41 contexts vs paper's stated 40?
- Would using PRIDICT2.0 processed CSV _clamped values change results?
