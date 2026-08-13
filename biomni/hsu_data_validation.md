# Hsu et al. (2026) Data Validation Report

## Source
- **File**: `41587_2026_3261_MOESM3_ESM.xlsx` (Supplementary Table 3)
- **Sheets used**: `Supp Table 4 LibMMR` (10,000 rows), `Supp Table 5 LibCV` (10,406 rows)
- **Sheet not used**: `Supp Table 3 Endo_gRNAs` (283 endogenous gRNA designs — not training data)

## Extraction Summary

| Metric | Value |
|--------|-------|
| LibMMR rows (wide) | 10,000 |
| LibCV rows (wide) | 10,406 |
| LibMMR long-format rows | 36,560 |
| LibCV long-format rows | 38,209 |
| **Total (long format)** | **74,769** |
| Unique spacers | 1,146 |
| Unique proto30 sequences | 1,149 |

## Per-Context Counts (verified exact match)

| Source | Cell type | PE type | Count |
|--------|-----------|---------|-------|
| LibMMR | HEK293T | PE2 | 8,906 |
| LibMMR | HEK293T | PE4 | 8,872 |
| LibMMR | HeLa | PE2 | 9,387 |
| LibMMR | HeLa | PE4 | 9,395 |
| LibCV | HEK293T | PE2 | 8,930 |
| LibCV | HEK293T | PE4 | 8,874 |
| LibCV | HeLa | PE2 | 10,207 |
| LibCV | HeLa | PE4 | 10,198 |
| **Total** | | | **74,769** |

Per-group: Liu_HEK293T = 35,582, Liu_HeLa = 39,187

## Efficiency Values

- **Scale**: Already fractions (0–1), NOT percentages. No conversion needed.
- Edited fraction: min=0.0000, max=0.9685, mean=0.1662
- Indel fraction: min=0.0000, max=1.0000, mean=0.0079
- Unedited fraction: min=0.0000, max=1.0000 (computed as 1 - edited - indel)

## Column Derivation

### LibMMR (Supp Table 4)
| OptiPrime column | Source column | Transformation |
|-----------------|--------------|----------------|
| `spacer` | `Designed 5G pegRNA spacer` | DNA→RNA (T→U) |
| `rtt` | `Designed pegRNA extension (hom-edit-pbs)` | First (len - PBS_len) chars, T→U |
| `pbs` | `Designed pegRNA extension (hom-edit-pbs)` | Last PBS_len chars, T→U |
| `full_unedited` | `Designed target (ps-pam-edit)` | Prepend 4bp pad "GATC" |
| `full_edited` | `Designed edited target (ps-pam-edit)` | Prepend same 4bp pad |
| `edited` | `{context}_editing` columns | Already fraction, melt to long |
| `indel` | `{context}_indel` columns | Already fraction, melt to long |

- PBS_len = len(`PBS` column) per row (mostly 13, range 9–15)
- Verified: last PBS_len chars of extension = revcomp(PBS column) for all tested rows (0 mismatches in 100)
- 4bp pad required because LibMMR targets start with protospacer directly (no upstream context).
  Pad content is arbitrary ("GATC") — it only positions the protospacer at offset 4 for
  OptiPrime's structural parsing (PS20_OFFSET=4). Same pad in both unedited and edited, so
  edit detection is unaffected.

### LibCV (Supp Table 5)
| OptiPrime column | Source column | Transformation |
|-----------------|--------------|----------------|
| `spacer` | `spacer` | DNA→RNA (T→U) |
| `rtt` | `extension` | First (len - 13) chars, T→U |
| `pbs` | `extension` | Last 13 chars, T→U |
| `full_unedited` | `unedited_target` | As-is (already has 4bp pad) |
| `full_edited` | `edited_target` | As-is |
| `edited` | `{context}_editing` columns | Already fraction, melt to long |
| `indel` | `{context}_indel` columns | Already fraction, melt to long |

- PBS_len = len(`pbs_bind` column) = 13 for all rows
- LibCV targets already include 4bp upstream pad (e.g., "CAAG" before protospacer)

## Metadata (matching OptiPrime `process_liu`)

| Field | Value |
|-------|-------|
| `scaffold_name` | `BlpI_F+E` |
| `motif` | `tevoPreQ1` |
| `cas9_type` | `PEmax-Cas9` |
| `cas9_pam` | `SpNGG` |
| `rt_name` | `PE2-RT` |
| `linker` | (empty) |
| `group` | `Liu_{cell_type}` (e.g., `Liu_HEK293T`) |
| `time` | 2.0 (HEK293T) or 4.0 (HeLa) — raw 3.0/5.0 minus 1.0 for protein expression |
| `weight` | 1.0 |
| `split` | NaN (folds assigned later) |

## OptiPrime Format Compliance

- **Required columns present**: `spacer`, `rtt`, `pbs`, `full_unedited`, `full_edited` ✓
- **T→U conversion**: Applied to spacer/rtt/pbs (no T remains) ✓
- **proto30 length**: All 30 characters ✓
- **weight > 0**: All rows ✓
- **No NaN in edited/indel/unedited/weight** ✓
- **unedited = 1 - (edited + indel)**: Computed ✓
- **Derived columns**: `pegrna`, `pre_hom`, `min_edit`, `post_hom`, hashes computed ✓

## Output
- **File**: `data/processed/hsu2026_74769.parquet` (6.9 MB, 74,769 rows × 31 columns)
- **Script**: `scripts/data/extract_hsu.py`
