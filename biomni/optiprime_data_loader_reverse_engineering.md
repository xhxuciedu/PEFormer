# OptiPrime Data Loader: Reverse-Engineering Report

## Overview

This report documents the complete data loading pipeline used by Hsu et al. (OptiPrime),
reverse-engineered from the source code at `/workspace/external/optiprime/`. The pipeline
transforms raw CSV files from multiple labs into a unified training dataset.

## Source Files Analyzed

| File | Purpose |
|------|---------|
| `scripts/pe/1_train.py` | Main training script; defines `preprocess_fn` |
| `scripts/pe/pe_datasets.py` | Lab-specific processors (`process_fname`) |
| `scripts/pe/pe_utils.py` | Core formatting (`format_pe_df`) and feature extraction |
| `scripts/pe/pe_constants.py` | Constants (PS20_OFFSET, scaffolds, PAMs, etc.) |
| `reaction/rx_dataset.py` | Dataset class (`RxDataset.load_dir`, `split_preset`) |

---

## 1. Data Loading Flow

```
RxDataset.load_dir(data_path, preprocess_fn)
  → globs *.csv from data_path
  → for each CSV: read → preprocess_fn(path, df) → collect
  → concatenate all processed DataFrames
```

### `preprocess_fn(path, df)` (1_train.py)

```python
def preprocess_fn(p, df):
    name_parts = p.stem.split('_')
    lab_name, cell_type = name_parts[0], name_parts[1]
    df['group'] = f'{lab_name}_{cell_type}'
    process_fname(p, df)          # Lab-specific metadata
    df = format_pe_df(p, df)      # Core formatting + filtering
    df['time'] = df['time'] - 1   # Subtract 1 for protein expression time
    df['spacer_hash'] = df['spacer'].apply(deterministic_hash)
    df['pegrna_hash'] = df['pegrna'].apply(deterministic_hash)
    df['edit_hash'] = df['min_edit'].apply(deterministic_hash)
    return df
```

**Key**: `group` is `{lab_name}_{cell_type}` — NOT including PE type or editor variant.
The time is decremented by 1 after formatting.

---

## 2. Filename Convention

Files are named `{lab_name}_{cell_type}_{...}_{pe_type_or_editor}.csv`.

The `process_fname` function dispatches on `lab_name` (first underscore-delimited token):

| Lab prefix | Processor | Studies |
|-----------|-----------|---------|
| `Liu` | `process_liu` | Hsu et al. 2026 (this paper) |
| `Schwank` | `process_schwank` | PRIDICT v1 (ref 54), PRIDICT2.0 (ref 56) |
| `Kim` | `process_hkim` | DeepPrime (ref 55) |
| `YKim` | `process_ykim` | DeepPE / Kim et al. 2021 (ref 59) |

---

## 3. Lab-Specific Processors

### `process_liu` (Hsu / OptiPrime)

```python
def process_liu(p, df):
    name_parts = p.stem.split('_')
    cell_type, pe_type = name_parts[1], name_parts[3]
    assert cell_type in ['HEK293T', 'HeLa']
    assert pe_type in ['PE2', 'PE4']
    df['scaffold_name'] = 'BlpI_F+E'
    df['motif'] = 'tevoPreQ1'
    df['cas9_type'] = 'PEmax-Cas9'
    df['pe_type'] = pe_type
    df['time'] = 3.0 if cell_type == 'HEK293T' else 5.0
```

- Filename: `Liu_{cell_type}_{...}_{PE2|PE4}.csv`
- 4 contexts: HEK293T PE2, HEK293T PE4, HeLa PE2, HeLa PE4
- All use PEmax-Cas9, tevoPreQ1 motif, BlpI_F+E scaffold
- Time: 3.0 (HEK293T) → 2.0 after subtraction; 5.0 (HeLa) → 4.0

### `process_schwank` (PRIDICT / PRIDICT2.0)

```python
def process_schwank(p, df):
    name_parts = p.stem.split('_')
    type_editor = name_parts[3]
    pe_type = type_editor[0:3]
    is_max = type_editor[3:] == 'max'
    df['cas9_type'] = 'PEmax-Cas9' if is_max else 'PE2-Cas9'
    df['pe_type'] = pe_type
    df['time'] = 7.0
```

- Filename: `Schwank_{cell_type}_{...}_{PE2|PE2max|PE4|PE4max}.csv`
- `pe_type` extracted from first 3 chars of `type_editor` (e.g., "PE2", "PE4")
- `is_max` = True if `type_editor` ends with "max" → PEmax-Cas9
- Time: 7.0 → 6.0 after subtraction
- **Note**: Does NOT set `motif` or `cas9_pam` — these default to `tevoPreQ1` and `SpNGG`
  in `format_pe_df`. This means all Schwank data is treated as epegRNA with SpNGG PAM.

### `process_hkim` (DeepPrime)

```python
def process_hkim(p, df):
    name_parts = p.stem.split('_')
    lib_name, details = name_parts[2], name_parts[3]
    pe_type = details[0:3]
    is_max = 'max' in details
    is_epeg = '-e' in details
    is_nrch = 'NRCH' in details
    df['cas9_type'] = 'PEmax-Cas9' if is_max else 'PE2-Cas9'
    df['pe_type'] = pe_type
    df['motif'] = 'tevoPreQ1' if is_epeg else 'none'
    df['cas9_pam'] = 'SpNRCH' if is_nrch else 'SpNGG'
    df['time'] = 8.0 if lib_name == 'LibClinvar' else 7.0
```

- Filename: `Kim_{cell_type}_{lib_name}_{details}.csv`
- `details` encodes PE type, max, epeg, NRCH in a single string
- `is_epeg` = True if '-e' in details → tevoPreQ1 motif
- `is_nrch` = True if 'NRCH' in details → SpNRCH PAM
- Time: 8.0 (LibClinvar) → 7.0; 7.0 (variants) → 6.0

### `process_ykim` (DeepPE / Kim 2021)

```python
def process_ykim(p, df):
    name_parts = p.stem.split('_')
    pam_var = name_parts[2]
    df['cas9_type'] = 'PE2-Cas9'
    df['pe_type'] = 'PE2'
    df['motif'] = 'none'
    df['cas9_pam'] = pam_var
    df['time'] = 3.0
```

- Filename: `YKim_{cell_type}_{pam_var}_{...}.csv`
- Always PE2-Cas9, no epeg, PAM from filename
- Time: 3.0 → 2.0 after subtraction
- **Paper says training data from refs 54-56 only; ref 59 (YKim) is cited but may not
  be in the training set.**

---

## 4. Core Formatting: `format_pe_df(p, df)`

### Required Columns (must be in CSV)

```python
_REQUIRED_COLUMNS = ['spacer', 'rtt', 'pbs', 'full_unedited', 'full_edited']
```

### Default Values (set if column missing)

```python
_DEFAULT_VALUES = {
    'scaffold_name': 'SpCas9_OG',
    'linker': '',
    'motif': 'tevoPreQ1',
    'cas9_type': 'PE2-Cas9',
    'cas9_pam': 'SpNGG',
    'rt_name': 'PE2-RT',
    'pe_type': 'PE2',
    'group': 'NO_GROUP',
    'time': 1.0,
    'split': np.nan,
    'weight': 1.0,
    'edited_frac': 0.0,
    'indel_frac': 0.0,
}
```

**Important**: `motif` defaults to `tevoPreQ1` (epegRNA) and `cas9_pam` defaults to `SpNGG`.
Labs that don't explicitly set these (e.g., Schwank) get epegRNA/SpNGG by default.

### Processing Steps

1. **Compute unedited fraction**:
   ```python
   df['unedited'] = 1 - (df['edited_frac'] + df['indel_frac'])
   ```

2. **Rename columns**:
   ```python
   df = df.rename(columns={'edited_frac': 'edited', 'indel_frac': 'indel'})
   ```

3. **Filter — three rules applied in order**:
   ```python
   df = df[df['weight'] > 0]                                    # Rule 1
   df = df.dropna(subset=['unedited', 'edited', 'weight'])      # Rule 2
   ```

4. **proto30 check**:
   ```python
   df['proto30'] = df['full_unedited'].str.slice(0, 30)
   assert (df['proto30'].str.len() == 30).all()                 # Rule 3
   ```

   `proto30` = first 30 bases of `full_unedited`. With `PS20_OFFSET = 4`:
   - Positions 0–3: upstream pad (4bp)
   - Positions 4–23: protospacer (20bp)
   - Positions 24–27: PAM (4bp, e.g., NGG + 1)
   - Positions 28–29: downstream (2bp)

5. **T→U conversion** (RNA alphabet):
   ```python
   df['spacer'] = df['spacer'].str.replace('T', 'U')
   df['rtt'] = df['rtt'].str.replace('T', 'U')
   df['pbs'] = df['pbs'].str.replace('T', 'U')
   ```

6. **Construct pegRNA sequence**:
   ```python
   df['pegrna'] = df['spacer'] + SCAFFOLDS[scaffold_name] + df['rtt'] + df['pbs'] + df['linker']
   ```

7. **Split edit into components** (`split_edit`):
   - `pre_hom`: sequence before the edit (identical in unedited/edited)
   - `min_edit`: `min_u:min_e` (the differing region)
   - `post_hom`: sequence after the edit (identical in unedited/edited)

8. **Extract PAMs** (`u_e_pam`):
   - `u_pam = proto30[PS20_OFFSET + 20 : PS20_OFFSET + 24]` = `proto30[24:28]`
   - `e_pam`: same positions from `full_edited` (may differ if edit overlaps PAM)

9. **Seed edit** (`seed_edit`): counts mismatches in positions 17–19 of protospacer
   (the seed region closest to the PAM).

10. **Fill NaN with empty string**: `df = df.fillna('')`

---

## 5. Key Constants (pe_constants.py)

```python
PS20_OFFSET = 4       # Protospacer starts at position 4 in full_unedited
POST_HOM_END = 4      # 4bp downstream homology after the edit
```

Scaffolds:
| Name | Length | Used by |
|------|--------|---------|
| SpCas9_OG | 80 | Default |
| BlpI_F+E | 84 | Hsu (Liu) |
| OG_F+E | 84 | — |
| GC_F+E | 84 | — |

---

## 6. Efficiency Scale

The `format_pe_df` function does NOT convert percentages to fractions. It uses
`edited_frac` and `indel_frac` as-is. Therefore:

- **Hsu data**: Already fractions (0–1) → use directly
- **DeepPrime**: Percentages (0–62.35) → must divide by 100 before loading
- **PRIDICT v1**: Percentages (0–93.79) → must divide by 100
- **PRIDICT2.0 supplementary**: Percentages (0–100) → must divide by 100
- **PRIDICT2.0 processed (_clamped)**: Already fractions (0–0.934) → use directly

The conversion must happen in the CSV files before `format_pe_df` processes them.

---

## 7. Fold Construction (`split_preset`)

```python
def split_preset(self, idx=None, split_name='split'):
    train_idx = (self.df[split_name] != idx)
    val_idx = (self.df[split_name] == idx)
    return self.make_split('train', train_idx), self.make_split('val', val_idx)
```

- Uses the `split` column in the DataFrame (defaults to NaN)
- For fold `idx`: train = all rows with `split != idx`, val = rows with `split == idx`
- The paper states "fivefold cross-validation stratified by protospacer sequence"
- **No fold assignment code found in the OptiPrime repository** — the `split` column
  must be populated by a separate script or notebook not included in the cloned repo
- Fold construction must ensure all records for the same protospacer go into the same fold

---

## 8. Summary of CSV Format Requirements

Each input CSV must contain at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `spacer` | str | Protospacer in RNA (T will be auto-converted to U) or DNA |
| `rtt` | str | Reverse transcriptase template in pegRNA orientation |
| `pbs` | str | Primer binding site in pegRNA orientation |
| `full_unedited` | str | Full target sequence (≥30bp), protospacer at position 4 |
| `full_edited` | str | Full edited target, same length/orientation as unedited |
| `edited_frac` | float | Editing efficiency as fraction (0–1) |
| `indel_frac` | float (optional) | Indel rate as fraction (0–1), defaults to 0.0 |
| `weight` | float (optional) | Sample weight, defaults to 1.0, must be > 0 |

Lab-specific columns (`scaffold_name`, `motif`, `cas9_type`, `cas9_pam`, `pe_type`,
`time`, `group`) are set by the processor functions based on the filename.
