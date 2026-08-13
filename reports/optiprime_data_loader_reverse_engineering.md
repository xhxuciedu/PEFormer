# Reverse-engineering the OptiPrime training corpus

Source of truth: the official OptiPrime repository, cloned to `external/optiprime/`.

```
repo:   https://github.com/alvin-hsu/optiprime-src
commit: 475db8a  "Initial commit for publication"
```

Paper: Hsu, Chen, Li et al., *Mechanistic machine learning for prediction of prime
editing outcomes*, Nature Biotechnology (2026), doi:10.1038/s41587-026-03261-7.
Preprint: doi:10.64898/2026.02.20.706353.

> Access note: the Nature article is paywalled (redirects to `idp.nature.com`) and
> bioRxiv is behind Cloudflare, which blocked both `curl` and the fetch tool. The
> findings below therefore come from the released source code and released model
> weights, which the task spec (§5) designates as the primary source of truth.
> No Zenodo deposit of the training corpus was found.

---

## 1. How the training corpus is assembled

`scripts/pe/1_train.py` builds the dataset with:

```python
dataset = RxDataset.load_dir(path=train_args.data_path,
                             rx_graph=train_args.rx_graph,
                             rx_inputs=ALL_INPUTS,
                             preprocess_fn=preprocess_fn)
train_ds, val_ds = dataset.split_preset(train_args.val_split)
```

`RxDataset.load_dir` (`reaction/rx_dataset.py:208`) simply globs `*.csv` in one
directory and concatenates every file:

```python
for p in path.glob('*.csv'):
    df = pd.read_csv(p)
    ...
return RxDataset.concatenate(datasets, ...)
```

**There is no per-study subsetting, sampling, or study-level filter in the loader.**
The corpus is exactly "every row of every CSV placed in the data directory", after the
row filters in §3. The composition of the 297,962 rows is therefore determined by which
CSV files the authors put in that directory and how they were pre-filtered upstream —
not by logic we can read off the training script.

## 2. Metadata is encoded in the file names

`preprocess_fn` in `scripts/pe/1_train.py` derives the experimental group from the file
name, then `process_fname` (`scripts/pe/pe_datasets.py`) dispatches per lab:

```python
name_parts = p.stem.split('_')
lab_name, cell_type = name_parts[0], name_parts[1]
df['group'] = f'{lab_name}_{cell_type}'
```

so every training file is named `{lab}_{cell_type}_{...}_{...}.csv`.

| Lab prefix | Handler | Fields set | Assay time (days) |
|---|---|---|---|
| `Liu` | `process_liu` | `scaffold_name='BlpI_F+E'`, `motif='tevoPreQ1'`, `cas9_type='PEmax-Cas9'`, `pe_type` from name | 3.0 HEK293T, 5.0 HeLa |
| `Schwank` | `process_schwank` | `cas9_type` = `PEmax-Cas9` if name part ends in `max` else `PE2-Cas9`; `pe_type` = first 3 chars | 7.0 |
| `Kim` | `process_hkim` | `cas9_type` (max), `pe_type`, `motif='tevoPreQ1'` if `-e` in name else `none`, `cas9_pam='SpNRCH'` if `NRCH` in name else `SpNGG` | 8.0 if lib is `LibClinvar` else 7.0 |
| `YKim` | `process_ykim` | `cas9_type='PE2-Cas9'`, `pe_type='PE2'`, `motif='none'`, `cas9_pam` from name part 2 | 3.0 |

Constraints asserted by `process_liu`: `cell_type in ['HEK293T','HeLa']` and
`pe_type in ['PE2','PE4']` — exactly the four contexts in the Supplementary workbook.

Lab identities:

* **Liu** = this study (David Liu lab) — the 74,769 new measurements.
* **Kim** = Hyongbum Henry Kim lab — DeepPrime (Yu et al., *Cell* 2023).
* **Schwank** = Gerald Schwank lab — PRIDICT (Mathis et al. 2023) and PRIDICT2.0
  (Mathis et al. 2025).
* **YKim** = a PAM-variant dataset handler that is **not** represented in the released
  weights (see §4), so it appears to be unused in the final model.

## 3. Row-level filters actually applied

`format_pe_df` (`scripts/pe/pe_utils.py:43`) is the only place rows are dropped:

```python
df['unedited'] = 1 - (df['edited_frac'] + df['indel_frac'])
df = df.rename(columns={'edited_frac': 'edited', 'indel_frac': 'indel'})
df = df[df['weight'] > 0]
df = df.dropna(subset=['unedited', 'edited', 'weight']).reset_index()
df = df.fillna('')
```

That is the complete filter set:

1. `weight > 0` — rows with zero weight are dropped "for efficiency". `weight`
   defaults to `1.0` when the column is absent, so this only bites where an upstream
   pipeline already assigned zero (presumably a read-depth threshold).
2. Non-null `unedited`, `edited`, `weight`. Because
   `unedited = 1 - (edited_frac + indel_frac)`, a missing **indel** rate propagates to
   a null `unedited` and drops the row too. The code even flags this:
   `# FIXME: Indels?`.

**Notably absent**: no filtering by edit size, edit type, editor version, cell type,
epegRNA status, PBS/RTT length, synthetic vs endogenous target, assay time point, target
length, or duplicate detection. There is no de-duplication across studies at all.

Required input columns (asserted): `spacer`, `rtt`, `pbs`, `full_unedited`,
`full_edited`. Optional columns and their defaults:

| column | default |
|---|---|
| `scaffold_name` | `SpCas9_OG` |
| `linker` | `''` |
| `motif` | `tevoPreQ1` |
| `cas9_type` | `PE2-Cas9` |
| `cas9_pam` | `SpNGG` |
| `rt_name` | `PE2-RT` |
| `pe_type` | `PE2` |
| `group` | `NO_GROUP` |
| `time` | `1.0` |
| `split` | `NaN` |
| `weight` | `1.0` |
| `edited_frac` | `0.0` |
| `indel_frac` | `0.0` |

## 4. The 12 experimental groups — recovered from the released weights

Each released fold checkpoint stores a per-group scaling factor for every mechanistic
rate. `weights/model_{1..5}/log_rates/*.pkl` each contain a `group_factors` dict whose
keys are the `{lab}_{cell_type}` group labels. All 9 rates in all 5 folds agree on the
**same 12 groups**:

| Lab | Groups | n |
|---|---|---:|
| Liu (this study) | `Liu_HEK293T`, `Liu_HeLa` | 2 |
| Kim (DeepPrime) | `Kim_HEK293T`, `Kim_HeLa`, `Kim_A549`, `Kim_DLD1`, `Kim_HCT116`, `Kim_MDA-MB-231`, `Kim_NIH3T3` | 7 |
| Schwank (PRIDICT / PRIDICT2.0) | `Schwank_HEK293T`, `Schwank_K562`, `Schwank_U2OS` | 3 |
| **Total** | | **12** |

This is a hard, code-derived constraint on the reconstruction:

* Only **three** prior labs contribute (consistent with references 54–56 being
  DeepPrime, PRIDICT and PRIDICT2.0).
* **`YKim` does not appear.** Despite `process_ykim` existing in the loader, no
  `YKim_*` group factor was learned, so PAM-variant datasets under that prefix were not
  in the corpus that produced the released models.
* The Kim contribution must span 7 cell lines, which matches DeepPrime's multi-cell-type
  "DP_variant" screens (293T, HeLa, A549, DLD1, HCT116, MDA-MB-231, NIH3T3) and rules
  out a Kim contribution limited to the HEK293T base library.
* The Schwank contribution must span HEK293T, K562 **and U2OS**.

A "context" in the proposal's sense (40 of them) is finer than a group: it is the
combination of group with `pe_type`, `cas9_type`, `motif` and `cas9_pam`, which the file
naming scheme varies within a lab.

## 5. Cross-validation folds

`split_preset` (`reaction/rx_dataset.py:135`) reads a **preset integer `split` column**
carried in the CSVs:

```python
train_idx = (self.df[split_name] != idx)
val_idx  = (self.df[split_name] == idx)
```

The protospacer-stratified fold assignment is therefore baked into the released data
files, not computed at train time. Since those CSVs are not published, **the published
fold assignment cannot be recovered from the repository**, and we must generate our own
deterministic protospacer-grouped folds (task spec §10). Baselines and PE-RankFormer
will be compared on our folds, which is fair as long as every model is evaluated on the
same held-out rows.

The repo also hashes grouping keys for leakage control, which confirms the intended
grouping level:

```python
df['spacer_hash']  = df['spacer'].apply(deterministic_hash)
df['pegrna_hash']  = df['pegrna'].apply(deterministic_hash)
df['edit_hash']    = df['min_edit'].apply(deterministic_hash)
```

`deterministic_hash` is `sha256(s)[:10]` (`scripts/utils.py:49`).

## 6. Sequence conventions (needed for baseline inference)

From `scripts/pe/pe_constants.py` and `pe_utils.py`:

* `PS20_OFFSET = 4`: `full_unedited` starts **4 bp upstream** of the 20 bp protospacer.
  `proto30 = full_unedited[:30]` is asserted to be exactly 30 nt, so `full_unedited`
  must extend at least 4 + 20 + 6 bases.
* The nick therefore sits at index `PS20_OFFSET + 17 = 21`; `seed_edit` compares
  `full_[un]edited[21:24]`, and `u_e_pam` reads the PAM at `full_unedited[24:28]`.
* `spacer`, `rtt`, `pbs` are stored as **RNA** (`T`→`U`) and concatenated with a named
  scaffold into `pegrna = spacer + SCAFFOLD + rtt + pbs + linker`, i.e. `rtt` and `pbs`
  are in **pegRNA orientation**, matching the convention chosen for our canonical schema.
* `split_edit` derives the edit by stripping the shared prefix and then the shared
  suffix of `full_unedited` / `full_edited` — algorithmically identical to our
  `pe_rankformer.data.seqops.diff_window`, so our `edit_type`/`edit_position` fields are
  computed the same way OptiPrime computes `min_edit`.
* Known scaffolds: `SpCas9_OG`, `OG_F+E`, `BlpI_F+E`, `GC_F+E`. Known editors:
  `PE2-Cas9`, `PEmax-Cas9`, `PE6e/f/g-Cas9`. Known cell types:
  `HEK293T, HeLa, A549, HAP1, K562, U2OS, DLD1, MDA-MB-231, NIH3T3` — note `HAP1` is
  declared but has no group factor, another sign that the declared vocabulary is wider
  than the corpus actually used.

**Consequence for the Hsu data**: the Supplementary Lib-MMR targets begin exactly at the
protospacer (offset 0), and Lib-CV targets begin 3–4 nt upstream. OptiPrime's
`proto30`/`PS20_OFFSET` convention needs 4 nt of upstream context that the Supplementary
workbook does not provide for Lib-MMR. Running the official OptiPrime model on our Hsu
test rows will require reconstructing that upstream context (from the library construct
or genome) and must be documented as a deviation if it cannot be recovered exactly.

## 7. What this does and does not pin down

Established from code and weights:

* the corpus is a plain concatenation of per-context CSVs — no clever subsetting;
* only two row filters exist (`weight > 0`, non-null outcome/weight);
* exactly 12 lab×cell groups, from Liu, Kim (7 cell lines) and Schwank (3 cell lines);
* `YKim` and `HAP1` are declared but unused;
* fold assignments were precomputed and are not recoverable;
* sequence/orientation conventions.

Not established, and therefore the open risk for exact reconstruction:

* which specific DeepPrime and PRIDICT/PRIDICT2.0 tables were exported, and what
  read-depth threshold set `weight` to zero upstream;
* whether measurements duplicated between DeepPrime and PRIDICT releases were removed
  before export (the loader would not remove them).

These are quantified in `reports/dataset_reconstruction_status.md`.

## 8. Update: exact per-partition counts recovered from the paper

After this report was first written, the user supplied the OptiPrime main-text and
Supplementary Information PDFs (`data/raw/hsu2026/hsu2026_main_text.pdf`,
`data/raw/hsu2026/hsu2026_supplementary_info.pdf`). Supplementary page 11 contains a bar
chart ("OptiPrime 5-fold cross-validation performance (all datasets)") with 42 bars, each
labeled with lab, cell type, a PEmax/epegRNA/MLH1dn/NRCH design flag, and its exact n.
Transcribed into `data/manifests/optiprime_context_counts.csv`, the 42 values sum to
**exactly 297,962** with no adjustment — see `reports/dataset_reconstruction_status.md`
for the full table and cross-checks. This supersedes §7 above as the authoritative
per-partition target for reconstruction (rather than only the lab-level group identity
recovered from the model weights).

The Data Availability statement (main text, page ~22) confirms no processed training
corpus was deposited: raw sequencing is on SRA (BioProject `PRJNA1314411`, plus
reanalyzed public accessions `PRJNA735408`, `PRJNA1055086`, `PRJNA1211588`), and code is
at `github.com/alvin-hsu/optiprime-src` (already cloned) and
`github.com/alvin-hsu/optiprime-front` (website front end, not needed here). This confirms
the per-partition figure is the best available ground truth for reconstruction.
