# OptiPrime input specification

Derived from reading, and then actually importing and executing, the official OptiPrime
training code (`external/optiprime/`, commit `475db8a`). All claims below were verified
by running `process_fname`/`format_pe_df` on toy rows
(`scripts/data/optiprime_compat.py`, `tests/test_optiprime_compat.py`), not just by
reading source.

## Runtime dependency note

`pe_utils.py` (which defines `format_pe_df`) transitively imports `jax`, `flax`, `chex`,
`optax`, `ViennaRNA` and `networkx` at module scope, even though `format_pe_df` itself
only needs `pandas`/`numpy`. These were installed into `.venv` (skipping `rs3`, whose
`scikit-learn` pin fails to build under Python 3.11 — `rs3` is only used by unrelated
scoring functions elsewhere in the file, not by `format_pe_df`). This lets us import and
run the **actual** OptiPrime preprocessing functions rather than a reimplementation, per
`data_collect_prompt.md` §B/§U.

## B.1 Required columns

`format_pe_df` (`scripts/pe/pe_utils.py:43`) asserts these columns exist, with no default:

| column | meaning |
|---|---|
| `spacer` | pegRNA spacer, DNA alphabet in the source CSV (converted to RNA internally) |
| `rtt` | pegRNA RT template (3' extension, edit-proximal part), DNA alphabet |
| `pbs` | pegRNA primer-binding site, DNA alphabet |
| `full_unedited` | genomic target window, protospacer strand, starting 4 bp upstream of the protospacer (`PS20_OFFSET = 4`); `full_unedited[:30]` (`proto30`) must be exactly 30 nt |
| `full_edited` | the same window after editing |

## B.2 Optional columns and their defaults

If absent, `format_pe_df` fills these in before any other processing:

| column | default | meaning |
|---|---|---|
| `scaffold_name` | `SpCas9_OG` | one of `SpCas9_OG`, `OG_F+E`, `BlpI_F+E`, `GC_F+E` |
| `linker` | `''` | sequence between RTT/PBS and any additional 3' motif |
| `motif` | `tevoPreQ1` | 3' pegRNA motif; `'none'` for non-epegRNA designs |
| `cas9_type` | `PE2-Cas9` | one of `PE2-Cas9`, `PEmax-Cas9`, `PE6e/f/g-Cas9` |
| `cas9_pam` | `SpNGG` | PAM variant, e.g. `SpNRCH` |
| `rt_name` | `PE2-RT` | reverse transcriptase variant |
| `pe_type` | `PE2` | `PE2` or `PE4` (PE4 = PE2 + dominant-negative MLH1) |
| `group` | `NO_GROUP` | overwritten by `preprocess_fn` to `{lab}_{cell_type}` |
| `time` | `1.0` | days post-transfection/electroporation the assay was read out |
| `split` | `NaN` | preset CV fold index; unpublished for the historical files |
| `weight` | `1.0` | per-row training weight; rows with `weight <= 0` are dropped |
| `edited_frac` | `0.0` | measured editing efficiency, **fraction** in [0, 1] |
| `indel_frac` | `0.0` | measured indel rate, fraction in [0, 1] |

## B.3 Filename convention (source of experimental metadata)

`preprocess_fn` (in each training script) reads `p.stem.split('_')` and dispatches by the
first token to `scripts/pe/pe_datasets.py::process_fname`, which sets `scaffold_name`,
`motif`, `cas9_type`, `cas9_pam`, `pe_type`, `time` from the remaining filename tokens
(see `reports/optiprime_filename_context_map.csv` for the full decode table). It also
always sets `df['group'] = f'{lab_name}_{cell_type}'` from `name_parts[0]`, `name_parts[1]`.

**Filenames are the only place OptiPrime learns cell type and lab identity** — this
information does not need to be, and is not, present as a data column.

## B.4 What `format_pe_df` computes/derives (not required as input)

* `unedited = 1 - (edited_frac + indel_frac)`, `edited = edited_frac` (renamed)
* `proto30 = full_unedited[:30]` if not already present
* `spacer`/`rtt`/`pbs` are converted `T`→`U` (DNA to RNA) in place
* `pegrna = spacer + SCAFFOLDS[scaffold_name] + rtt + pbs + linker`
* `pre_hom`, `min_edit` (`"{min_u}:{min_e}"`), `post_hom` via `split_edit` — strips the
  shared prefix then shared suffix of `full_unedited`/`full_edited`, identical in
  approach to our own `pe_rankformer.data.seqops.diff_window`
* `u_pam`, `e_pam` at `proto30[24:28]` and the corresponding edited-window slice
* `seed_edit`: number of mismatches in `full_[un]edited[21:24]` (the seed, adjacent to
  the nick)
* `hom_len = len(post_hom)`

## B.5 Row filter (the complete filter — see `reports/dataset_reconstruction_status.md`
for why this alone does not explain the Liu 74,769→65,594 gap)

```python
df = df[df['weight'] > 0]
df = df.dropna(subset=['unedited', 'edited', 'weight']).reset_index()
```

Since `unedited = 1 - (edited_frac + indel_frac)`, a null `indel_frac` also drops the row.
No other row-level filter exists in `format_pe_df`.

**Upstream bug found while testing.** If every row of a file fails the filter,
`format_pe_df` crashes with `ValueError: Columns must be same length as key`, because
`df.apply(split_edit, axis=1, result_type='expand')` on a zero-row frame returns zero
columns rather than three, and the subsequent `df[['pre_hom','min_edit','post_hom']] = ...`
assignment fails. This cannot occur for any real per-context file we assemble (some rows
always survive), but `pe_rankformer.data.optiprime_compat.format_pe_df` pre-checks and
returns an empty frame instead of calling into the buggy path, so our wrapper never
crashes on adversarial/toy inputs.

## Verification

`tests/test_optiprime_compat.py` imports `process_fname` and `format_pe_df` directly from
`external/optiprime` and checks, on toy rows, that: (1) a `Liu_HEK293T_x_PE2.csv`-style
filename produces `cas9_type='PEmax-Cas9'`, `motif='tevoPreQ1'`, `time=3.0`; (2) a
`weight=0` row is dropped; (3) a row with `indel_frac=NaN` is dropped; (4) `pegrna`
concatenation matches our own PBS/RTT orientation convention
(`pe_rankformer.data.schema`).
