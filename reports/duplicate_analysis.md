# Duplicate analysis

Corpus: `data/processed/optiprime_full_297962.parquet` (262,508 rows)

## Method

- `design_fp` = hash(spacer, pbs, rtt, full_unedited, full_edited, cell_type, pe_type) -- identifies the same pegRNA/target/context design regardless of measured outcome.

- `obs_fp` = hash(design_fp, editing_efficiency rounded to 1e-6) -- identifies exact duplicate measurements.

## Results

- Exact duplicate observations (same design *and* same measured efficiency): **10,412**, of which **2** have `edited > 0`.

  The gap (10,410 rows) is an artifact of DeepPrime's zero-inflated efficiency distribution (~41-50% of raw DeepPrime measurements are exactly 0, confirmed against `external/deepprime/data/*.csv` directly) -- many *different* designs coincidentally share `edited == 0.0`, which is not duplication.

- Rows sharing a design fingerprint with at least one other row: **49,708** (29,731 distinct duplicate groups)

- Of those, design fingerprints spanning more than one `source_study` (genuine cross-study overlap, e.g. a design tested in both DeepPrime and PRIDICT2.0): **0**

## Decision

No automatic deduplication applied. Per task spec (§W), OptiPrime's own loader (`RxDataset.load_dir`, see `reports/optiprime_data_loader_reverse_engineering.md`) performs no cross-study deduplication either -- it concatenates every row of every file placed in its data directory. Rows sharing a design fingerprint within the *same* source (e.g. a design measured at multiple replicate timepoints, or the same pegRNA tested under both PE2 and PE4) are expected and retained as independent observations, consistent with the source studies' own experimental design.
