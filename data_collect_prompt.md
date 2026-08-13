led Training-Data Reconstruction Instructions

The first major scientific objective is to reconstruct, as closely as possible, the **exact 297,962-example PE training dataset used by Hsu et al. for OptiPrime**.

Do not treat data collection as a generic literature aggregation exercise.

Use the **OptiPrime source code itself as the primary specification** for what Hsu et al. actually loaded and how the records were formatted.

The target arithmetic is:

[
297,962
=======

74,769
+
223,193.
]

where:

* **74,769** measurements come from the new Hsu et al. Lib-MMR / Lib-CV experiments already supplied locally;
* **223,193** measurements must be reconstructed from prior published datasets.

The historical studies cited by Hsu et al. are:

1. Mathis et al. — original PRIDICT study;
2. Yu et al. — DeepPrime study;
3. Mathis et al. — PRIDICT2.0 study.

Do not assume that Hsu et al. used every published measurement from these papers.

The central data-reconstruction problem is:

[
N_{\rm PRIDICT}
+
N_{\rm DeepPrime}
+
N_{\rm PRIDICT2}
================

223,193
]

after applying the same filtering/formatting choices used by OptiPrime.

---

# A. Clone the four authoritative code repositories

Create:

```bash
mkdir -p external
cd external
```

Clone:

```bash
git clone https://github.com/alvin-hsu/optiprime-src.git
git clone https://github.com/uzh-dqbm-cmi/PRIDICT.git
git clone https://github.com/uzh-dqbm-cmi/PRIDICT2.git
git clone https://github.com/yumin-c/DeepPrime.git
```

If one of these repositories has moved or redirects, locate the current official repository associated with the publication and document the change.

Immediately save commit hashes:

```bash
for d in optiprime-src PRIDICT PRIDICT2 DeepPrime; do
    echo "===== $d ====="
    git -C "$d" rev-parse HEAD
done > ../reports/external_repo_commits.txt
```

Do not modify these repositories unless necessary.

Our own conversion code belongs under:

```text
src/pe_rankformer/data/
```

---

# B. Reverse-engineer the exact input format expected by OptiPrime

Before downloading historical data, inspect these files:

```text
external/optiprime-src/scripts/pe/1_train.py
external/optiprime-src/scripts/pe/pe_datasets.py
external/optiprime-src/scripts/pe/pe_utils.py
external/optiprime-src/reaction/rx_dataset.py
```

Read them carefully.

Create:

```text
reports/optiprime_input_specification.md
```

Document every requirement.

## B.1 Required columns

The OptiPrime formatting code requires each source CSV to contain at least:

```text
spacer
rtt
pbs
full_unedited
full_edited
```

The training datasets should additionally contain the experimental outcome fields:

```text
edited_frac
indel_frac
```

and, when needed:

```text
weight
split
```

Do not invent alternative names in the final OptiPrime-compatible intermediate files.

Our unified modeling table may use more descriptive names, but preserve an OptiPrime-compatible representation separately.

---

# C. Understand exactly how OptiPrime filters rows

Replicate the logic from `format_pe_df()`.

The relevant sequence should be reproduced explicitly in our own validation script:

```python
unedited_frac = 1.0 - (edited_frac + indel_frac)
```

Then:

```python
keep = weight > 0
```

and remove rows with missing values in the quantities required by OptiPrime.

Write a test that compares our implementation to:

```text
external/optiprime-src/scripts/pe/pe_utils.py
```

on several toy examples.

Important:

**Count rows both before and after OptiPrime-equivalent filtering.**

For every source file generate:

```text
file
raw_rows
weight_zero_rows
missing_efficiency_rows
invalid_rows
retained_rows
```

Save this as:

```text
reports/source_filtering_counts.csv
```

---

# D. Decode the filename convention used by OptiPrime

This is extremely important because OptiPrime infers experimental metadata from filenames.

Inspect:

```text
external/optiprime-src/scripts/pe/pe_datasets.py
```

The training code recognizes dataset files by prefixes corresponding to:

```text
Liu
Schwank
Kim
YKim
```

Do not arbitrarily rename source CSVs until you understand what these categories correspond to.

The filename determines metadata such as:

```text
cell type
PE2 / PE4
PEmax vs non-max
epegRNA status
Cas9 PAM variant
experimental time
```

Create a table:

```text
reports/optiprime_filename_context_map.csv
```

with columns:

```text
filename_pattern
lab_prefix
cell_type
library
editor
PE_type
PEmax
epegRNA
PAM_variant
time
inferred_by_function
```

Explicitly reproduce the behavior of:

```text
process_liu()
process_schwank()
process_hkim()
process_ykim()
```

Do not assume the historical source is only determined by publication title.

Use the filename parsing rules to infer which individual experiments Hsu intended to include.

---

# E. Reconstruct the local Hsu data first

The existing supplementary Excel file is already in:

```text
data/
```

Locate it automatically rather than hard-coding the exact filename if possible.

Inspect all worksheets.

Identify:

```text
Lib-MMR
Lib-CV
```

or their exact workbook sheet names.

These two tables contain approximately:

```text
10,000 Lib-MMR designs
10,406 Lib-CV designs
```

with editing measurements for the four combinations:

```text
HEK293T + PE2
HEK293T + PE4
HeLa + PE2
HeLa + PE4
```

---

# F. Convert Hsu Supplementary Tables into long format

Write:

```text
scripts/data/extract_hsu2026.py
```

The script should:

1. locate the workbook;
2. load the two relevant sheets;
3. identify the sequence/design columns;
4. identify all four editing-efficiency columns;
5. identify corresponding indel columns;
6. melt the table into one row per design × experimental condition;
7. discard only measurements whose editing efficiency is missing;
8. preserve the original spreadsheet row number and ID.

The resulting long-form table must have exactly:

```text
74,769 rows
```

Verify with:

```python
assert len(hsu_long) == 74769
```

If this assertion fails, stop and diagnose before proceeding.

Do not modify the source Excel file.

---

# G. Preserve Hsu source provenance

For every Hsu observation include:

```text
source_study = "Hsu2026"
source_library = "LibMMR" or "LibCV"
source_sheet
source_excel_row
original_design_id
cell_type
pe_type
editing_efficiency
indel_fraction
```

Also preserve:

```text
spacer
PBS
RTT / extension
unedited target
edited target
barcode
```

when present.

Write:

```text
data/processed/hsu2026_74769.parquet
```

and:

```text
data/processed/hsu2026_74769.csv
```

The Parquet file is the authoritative downstream representation.

---

# H. Create OptiPrime-compatible Hsu CSVs

In addition to our unified table, create source files that could actually be loaded by the OptiPrime training code.

At minimum each row needs:

```text
spacer
rtt
pbs
full_unedited
full_edited
edited_frac
indel_frac
```

and optionally:

```text
weight
split
```

Use filenames compatible with `process_liu()`.

Before deciding filenames, inspect exactly how:

```python
p.stem.split("_")
```

is interpreted by `process_liu()`.

Write an automated test:

```python
from pathlib import Path
from pe_datasets import process_liu
```

and verify that the filenames produce the intended:

```text
cell_type
pe_type
time
motif
cas9_type
```

Do not rely on visual inspection alone.

---

# I. Do not download Hsu SRA initially

The Hsu raw sequencing project is:

```text
PRJNA1314411
```

but do **not** download these FASTQs initially.

The processed efficiencies needed for training are already present in the supplementary workbook.

Raw sequencing is only needed if:

1. a supplementary value is ambiguous;
2. a sequence/design cannot be reconstructed;
3. we want to independently reproduce the experimental quantification.

For the machine-learning pilot, processed labels should be considered authoritative.

---

# J. Identify the three historical source datasets

Hsu et al. explicitly cite three previous studies for model training:

```text
Mathis et al. 2023     PRIDICT
Yu et al. 2023         DeepPrime
Mathis et al. 2024/25  PRIDICT2.0
```

The Hsu paper also identifies the following existing SRA projects analyzed in the study:

```text
PRJNA735408
PRJNA1055086
PRJNA1211588
```

Do not assume which accession corresponds to which processed table solely from the accession number.

Resolve the mapping from:

```text
BioProject metadata
paper Methods
supplementary tables
source repository
```

and record the mapping in:

```text
data/manifests/historical_bioproject_mapping.csv
```

---

# K. PRIDICT 1.0 data acquisition

Begin with the official repository:

```text
external/PRIDICT/
```

Inspect:

```bash
find external/PRIDICT -maxdepth 3 -type f | sort
```

Search for:

```bash
grep -RniE \
"dataset|training|editing|efficiency|pegRNA|csv|xlsx|pickle|parquet" \
external/PRIDICT
```

Also inspect its README for the official Supplementary Files link.

The PRIDICT paper reports a large high-throughput dataset; however, do not assume every row belongs in OptiPrime.

Prefer, in order:

1. processed training/evaluation tables included in GitHub;
2. processed supplementary tables linked by the official repository;
3. supplementary Excel/CSV files from the publication;
4. raw SRA data only if necessary.

Create:

```text
data/raw/pridict/
```

Store the original downloaded files unchanged.

For each candidate file record:

```text
filename
number of rows
column names
cell type
editor
library
edit types
whether efficiency is directly available
```

in:

```text
reports/pridict_inventory.csv
```

---

# L. PRIDICT2.0 data acquisition

Inspect:

```text
external/PRIDICT2/
```

and recursively list files:

```bash
find external/PRIDICT2 -maxdepth 4 -type f | sort
```

Search for:

```bash
grep -RniE \
"Library-Diverse|training|dataset|HEK293T|K562|editing|efficiency" \
external/PRIDICT2
```

Also follow the official repository's Supplementary Files link if the training data are not directly stored in Git.

The PRIDICT2.0 study contains multiple experimental datasets and the model itself was trained on substantially more data than OptiPrime ultimately incorporated.

Therefore:

**do not import every PRIDICT2.0 row automatically.**

Identify candidate datasets separately.

Create:

```text
reports/pridict2_inventory.csv
```

with:

```text
source_file
library_name
cell_type
editor
number_of_rows
edit_types
editing_efficiency_field
possible_optiprime_use
notes
```

---

# M. DeepPrime data acquisition

Inspect:

```text
external/DeepPrime/
```

The official training repository contains training scripts such as:

```text
train_base.py
train_ft.py
```

and expects CSV data under its `data/` directory.

Search:

```bash
find external/DeepPrime -maxdepth 4 -type f | sort
```

and:

```bash
grep -RniE \
"read_csv|data/|ClinVar|HEK|PE2|PEmax|NRCH|editing|efficiency" \
external/DeepPrime
```

Read:

```text
train_base.py
train_ft.py
utils/data.py
utils/preprocess.py
```

Determine:

1. exact input columns expected by DeepPrime;
2. names of source libraries;
3. cell lines;
4. editor versions;
5. PEmax versus PE2;
6. epegRNA conditions;
7. SpNGG versus alternative PAM variants;
8. outcome column.

Create:

```text
reports/deepprime_inventory.csv
```

Do not reconstruct DeepPrime records from model predictions.

We need **experimental measured efficiency labels**.

---

# N. Prefer processed labels over SRA for all historical datasets

The raw SRA projects are a fallback.

Do not initially download hundreds of gigabytes of sequencing data.

The required modeling table should ideally be reconstructed from:

```text
published processed tables
+
source repositories
+
supplementary Excel/CSV files.
```

Use raw FASTQs only when a necessary measurement cannot otherwise be recovered.

If raw data are required, use the NCBI SRA Toolkit:

```bash
prefetch <ACCESSION>
fasterq-dump <ACCESSION> \
    --threads 16 \
    --split-files \
    --outdir data/raw/sra/
```

Compress afterward:

```bash
pigz -p 16 data/raw/sra/*.fastq
```

But do not perform this until the processed-data route has been exhausted.

---

# O. Build source-specific converter scripts

Do not write one giant heuristic parser.

Create separate adapters:

```text
src/pe_rankformer/data/hsu.py
src/pe_rankformer/data/pridict.py
src/pe_rankformer/data/pridict2.py
src/pe_rankformer/data/deepprime.py
```

Each adapter must output the same canonical schema.

For example:

```python
CANONICAL_COLUMNS = [
    "record_id",
    "source_study",
    "source_dataset",
    "source_row_id",

    "spacer",
    "rtt",
    "pbs",
    "full_unedited",
    "full_edited",

    "cell_type",
    "pe_type",
    "cas9_type",
    "cas9_pam",
    "scaffold_name",
    "motif",
    "time",

    "edited_frac",
    "indel_frac",

    "weight",
]
```

Preserve additional source-specific columns separately.

Never discard original IDs.

---

# P. Normalize DNA/RNA orientation carefully

The OptiPrime formatter converts:

```text
spacer
rtt
pbs
```

from DNA `T` notation into RNA `U` notation internally.

Therefore store the canonical source representation consistently.

Recommended:

```text
*_dna
*_rna
```

or preserve the original plus a normalized version.

For example:

```text
spacer_original
spacer_dna
rtt_original
rtt_dna
pbs_original
pbs_dna
```

Before feeding data into our model, standardize all sequence alphabets explicitly.

Never silently mix:

```text
T
```

and:

```text
U
```

across datasets.

Write unit tests for round-trip conversion.

---

# Q. Reconstruct `full_unedited` and `full_edited` only when necessary

These are mandatory for OptiPrime.

If the historical table already provides them, use them directly.

If only:

```text
target sequence
edit position
edit description
```

are supplied, reconstruct the sequences algorithmically.

For every reconstructed sequence:

1. preserve the original source fields;
2. store a flag:

```text
full_sequence_reconstructed = True
```

3. verify the reported edit by aligning:

```text
full_unedited
full_edited
```

4. ensure the minimal edit matches the source annotation.

Reject ambiguous reconstructions rather than guessing.

Create:

```text
reports/reconstructed_sequence_audit.csv
```

---

# R. Derive PBS and RTT only from source design information

Do not infer PBS or RTT purely from generic prime-editing rules unless the source record allows an unambiguous reconstruction.

If source data include:

```text
PBS length
RTT length
target
edit
nick position
```

derive sequences deterministically.

Then test:

```python
len(pbs) == reported_pbs_length
len(rtt) == reported_rtt_length
```

When source design explicitly contains PBS/RTT sequences, those values take precedence.

---

# S. Determine which historical rows Hsu actually used

Once all candidate historical data are available, construct:

```text
data/interim/all_historical_candidates.parquet
```

Do **not** yet combine it with Hsu.

Compare candidate source datasets against the behavior expected by:

```text
process_schwank()
process_hkim()
process_ykim()
```

in OptiPrime.

The filename parser gives strong clues about which experimental contexts were expected.

For every candidate dataset determine whether its metadata can map cleanly to an OptiPrime filename/context.

Create:

```text
reports/historical_context_matching.csv
```

with:

```text
source_study
source_dataset
candidate_rows
optiprime_lab_prefix
cell_type
pe_type
max_status
epeg_status
pam_variant
time
mapping_confidence
include_exclude_reason
```

---

# T. Reconstruct OptiPrime-compatible historical CSV files

For every historical context believed to have been used, generate a CSV whose filename is intentionally compatible with:

```text
process_fname()
```

Do not guess filename structure.

Programmatically test each filename by running the actual OptiPrime function:

```python
from scripts.pe.pe_datasets import process_fname
```

and verifying the inferred metadata.

For each file:

```python
test_df = pd.read_csv(...)
processed = preprocess_fn(path, test_df)
```

Check:

```text
retained row count
group
cell type
PE type
Cas9 type
PAM type
motif
time
```

Save the exact files under:

```text
data/interim/optiprime_compatible/
```

---

# U. Run the actual OptiPrime preprocessing code on our reconstructed files

This is a critical validation step.

Do not merely imitate the Hsu code.

Actually import and run their preprocessing functions where practical.

For each reconstructed CSV:

```python
df = pd.read_csv(path)
df2 = preprocess_fn(path, df)
```

Record:

```text
input rows
output rows
```

Then run the equivalent of:

```python
RxDataset.load_dir(...)
```

on the complete reconstructed data directory.

The resulting concatenated dataset should contain:

```text
297,962 rows
```

If it does, save the complete OptiPrime-compatible directory unchanged as a versioned artifact.

---

# V. Row-count reconciliation

Maintain a live table:

```text
reports/row_count_reconciliation.csv
```

Example structure:

| Source  | Dataset/context      | Raw candidate | After source QC | After OptiPrime filtering |
| ------- | -------------------- | ------------: | --------------: | ------------------------: |
| Hsu     | LibMMR / HEK293T PE2 |           ... |             ... |                       ... |
| Hsu     | LibMMR / HEK293T PE4 |           ... |             ... |                       ... |
| Hsu     | LibCV / HeLa PE2     |           ... |             ... |                       ... |
| Schwank | ...                  |           ... |             ... |                       ... |
| Kim     | ...                  |           ... |             ... |                       ... |
| YKim    | ...                  |           ... |             ... |                       ... |

The final two checks are:

```python
assert hsu_count == 74769
assert historical_count == 223193
assert hsu_count + historical_count == 297962
```

Do not weaken these checks simply to move on.

---

# W. Detect duplicated historical experiments

This is essential because PRIDICT2.0 may reuse or extend data related to earlier PRIDICT experiments.

Construct a conservative experimental fingerprint such as:

```python
fingerprint = hash(
    spacer,
    pbs,
    rtt,
    full_unedited,
    full_edited,
    cell_type,
    pe_type,
    rounded_edited_frac
)
```

Also construct a sequence/design fingerprint excluding outcome:

```python
design_fingerprint = hash(
    spacer,
    pbs,
    rtt,
    full_unedited,
    full_edited,
    cell_type,
    pe_type
)
```

Find:

```text
exact duplicate measurements
same design / same context / different measurements
same design across different publications
```

Do not automatically deduplicate.

First determine whether OptiPrime retained those rows independently.

Document decisions in:

```text
reports/duplicate_analysis.md
```

---

# X. Check efficiency units

Historical publications may encode editing efficiency as:

```text
0–1
```

or:

```text
0–100
```

For each source file examine:

```python
df[col].describe()
df[col].quantile([0, .01, .5, .99, 1])
```

Never infer scale from column name alone.

Our canonical representation must use:

```text
0 <= edited_frac <= 1
```

and:

```text
0 <= indel_frac <= 1
```

Add assertions:

```python
assert ((df.edited_frac >= 0) & (df.edited_frac <= 1)).all()
assert ((df.indel_frac >= 0) & (df.indel_frac <= 1)).all()
```

Report every transformation applied.

---

# Y. Verify biological consistency

For every retained observation:

```python
assert set(full_unedited) <= set("ACGT")
assert set(full_edited) <= set("ACGT")
```

Check spacer length distributions.

Check PBS and RTT length distributions.

Check:

```python
edited_frac + indel_frac <= 1 + tolerance
```

Plot distributions by source.

Create:

```text
results/data_qc/
```

with figures for:

```text
editing efficiency by source
editing efficiency by context
PBS length by source
RTT length by source
edit type by source
edit length by source
cell type frequencies
editor frequencies
```

Large discrepancies may reveal incorrect parsing.

---

# Z. Recover the 40 experimental contexts

Hsu reports that the 297,962 measurements span:

```text
40 experimental contexts
```

After reconstruction, calculate our context count.

Define a context using the metadata actually represented in the OptiPrime code, such as:

```text
lab/source
cell type
PE type
Cas9 type
PAM variant
scaffold/motif status
relevant library/editor configuration
```

Try to reproduce:

```python
assert n_contexts == 40
```

Do not force 40 by arbitrary grouping.

If the count differs, investigate the historical mapping.

This is an important independent validation of dataset reconstruction.

---

# AA. Validate source identity against OptiPrime's lab groups

OptiPrime creates:

```python
group = f"{lab_name}_{cell_type}"
```

and also derives:

```text
spacer_hash
pegrna_hash
edit_hash
```

Reproduce these quantities using the original OptiPrime code where possible.

Generate:

```text
reports/lab_cell_group_counts.csv
```

This gives another way to compare our reconstruction against expected training composition.

---

# AB. Search source code for hidden clues before using raw SRA

Before downloading SRA FASTQs, systematically search OptiPrime code for:

```bash
grep -RniE \
"Schwank|Kim|YKim|LibClinvar|PEmax|NRCH|HEK|K562|HeLa" \
external/optiprime-src
```

Also search the previous model repositories for dataset names found in these matches.

This cross-repository name matching is likely the fastest way to identify the exact historical source files.

---

# AC. Raw SRA data are the last fallback

Hsu states that previously existing sequencing data analyzed in the paper are publicly available under:

```text
PRJNA735408
PRJNA1055086
PRJNA1211588
```

Use them only if the processed data are insufficient to recover:

```text
experimental efficiency
target sequence
pegRNA design
experimental context
```

If FASTQ processing becomes necessary, first reconstruct the exact original pipeline from the source publication.

Do not invent a generic CRISPResso pipeline and assume it matches the authors' processed labels.

For each raw-data reconstruction validate against at least several known published processed values before processing the entire project.

---

# AD. Produce two final datasets

Create two separate artifacts.

## 1. Exact-as-possible OptiPrime-compatible dataset

```text
data/processed/optiprime_training_297962.parquet
```

This should reflect Hsu's training observations and Hsu-compatible fields.

It should contain:

```text
297,962 rows
```

if reconstruction succeeds.

## 2. Rich unified PE modeling dataset

```text
data/processed/pe_full_canonical.parquet
```

This should contain all useful metadata we can recover, including:

```text
record_id
source_study
source_dataset
original_id

spacer
PBS
RTT

full_unedited
full_edited

edit_type
edit_length

cell_type
PE_type
Cas9_type
PAM_variant
scaffold
motif

editing_efficiency
indel_fraction

experimental_context
protospacer
target_group

source_file
source_row
```

The second table will be used by PE-RankFormer.

---

# AE. Produce a complete reconstruction report before model training

Before starting the full GPU training run, write:

```text
reports/training_data_reconstruction.md
```

with the following sections:

```text
1. Target dataset definition
2. Hsu 74,769 reconstruction
3. PRIDICT source data
4. DeepPrime source data
5. PRIDICT2.0 source data
6. Mapping to Liu / Schwank / Kim / YKim
7. Experimental-context reconstruction
8. Filtering rules
9. Duplicate analysis
10. Sequence normalization
11. Missing-data handling
12. Final row counts
13. Final context counts
14. Remaining discrepancies
15. Reproducibility instructions
```

Include a final table:

```text
Source                          Retained observations
----------------------------------------------------
Hsu 2026                        74,769
PRIDICT / Schwank               XXXXX
DeepPrime / Kim                 XXXXX
PRIDICT2 / Schwank              XXXXX
other Hsu-loader source         XXXXX
----------------------------------------------------
Historical total               223,193
Grand total                    297,962
```

Do not fill the `XXXXX` values from assumptions.

Determine them from the actual source data and Hsu loader behavior.

---

# AF. Gate full training on data validation

Do not begin the main scientific training run until one of the following is true.

## Preferred

```text
exact reconstructed count = 297,962
experimental contexts = 40
Hsu count = 74,769
all required sequence fields available
all source mappings documented
```

## Acceptable fallback

If exact reconstruction proves impossible because Hsu did not release a necessary processed intermediate:

1. document precisely what is missing;
2. retain only verified observations;
3. report the exact resulting count;
4. label the dataset:

```text
OptiPrime-compatible reconstructed corpus
```

rather than:

```text
exact OptiPrime training corpus
```

5. continue the pilot only after the discrepancy is scientifically understood.

Do not silently substitute a different training corpus.

---

# AG. Most important principle

The task is **not**:

> download PRIDICT, DeepPrime, and PRIDICT2 data and concatenate them.

The task is:

[
\boxed{
\text{recover the exact experimental observations and contexts that Hsu's loader consumed}
}
]

using:

```text
Hsu supplementary tables
+
published processed datasets
+
source repositories
+
OptiPrime filename/context logic
+
OptiPrime filtering code
```

as mutually cross-checking sources.

Only once this reconstruction is validated should full PE-RankFormer training begin.

