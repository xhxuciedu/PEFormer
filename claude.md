# Research Task: Develop and Evaluate a Pure-Learning Model for Prime-Editing Efficiency Prediction

You are acting as a senior machine-learning researcher and computational biologist. Conduct a rigorous pilot study to determine whether a modern, minimally mechanistic, end-to-end learning framework can outperform current leading prime-editing efficiency predictors, especially **OptiPrime**, **DeepPrime-FT**, and **PRIDICT2.0-HEK**.

The objective is **predictive performance**, not mechanistic interpretability.

The central hypothesis is:

> A modern relational sequence model trained jointly on the full heterogeneous prime-editing dataset, with explicit modeling of the relationships among the unedited target, edited target, pegRNA design, and experimental context, and trained partly for within-target ranking, can match or outperform a strongly mechanistically constrained model such as OptiPrime.

This is a **pilot study in model scope, not data scope**.

**Use the full training corpus if at all possible. The target dataset is the 297,962 PE experiments used in Hsu et al., Nature Biotechnology (2026).**

Do not deliberately train the final pilot model on a reduced subset for convenience.

Small subsets may be used only for debugging, unit tests, and short pipeline smoke tests.

---

# 1. Set up the project and Python environment

## 1.1 Inspect the repository first

Before changing anything:

1. Inspect the current repository.
2. Run `git status`.
3. Examine the existing directory structure and files.
4. Do not overwrite existing work.
5. Identify the Supplementary Excel workbook already placed in `data/`.
6. Identify its relevant Hsu/OptiPrime sheets, particularly the Lib-MMR and Lib-CV tables.

Record the initial state in:

```text
reports/project_inventory.md
```

---

## 1.2 Create a clean Python virtual environment

Create:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Use an appropriate modern Python version, preferably Python 3.11 unless a required package creates a compatibility problem.

Upgrade core packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

Install the libraries needed for the project.

At minimum consider:

```text
torch
numpy
pandas
polars
pyarrow
scipy
scikit-learn
openpyxl
pyyaml
tqdm
matplotlib
statsmodels
einops
tensorboard
pytest
requests
biopython
```

Use a current PyTorch version supporting the installed NVIDIA GPU and BF16.

Check:

```python
torch.cuda.is_available()
torch.cuda.get_device_name()
torch.cuda.get_device_properties()
```

The machine has a single **RTX PRO 6000 Blackwell-class GPU** with ample VRAM. Use it efficiently.

Prefer:

* BF16 mixed precision
* PyTorch scaled-dot-product attention
* Flash attention through PyTorch when available
* `torch.compile()` only if profiling shows a benefit
* pinned-memory data loaders
* sensible CPU worker counts
* no unnecessary CPU/GPU transfers.

Create:

```text
requirements.txt
```

or preferably:

```text
pyproject.toml
```

with reproducible dependencies.

Also save:

```text
reports/environment.txt
```

containing:

* Python version
* PyTorch version
* CUDA version
* GPU name
* GPU memory
* package versions.

---

# 2. Create a clean project structure

Organize the repository approximately as:

```text
.
├── data/
│   ├── raw/
│   │   ├── hsu2026/
│   │   ├── deepprime/
│   │   ├── pridict/
│   │   └── pridict2/
│   ├── interim/
│   ├── processed/
│   └── manifests/
│
├── external/
│   ├── optiprime/
│   ├── deepprime/
│   ├── pridict/
│   └── pridict2/
│
├── src/
│   └── pe_rankformer/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── utils/
│
├── scripts/
│   ├── data/
│   ├── train/
│   └── evaluate/
│
├── configs/
├── tests/
├── results/
├── checkpoints/
├── logs/
└── reports/
```

Adapt this structure intelligently to the existing repository rather than blindly replacing it.

All major commands should be runnable from scripts or a clear CLI.

Avoid putting substantial analysis logic exclusively in notebooks.

---

# 3. Reconstruct the full OptiPrime training dataset

This is a major research task.

The target is:

[
N = 297,962
]

prime-editing experiments.

Hsu et al. state that this combines:

1. their newly generated paired pegRNA-target screens; and
2. data from previous studies cited as references 54–56.

The local Supplementary Tables from Hsu et al. are already available under `data/`.

Your job is to reconstruct as closely as possible the **exact 297,962 observations actually used to train OptiPrime**.

Do not merely concatenate every PE dataset you can find.

---

# 4. Recover the Hsu et al. data first

Inspect the supplied Excel workbook programmatically.

Identify the Lib-MMR and Lib-CV sheets and reproduce the previously observed fact that the four editing-efficiency columns across these libraries yield exactly:

[
74,769
]

nonmissing PE-efficiency measurements.

Reshape them into long format.

Each row should ultimately represent one experiment/design/context combination.

Create a canonical schema such as:

```text
record_id
source_study
source_dataset
source_row_id

target_id
protospacer
spacer
pam

unedited_sequence
edited_sequence

pbs_sequence
pbs_length
rtt_sequence
rtt_length

edit_type
edit_length
edit_position

cell_type
prime_editor
pe_condition
scaffold_type
epegRNA_flag

editing_efficiency
indel_rate

experimental_context_id
```

Not every source dataset will provide every field. Missing metadata should remain explicitly missing rather than being invented.

Verify:

```text
Hsu retained efficiency observations = 74,769
```

Create:

```text
data/processed/hsu2026_74769.parquet
```

and:

```text
reports/hsu_data_validation.md
```

Document exactly how the 74,769 rows were obtained.

---

# 5. Identify exactly which previous studies were used

Read the Hsu et al. 2026 paper carefully, especially:

* Methods
* Supplementary Methods
* Data availability
* references 54–56
* OptiPrime source code
* data loading code
* training scripts
* configuration files.

Clone the **official** public repositories for:

* OptiPrime
* DeepPrime
* PRIDICT
* PRIDICT2.0

into `external/`.

Do not modify those repositories except when absolutely necessary for reproducibility; preferably wrap them from our own code.

Inspect the OptiPrime source code to determine:

1. names of the historical datasets it expects;
2. source laboratory/study labels;
3. source-specific filtering;
4. experimental-condition metadata;
5. preprocessing rules;
6. duplicate handling;
7. efficiency scaling;
8. inclusion/exclusion criteria;
9. target/protospacer definitions;
10. exact row counts whenever available.

Treat the OptiPrime training loader as an important source of truth.

Create:

```text
reports/optiprime_data_loader_reverse_engineering.md
```

---

# 6. Download processed historical data

Prefer **processed pegRNA-level data** over raw SRA sequencing whenever available.

The goal is model training, not re-running all sequencing analyses.

For each prior study:

1. Locate the official paper.
2. Locate its Supplementary Data.
3. Locate its official GitHub repository.
4. Locate deposited processed tables.
5. Locate raw sequencing accession numbers only as a fallback.
6. Record all provenance.

Download the processed data into:

```text
data/raw/pridict/
data/raw/deepprime/
data/raw/pridict2/
```

Do not depend on files that are silently fetched at training time.

Create a machine-readable manifest:

```text
data/manifests/data_sources.csv
```

with:

```text
study
citation
download_source
filename
checksum
download_date
description
rows_raw
rows_retained
notes
```

Calculate SHA256 checksums for every downloaded source file.

---

# 7. Reconstruct the exact 223,193 historical observations

Since:

[
297,962 - 74,769 = 223,193,
]

the prior studies must contribute exactly:

[
223,193
]

observations after the Hsu preprocessing/filtering rules.

Do not assume every measurement from DeepPrime/PRIDICT/PRIDICT2.0 was used.

Determine the exact subset.

Check systematically for filtering by:

* edit size
* edit type
* prime-editor version
* cell type
* pegRNA configuration
* epegRNA status
* PE2/PE4 or other conditions
* synthetic versus endogenous target
* missing efficiencies
* assay time point
* minimum sequencing coverage
* target sequence length
* duplicated observations
* duplicate datasets appearing in multiple publications
* incompatible outcome definitions.

Create a source-level reconciliation table:

```text
source                 raw rows     candidate rows     retained rows
Hsu Lib-MMR
Hsu Lib-CV
PRIDICT ...
DeepPrime ...
PRIDICT2 ...
---------------------------------------------------------------
TOTAL                                               297,962
```

The exact final total should be:

```text
297962
```

---

# 8. Hard data-reconstruction rule

The final pilot training dataset must use the **full reconstructed corpus**.

Therefore:

```text
assert len(full_training_table) == 297962
```

if exact reconstruction is successful.

Do not silently remove observations merely because a feature is unavailable.

If the exact count cannot initially be reproduced:

1. identify the discrepancy;
2. trace the OptiPrime loader further;
3. inspect supplementary tables;
4. inspect source repositories;
5. check duplicated datasets;
6. check source-specific filters;
7. produce a discrepancy table.

Write:

```text
reports/dataset_reconstruction_status.md
```

If exact reconstruction remains impossible after serious effort, preserve a maximally verified dataset, but **do not describe it as the exact OptiPrime corpus**.

Software development and short smoke tests may proceed while reconstruction is being resolved, but final pilot results should preferably use the exact 297,962 observations.

---

# 9. Standardize experimental context

The corpus spans many experimental conditions.

Represent experimental context explicitly rather than pretending all experiments are equivalent.

Create categorical variables such as:

```text
cell_type
prime_editor
PE2_vs_PE4
source_study
source_dataset
scaffold
epegRNA_status
```

Include only information actually available or safely derivable.

Create an integer `experimental_context_id` that uniquely identifies a reproducible context definition.

Write:

```text
reports/context_inventory.csv
```

showing:

```text
context_id
source
cell_type
editor
other_conditions
n_examples
```

---

# 10. Avoid data leakage

This is critical.

Hsu et al. used **protospacer-stratified splitting** because related pegRNAs sharing the same target should never be divided between train and test sets.

Implement a group-based splitting system.

At minimum:

```text
all observations sharing a protospacer must belong to the same fold.
```

Also investigate whether a stricter target-level grouping is needed when multiple protospacers come from effectively the same target/edit.

Create tests verifying zero leakage.

For example:

```python
assert not (
    set(train.protospacer)
    & set(validation.protospacer)
)

assert not (
    set(train.protospacer)
    & set(test.protospacer)
)
```

Use a fixed random seed.

Save the fold assignment permanently:

```text
data/processed/fold_assignments.parquet
```

Do not regenerate folds between model runs.

---

# 11. Pilot-model philosophy

Do **not** reproduce the OptiPrime mechanistic ODE.

Do not impose:

```text
binding -> nicking -> RT -> heteroduplex -> MMR -> product
```

as a fixed computational graph.

The objective is to test whether a high-capacity, modern learner with substantially weaker mechanistic bias can learn the relevant sequence-function relationships directly.

However, useful **structural information about the experimental objects** should be represented explicitly.

This distinction is important:

* acceptable bias: which sequence is WT, edited, spacer, PBS or RTT;
* acceptable bias: cell/editor identity;
* acceptable bias: sequence alignment;
* avoid strong bias: forcing features into predetermined biochemical reaction rates.

---

# 12. Implement one primary model: PE-RankFormer

The pilot should focus on **one architecture with a high probability of success**.

Do not spend large amounts of GPU time on architecture search.

Call the initial model something like:

```text
PE-RankFormer
```

or another clear internal name.

The model should contain four main elements:

1. paired WT/edited sequence representation;
2. segment-aware pegRNA representation;
3. target-pegRNA cross-attention;
4. experimental-context conditioning.

A fifth key component is the training objective:

5. joint efficiency regression + within-target ranking.

---

# 13. Input representation

## 13.1 WT-edited paired sequence

Prime editing predicts a transformation:

[
S_{\mathrm{WT}} \rightarrow S_{\mathrm{edited}}.
]

Align the unedited and desired edited sequences.

Represent each aligned position as a paired token:

```text
(A,A)
(A,C)
(A,G)
...
(C,T)
...
(-,A)
(A,-)
```

where gap tokens allow insertion/deletion representation.

This produces approximately a small finite vocabulary of paired states.

Example:

```text
WT:       A C G C T A
Edited:   A C G T T A

Tokens:   AA CC GG CT TT AA
```

This makes the edit itself immediately visible to the model.

Do not rely on a Transformer to discover the difference between two distant concatenated sequences if an aligned representation is available.

---

# 14. pegRNA encoder

Represent:

```text
spacer
PBS
RTT
```

as nucleotide tokens with segment-type embeddings:

```text
SEG_SPACER
SEG_PBS
SEG_RTT
```

Include:

* nucleotide embedding
* segment embedding
* position embedding.

Do not use a large set of hand-engineered thermodynamic features in the primary pilot.

The purpose is to test sequence learning.

Simple scalar design metadata such as lengths or known categorical edit type may be included as auxiliary inputs if available consistently, but first determine whether the sequence representation already contains the same information.

---

# 15. Core Transformer architecture

Use a compact model appropriate for ~300k observations.

Do not start with a giant foundation model.

Suggested initial scale:

```text
d_model:          256
attention heads:  8
FFN dimension:    1024
dropout:          0.10
```

Suggested:

```text
WT/edit encoder:  4 Transformer blocks
pegRNA encoder:   4 Transformer blocks
cross-attention:  2 bidirectional interaction blocks
```

Target model size:

```text
approximately 10–25 million parameters
```

Adjust if profiling strongly suggests otherwise.

The model should be easily trainable on one RTX PRO 6000 GPU.

---

# 16. Cross-attention

After independent encoding:

[
H_E = f_{\rm edit}(S_{\rm WT},S_{\rm edited})
]

and:

[
H_P = f_{\rm pegRNA}(\text{spacer},\text{PBS},\text{RTT}),
]

use cross-attention:

[
H_E \leftrightarrow H_P.
]

This should allow direct learning of relationships among:

* edit and RTT;
* target and spacer;
* nick-relative sequence context;
* PBS and target sequence;
* edit location and pegRNA geometry.

Do not hard-code which interaction is biologically important.

Let attention learn it.

---

# 17. Experimental-context conditioning

The same sequence/design can behave differently across:

* cell lines;
* editors;
* PE2 versus PE4-like contexts;
* studies/protocols.

Create learned embeddings for categorical context variables.

For the pilot, use a simple and robust conditioning method such as:

```text
sequence representation
+
context embedding
-> FiLM / gated residual conditioning
-> prediction head
```

or concatenate a compact context embedding before the prediction MLP.

Do not begin with a complicated mixture-of-experts architecture unless the simpler model clearly saturates.

The pilot should maximize probability of producing a reliable result rather than architectural novelty for its own sake.

---

# 18. Output

Predict desired editing efficiency:

[
\hat y \in [0,1].
]

Use a sigmoid output or another numerically stable bounded transformation.

Standardize source efficiencies consistently before training.

Make certain whether source tables use:

```text
0–1
```

or:

```text
0–100
```

and convert all to one convention.

Unit-test this.

---

# 19. Regression loss

Use a robust efficiency-prediction objective.

Start with:

```text
Smooth L1 / Huber loss
```

on either:

1. raw efficiency; or
2. a carefully clipped logit transform.

Compare the two cheaply on the validation set if necessary.

Do not conduct an extensive loss-function search.

---

# 20. The key methodological component: within-target ranking

Efficiency prediction is ultimately a selection problem.

For the same target/edit and experimental context, we want the model to rank:

```text
pegRNA_1
pegRNA_2
...
pegRNA_n
```

correctly.

Define ranking groups conservatively.

A ranking group should normally share:

```text
same target/edit
same experimental context
```

and differ primarily in pegRNA design.

Implement pairwise RankNet-style loss.

For a pair with:

[
y_i > y_j,
]

use:

[
L_{\text{rank}}
===============

\log\left(
1+\exp[-(\hat y_i-\hat y_j)]
\right).
]

Total loss:

[
L
=

L_{\text{reg}}
+
\lambda L_{\text{rank}}.
]

Start with a modest ranking weight such as:

```text
lambda_rank = 0.2
```

and test:

```text
lambda_rank = 0
```

as the key ablation.

Do not perform a large hyperparameter sweep.

If measurement noise makes tiny efficiency differences unreliable, introduce a small minimum pairwise difference threshold and document it.

For example, do not force ranking labels for nearly tied observations unless justified.

---

# 21. Efficient ranking-pair sampling

Do not materialize every possible pair.

Within each eligible target/context group:

* sample a limited number of informative pairs per epoch;
* favor pairs with meaningful measured efficiency differences;
* avoid allowing very large groups to dominate training.

Make pair sampling reproducible.

Unit-test the ranking-pair generator.

---

# 22. Training strategy

Use:

```text
optimizer: AdamW
initial learning rate: around 2e-4 to 3e-4
weight decay: around 0.01
warmup: ~5% of steps
scheduler: cosine decay
mixed precision: BF16
gradient clipping: 1.0
```

Use early stopping based primarily on validation Spearman correlation, with regression loss as a secondary diagnostic.

Maximum epoch count can initially be around:

```text
20–30 epochs
```

but actual convergence should determine stopping.

---

# 23. Batch size and GPU utilization

Do not guess the optimal batch size.

Perform an automatic or manual scaling test:

```text
128
256
512
1024
```

until GPU utilization is good without memory instability.

Record:

```text
examples/sec
GPU memory peak
step time
epoch time
```

Use gradient accumulation only if needed.

Create:

```text
reports/compute_profile.md
```

---

# 24. Measure training time rather than assuming it

Before a full run:

1. train for several hundred representative steps;
2. discard warm-up timing;
3. measure stable throughput;
4. estimate epoch duration;
5. estimate complete training time.

Report:

```text
parameters
batch size
sequence lengths
steps/epoch
examples/sec
minutes/epoch
peak VRAM
estimated full runtime
```

The pilot is specifically intended to determine feasibility on one GPU.

---

# 25. Development protocol

Use the full dataset for the actual pilot.

A small subset may be used only for:

```text
parser debugging
unit tests
one-batch overfitting
short smoke test
GPU profiling
```

Do not report subset performance as scientific evidence.

Before large training, confirm the model can intentionally overfit a tiny dataset. This is an important implementation sanity check.

---

# 26. Primary pilot split

Use one predeclared protospacer-disjoint held-out fold as the first pilot test.

Do **not** repeatedly look at this test fold while modifying the model.

Suggested process:

```text
remaining groups:
    training
    validation

locked fold:
    test
```

Within the non-test portion, create a group-disjoint validation set.

Use validation data for:

* learning-rate choice;
* early stopping;
* lambda_rank choice if absolutely necessary.

Keep the test fold locked until the model is finalized.

---

# 27. Pilot ablations

Keep ablations minimal.

The essential comparison is:

### Model A — main model

```text
paired WT/edit representation
+ pegRNA Transformer
+ cross-attention
+ context conditioning
+ regression loss
+ ranking loss
```

### Model B — no ranking loss

Same model with:

```text
lambda_rank = 0
```

This answers whether target-aware ranking training actually matters.

### Model C — no experimental context

If compute permits, remove context embeddings while leaving everything else fixed.

This determines whether joint learning across heterogeneous experiments benefits from explicit context conditioning.

Do not launch dozens of architectural variants.

---

# 28. Published baselines

The three primary published baselines are:

```text
OptiPrime
DeepPrime-FT
PRIDICT2.0-HEK
```

Also retain PRIDICT2.0-K562 when scientifically appropriate.

Use official implementations or official model predictions wherever possible.

Do not reimplement baseline models from scratch unless required.

Record exact versions/commits.

Create wrappers under:

```text
src/pe_rankformer/evaluation/baselines/
```

---

# 29. Be careful about baseline leakage

The historical OptiPrime corpus contains data that were originally used to train DeepPrime and PRIDICT-family models.

Therefore evaluating those published pretrained models on all 297,962 records could produce contaminated results.

The most important apples-to-apples benchmark is the **new Hsu et al. 74,769 observations**, because these were not part of the older model-training datasets.

For the locked Hsu test observations, compare:

```text
PE-RankFormer
OptiPrime
DeepPrime-FT
PRIDICT2.0-HEK
PRIDICT2.0-K562 where applicable
```

Make clear which baseline supports which experimental context.

For older historical datasets, label any evaluation with a possible training-overlap caveat.

---

# 30. Primary performance metrics

Report standard global regression metrics:

```text
Pearson r
Spearman rho
MAE
RMSE
```

But do not stop there.

Global correlation can be misleading because models can perform well simply by learning that some targets are intrinsically easier than others.

---

# 31. Within-target metrics

For every target/context group with enough alternative pegRNAs, calculate:

```text
within-target Spearman rho
```

Report:

```text
macro mean
median
distribution
```

Require an appropriate minimum number of candidates, for example:

```text
n >= 5
```

and separately report sensitivity for smaller thresholds.

This is a key metric.

---

# 32. Top-k pegRNA-selection metrics

For every eligible target group, calculate practical selection metrics.

## Top-1 regret

Let:

[
i^* = \arg\max_i \hat y_i.
]

Then:

[
R_1 =
y_{\max} - y_{i^*}.
]

Lower is better.

---

## Top-3 regret

Among the three model-selected candidates:

[
R_3 =
y_{\max}
--------

\max_{i\in\text{predicted top 3}} y_i.
]

This corresponds to experimentally testing three candidates.

---

## Top-k recall

Measure whether the truly best or top-performing pegRNA appears within:

```text
predicted top 1
predicted top 3
predicted top 5
```

---

## NDCG

Calculate:

```text
NDCG@3
NDCG@5
```

when group size permits.

---

# 33. Target-level uncertainty and confidence intervals

Because pegRNAs within one target are not statistically independent, use **target-level bootstrap resampling**, not naive row-level bootstrapping.

Report 95% bootstrap confidence intervals for:

```text
Spearman
within-target Spearman
top-1 regret
top-3 regret
NDCG
```

When comparing two methods, use paired target-level bootstrap differences.

---

# 34. Stratified evaluation

Report performance separately by:

```text
source study
cell type
prime editor
PE2/PE4-like context
edit type
edit length
PBS length range
RTT length range
```

Do not over-interpret very small groups.

The objective is to understand generalization failures, not to generate many statistically unstable claims.

---

# 35. Hsu-specific benchmark

Because the Hsu 74,769 measurements are especially important and were used for the published comparison among OptiPrime, DeepPrime-FT and PRIDICT2.0, generate a dedicated table:

```text
Model                   Pearson    Spearman    Within-target rho    Top1 regret    Top3 regret
------------------------------------------------------------------------------------------------
OptiPrime
DeepPrime-FT
PRIDICT2.0-HEK
PRIDICT2.0-K562
PE-RankFormer
PE-RankFormer no-rank
```

Where possible, reproduce published baseline metrics independently before interpreting differences.

---

# 36. Full fivefold cross-validation only after pilot success

The initial feasibility pilot should use one locked protospacer-disjoint fold.

If the result is competitive, then run full fivefold cross-validation using the same fold assignments.

Do not conduct fivefold training while the model is still changing.

Once the architecture and hyperparameters are locked:

```text
fold 0
fold 1
fold 2
fold 3
fold 4
```

can be trained sequentially on the single GPU.

Record all run metadata.

---

# 37. Random seeds

During model development, use one fixed seed.

For the finalized pilot architecture, if GPU time permits, run approximately three seeds on the primary locked split.

Do not spend GPU time running many seeds before the model is stable.

Report variability.

---

# 38. Optional external validation

Inspect the Hsu Supplementary Data for arrayed endogenous experiments such as the ATP1A3 validation sets or other datasets not included in OptiPrime training.

If clean independent data are available, evaluate the finalized model there **without fine-tuning**.

Treat this as a valuable secondary test, but do not allow it to block completion of the primary pilot.

---

# 39. Clean software requirements

Code quality matters.

Use:

* type hints where useful;
* dataclasses/config objects;
* YAML configuration files;
* deterministic seeds;
* structured logging;
* reusable dataset classes;
* reusable metrics;
* checkpointing;
* early stopping;
* unit tests.

Avoid:

* giant monolithic scripts;
* hard-coded local paths;
* copy-pasted preprocessing;
* hidden manual Excel edits;
* notebook-only results;
* undocumented dataset transformations.

---

# 40. Required tests

At minimum implement tests for:

```text
sequence alignment
paired-token encoding
PBS/RTT segmentation
efficiency scale conversion
target/protospacer grouping
fold leakage
ranking-pair generation
context encoding
dataset row counts
model forward pass
model output range
checkpoint save/load
```

A critical test should verify:

```text
all records for the same protospacer are assigned to one fold only
```

---

# 41. Reproducibility

Every experiment should have:

```text
config file
git commit
random seed
dataset hash
fold assignment
start/end time
GPU information
metrics
checkpoint path
```

Create a run directory such as:

```text
results/runs/<run_id>/
```

containing:

```text
config.yaml
metrics.json
training_history.csv
predictions.parquet
run_info.json
```

---

# 42. Save predictions, not only summary metrics

For every test example save:

```text
record_id
target_id
protospacer
context
true_efficiency
predicted_efficiency
fold
model
```

This allows later statistical comparison without retraining.

---

# 43. Produce publication-quality figures

Generate at minimum:

1. predicted versus observed editing efficiency;
2. global rank-correlation comparison;
3. distribution of within-target Spearman correlations;
4. top-k regret comparison;
5. performance by experimental context;
6. training/validation curves;
7. ablation comparison.

Use matplotlib.

Make figures clean enough for a manuscript.

Store:

```text
results/figures/
```

as PDF and PNG when practical.

---

# 44. Pilot success criteria

The pilot is scientifically successful if it demonstrates at least one of the following:

### Strong success

PE-RankFormer exceeds OptiPrime on the held-out Hsu data in global Spearman correlation and maintains good calibration/regression performance.

### Very interesting success

Global Spearman is similar to OptiPrime, but PE-RankFormer is significantly better in:

```text
within-target ranking
top-1 regret
top-3 regret
NDCG
```

This would support the hypothesis that explicitly optimizing the selection problem produces a more useful pegRNA design model.

### Feasibility success

The model reaches performance close enough to OptiPrime to justify further architecture development, while training comfortably on one GPU.

### Negative result

If the pure-learning model remains substantially worse despite correct reconstruction, leak-free splitting, good optimization and reasonable tuning, document this clearly.

That result would itself suggest that OptiPrime's mechanistic inductive bias contributes important generalization.

Do not manipulate evaluation choices to force a positive result.

---

# 45. Important scientific controls

Before concluding that the Transformer architecture helps, verify that improvement is not merely due to:

```text
more training data
different splits
test leakage
lab/context information leakage
duplicated observations
different filtering
using test data for hyperparameter tuning
```

Make the comparisons as fair as possible.

---

# 46. Research questions the final pilot should answer

At the end, answer explicitly:

### Q1

Can a minimally mechanistic relational Transformer trained on the full 297,962-example corpus match or outperform OptiPrime?

### Q2

Does explicitly representing:

[
S_{\rm WT}\rightarrow S_{\rm edited}
]

as paired tokens improve prediction?

### Q3

Does explicit experimental-context conditioning improve generalization across heterogeneous PE datasets?

### Q4

Does adding within-target ranking loss improve actual pegRNA selection even if global Pearson/Spearman changes little?

### Q5

Can the full model be trained practically on one RTX PRO 6000 Blackwell GPU?

### Q6

Which errors remain—target-level editability, within-target pegRNA ranking, particular cell types, edit classes, or experimental contexts?

---

# 47. Work systematically

Proceed in this order:

```text
1. environment
2. repository inventory
3. Hsu data extraction
4. historical-data acquisition
5. exact 297,962-row reconstruction
6. dataset validation
7. fold construction
8. model implementation
9. unit tests
10. tiny overfit test
11. GPU profiling
12. full-data main training
13. no-ranking ablation
14. baseline inference
15. locked test evaluation
16. statistical analysis
17. figures
18. final research report
```

Do not jump directly to model training before validating the dataset.

---

# 48. Keep a research log

Maintain:

```text
reports/research_log.md
```

Update it throughout the work with:

```text
date/time
task
finding
decision
unexpected problem
resolution
next step
```

Particularly record all data-reconstruction decisions.

---

# 49. Final deliverables

At completion, the repository should contain:

### Data

```text
data/processed/hsu2026_74769.parquet
data/processed/optiprime_full_297962.parquet
data/processed/fold_assignments.parquet
data/manifests/data_sources.csv
```

### Code

Clean preprocessing, modeling, training and evaluation code.

### Models

Best checkpoint(s).

### Predictions

Per-example test predictions for all evaluated models.

### Reports

```text
reports/project_inventory.md
reports/environment.txt
reports/optiprime_data_loader_reverse_engineering.md
reports/dataset_reconstruction_status.md
reports/compute_profile.md
reports/research_log.md
reports/pilot_results.md
```

### Figures

Publication-quality figures in:

```text
results/figures/
```

---

# 50. Final report structure

Write `reports/pilot_results.md` with:

```text
1. Executive summary
2. Data reconstruction
3. Exact source composition of the 297,962 experiments
4. Dataset quality control
5. Model architecture
6. Training procedure
7. GPU utilization and training time
8. Primary held-out results
9. Comparison with OptiPrime
10. Comparison with DeepPrime-FT
11. Comparison with PRIDICT2.0
12. Within-target ranking results
13. Top-k pegRNA-selection performance
14. Ablation results
15. Failure analysis
16. Conclusions
17. Recommendation for the next phase
```

Be quantitative.

Do not merely state that performance is "good".

---

# 51. Final principle

The purpose of this study is **not to demonstrate that Transformers are fashionable**.

The scientific test is whether the combination of:

[
\boxed{
\text{full heterogeneous training corpus}
+
\text{relational WT/edit representation}
+
\text{target--pegRNA interaction modeling}
+
\text{experimental-context conditioning}
+
\text{within-target ranking}
}
]

can learn prime-editing efficiency more accurately than a model with a strongly prescribed mechanistic architecture.

Keep the pilot focused enough that a negative result is interpretable.

Prioritize:

```text
data correctness
fair evaluation
one strong model
clean implementation
reproducibility
```

over broad exploratory model search.

Start now with **Step 1: inspect the repository, create `.venv`, install and validate the software environment, then begin reconstructing the 297,962-example dataset.**
