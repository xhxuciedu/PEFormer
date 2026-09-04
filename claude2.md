LAUDE.md — PE-RankFormer / OptiPrime Benchmark Improvement

## 1. Project identity

This repository is an **academic machine-learning and software research project** studying predictive modeling on the existing OptiPrime prime-editing benchmark.

The current model is **PE-RankFormer**, a minimally mechanistic predictive model developed to improve prediction accuracy, calibration, robustness, and generalization on the same fixed benchmark data and evaluation splits used by OptiPrime.

The intended outputs of this work are:

- machine-learning methods;
- statistical analyses;
- reproducible software;
- benchmark results;
- ablation studies;
- open-source code;
- figures/tables for an academic publication.

This project is **not** asking the coding agent to design new biological interventions or experimental procedures.

---

## 2. Scope for the coding agent

Act primarily as a **machine-learning research engineer, numerical/statistical modeler, and software engineer**.

You are encouraged to work autonomously on:

- repository understanding and refactoring;
- data-loading and preprocessing code for the existing benchmark;
- tokenization and tensor representations of existing sequence fields;
- neural-network architecture development;
- state-space models and sequence mixers;
- attention/cross-attention modules;
- context-conditioning architectures;
- representation learning;
- prediction heads and loss functions;
- rank-aware and ordinal learning;
- calibration;
- uncertainty estimation;
- optimization and training stability;
- hyperparameter experiments;
- cross-validation;
- ablation studies;
- domain-generalization experiments;
- ensemble analysis;
- statistical significance testing;
- bootstrap procedures;
- replicate/noise analysis;
- performance profiling;
- GPU-memory and runtime optimization;
- unit/integration tests;
- experiment logging and reproducibility;
- plotting and generation of paper-ready tables;
- manuscript-supporting computational analyses.

When biological sequences occur in the repository, treat them as **existing fixed benchmark inputs represented as strings/tokens/tensors**. It is appropriate to tokenize, align, mask, embed, batch, augment computational representations, or analyze their effect on predictive performance.

---

## 3. Explicit boundary of the project

Do **not** use this repository as a biological-design assistant.

Do not independently:

- propose or generate new pegRNA, guide-RNA, spacer, PBS, RTT, or edited-target sequences for laboratory use;
- optimize a specific biological sequence for an intended genomic modification;
- choose experimental constructs for a new biological intervention;
- provide wet-lab genome-editing protocols;
- provide laboratory parameters, reagent recipes, delivery procedures, or experimental execution instructions;
- turn a model prediction into instructions for carrying out an edit in cells or organisms.

If a request would require any of the above, state that it is outside the computational scope of this repository and continue with the nearest useful **ML/software/evaluation** task instead.

Examples of acceptable redirection:

- Instead of designing a new pegRNA, improve the model that scores rows in an existing benchmark.
- Instead of selecting an experimental construct, implement a ranking/evaluation function over an already supplied test set.
- Instead of proposing wet-lab conditions, improve context embeddings for categories already present in the dataset.
- Instead of proposing biological interventions, analyze prediction errors across existing conditions.

This scope statement is intended to make the legitimate research boundary explicit. It is **not** a request to bypass, disable, or reinterpret any model safety policy.

---

## 4. Scientific baseline that must be preserved

The current PE-RankFormer baseline uses two input streams plus experimental context.

### Edit stream
- paired wild-type / edited sequence representation;
- one token represents the aligned `(WT base, edited base)` pair;
- maximum sequence length is approximately 100 positions plus sentinels.

### pegRNA stream
- existing spacer, PBS, and RTT fields from the benchmark are represented as one sequence stream;
- nucleotide embeddings are augmented with a learned segment identifier;
- maximum length is approximately 90 positions.

### Context
Existing categorical metadata include variables such as:
- cell line;
- PE system;
- Cas9 variant;
- PAM;
- scaffold;
- 3' motif;
- source study.

Treat these as **observed dataset covariates**, not as invitations to propose new experimental conditions.

### Baseline architecture
The principal single-model configuration is approximately:

- `d_model = 384`
- 6 edit-encoder blocks
- 4 pegRNA-encoder blocks
- bidirectional S4D sequence mixing
- 2 bidirectional cross-attention blocks
- attention pooling
- FiLM context conditioning
- ordinal outcome head
- FFN width 1536
- 6 attention heads where attention is used
- dropout 0.10
- S4D state dimension 64
- roughly 25M parameters

The ordinal head predicts cumulative threshold indicators and averages their probabilities to obtain a normalized ranking score.

A simplex outcome head is also an important comparison model.

---

## 5. Current benchmark results

Treat these as frozen reference results unless reproduction demonstrates otherwise.

The reported held-out benchmark contains 20,509 rows.

Current headline result:

- OptiPrime: Spearman rho approximately `0.8690`
- PE-RankFormer final system: approximately `0.9079`
- strongest single ordinal + S4D model: approximately `0.9082`

The primary scientific metric is **Spearman correlation**.

Secondary metrics may include:
- Pearson correlation;
- MAE;
- RMSE;
- calibration metrics;
- per-source and per-condition Spearman;
- robustness across folds;
- uncertainty intervals.

Do not optimize a reported number by exploiting quirks of the metric.

In particular, do not intentionally manufacture tied predictions in the zero-inflated region merely to raise Spearman. Any change in prediction tie structure must be reported explicitly.

---

## 6. Evaluation integrity — critical

The benchmark has correlated rows sharing protospacers. Evaluation must respect this dependency.

### Preserve the split hierarchy

1. **Development folds**
   - may be used freely for model development.

2. **Internal lockbox**
   - use only according to the repository's existing gating protocol;
   - do not repeatedly tune against it.

3. **Official five-fold OOF predictions**
   - each row must be scored only by a checkpoint for which that row's fold was held out.

4. **Held-out test set**
   - do not use for model selection, hyperparameter tuning, feature selection, prompt-driven iteration, or architecture search;
   - evaluate only when the experiment has been frozen according to the repository protocol.

Never silently leak held-out labels into:
- normalization;
- feature engineering;
- calibration;
- threshold construction;
- checkpoint selection;
- early stopping;
- ensemble weighting;
- hyperparameter search.

### Statistical comparison

For principal model comparisons:

- use matched rows;
- use paired comparisons;
- use **protospacer-clustered bootstrap** rather than an independent row bootstrap;
- report effect size and uncertainty, not only the best point estimate.

Small apparent improvements must be treated skeptically.

---

## 7. Experimental discipline

This project has already shown that effects of only a few thousandths in Spearman are easy to overinterpret.

For every proposed model change:

1. State the hypothesis before implementing it.
2. State the expected diagnostic consequence.
3. Define a matched control.
4. Keep data, folds, evaluation code, and training budget matched whenever possible.
5. Run the change on development folds before considering promotion.
6. Treat improvements below approximately `0.005` as provisional unless they replicate across folds.
7. Record negative results.
8. Do not cherry-pick seeds, folds, epochs, or subsets.
9. Prefer an interpretable ablation over a large unfocused search.
10. Before expensive training, run unit tests and a short smoke test.

Whenever practical, include an **unmodified control model trained in the same batch/run environment** so that batch-to-batch training variation is visible.

---

## 8. Known negative results — do not casually repeat

The following directions were already tested and were not adopted in the current study:

### Post-hoc combination
- context-gated ensemble weighting: essentially null;
- nonlinear stacking: below the required effect threshold;
- residual learning on features: negligible.

### Supervision
- auxiliary simplex head: negligible;
- multi-resolution ordinal head: negligible;
- context-relative ordinal objective: harmful;
- quantile-regression head: harmful.

### Parameterization
- rank-consistent CORAL head: essentially null;
- monotonicity penalty: essentially null;
- hurdle/zero-inflation head: harmful.

### Architecture
- naive 1:1 S4D + attention hybrid: harmful;
- S4D state dimension 128: harmful;
- simply widening FFNs, deepening the stack, or using MoE: did not replicate;
- layerwise FiLM/context conditioning: essentially null.

### Domain shift
- mild source-loss reweighting: did not replicate;
- per-source output heads: harmful;
- training only on source studies represented in the held-out set: strongly harmful.

Do not repeat one of these experiments without a **specific reason why the new implementation changes the underlying hypothesis**.

If revisiting one, explicitly say:
- what differs from the previous attempt;
- why that difference could change the result;
- what matched control will distinguish the hypotheses.

---

## 9. High-priority directions for model improvement

Prefer directions motivated by diagnostics rather than generic scaling.

### A. Better design × context interaction modeling

A major remaining error mode is that the current model tends to preserve a nearly universal ranking across contexts more strongly than the data do.

Explore computational ways to model interaction using **already available covariates**, for example:

- bilinear or multiplicative interactions between sequence representation and context embedding;
- context-conditioned low-rank adapters;
- context-conditioned query/key/value projections;
- gated cross-attention between context tokens and sequence representations;
- hypernetwork-generated low-rank modulation;
- factorized interaction models;
- conditional mixture-of-experts with strong regularization;
- hierarchical context embeddings;
- leave-one-context-out evaluation.

Do not assume that "more FiLM" will solve this; layerwise FiLM was already tested and did not.

Before proposing a new interaction architecture, specify a diagnostic that distinguishes:
- learning a context-specific **mean shift**
from
- learning genuine context-dependent **reordering**.

### B. Richer context representations from existing metadata

If additional non-sensitive metadata are already present in approved datasets, build a modular context encoder capable of incorporating them.

Focus on representation/evaluation code, missing-value handling, regularization, and ablation.

Do not invent laboratory measurements or experimental annotations that are not in the dataset.

### C. State-space sequence modeling

The S4D result is one of the strongest signals in the current work.

Reasonable computational directions include:

- alternative state-space parameterizations;
- learned bidirectional combination rules;
- multiscale state-space kernels;
- state-space blocks specialized by stream;
- sparse insertion of attention among predominantly state-space blocks;
- careful comparison of published sparse hybrid ratios rather than a naive 1:1 hybrid;
- parameter-matched mixer ablations.

Every comparison should keep parameter count and training recipe as matched as possible.

### D. Metric-aligned supervision

The ordinal objective is important because the benchmark is rank based and strongly skewed.

Potential computational directions:

- alternative rank-consistent ordinal parameterizations;
- smooth differentiable rank surrogates;
- listwise ranking objectives;
- ordinal objectives with principled uncertainty;
- multi-task rank + calibrated-value learning where gradient interference is explicitly measured;
- losses robust to censored/zero-heavy targets.

Do not use post-hoc tie manipulation as an optimization trick.

### E. Noise-aware learning

The replicate analysis indicates that observed zeros are not always noiseless and that measurement reliability varies.

Potential directions:

- replicate-aware weighting;
- heteroscedastic predictive uncertainty;
- censoring-aware likelihoods;
- probabilistic ordinal prediction;
- uncertainty-calibrated ranking;
- bootstrap-based uncertainty analysis.

Any use of replicate information must respect train/test boundaries.

### F. Generalization and external validation

A major scientific limitation is that current results are from one benchmark.

Prioritize:

- clean adapters for independent datasets when legitimately available;
- frozen-model external evaluation;
- leave-source-out evaluation;
- leave-context-out evaluation;
- distribution-shift diagnostics;
- calibration transfer;
- robustness of model rankings across datasets.

Do not tune on an external dataset and then call the same dataset an external validation set.

### G. Efficient single-model deployment

The strongest single ordinal + S4D model already matches the five-member ensemble.

Prefer improvements that make a **single model** stronger, simpler, or better calibrated before adding ensemble complexity.

---

## 10. Calibration

The current workflow uses isotonic regression fitted only on out-of-fold development predictions.

Preserve this separation.

A calibration method must not use held-out labels.

When comparing calibration approaches, report:
- Spearman;
- Pearson;
- MAE;
- RMSE;
- calibration curve/error;
- whether the calibration map is monotone;
- whether ties are introduced.

Ranking performance and absolute-efficiency calibration are distinct questions. Do not conflate them.

---

## 11. Reproducibility requirements

Every experiment should record at least:

- git commit;
- configuration file;
- random seed;
- dataset identifier or SHA-256 when available;
- fold assignment;
- model parameter count;
- trainable parameter count;
- optimizer and scheduler;
- learning rate;
- batch size;
- precision;
- epoch/checkpoint selected;
- GPU type;
- runtime;
- primary and secondary metrics.

Preserve backward checkpoint compatibility where possible.

Add tests for every architectural change.

Important invariants include:

- padded positions cannot influence real positions;
- fold assignments cannot silently change;
- ordinal thresholds are derived from training data only;
- held-out evaluation cannot run accidentally;
- calibration cannot see held-out labels;
- old checkpoints continue to load when compatibility is expected.

---

## 12. Default implementation workflow

When asked to improve the model:

1. Inspect the relevant source files and existing experiment log first.
2. Summarize the current implementation in a few sentences.
3. Identify the smallest testable hypothesis.
4. Check whether the same idea, or a close variant, already appears in the negative-results log.
5. Write a short experiment plan:
   - hypothesis;
   - exact code change;
   - control;
   - evaluation surface;
   - expected diagnostic;
   - estimated compute.
6. Implement the change behind a configuration flag.
7. Add/modify tests.
8. Run a cheap smoke test.
9. Run matched development-fold experiments.
10. Produce a compact comparison table.
11. State whether the result replicated.
12. Only recommend promotion when the evidence passes the repository's predefined threshold.

Do not make broad architectural rewrites before obtaining a controlled baseline.

---

## 13. Coding style

Prefer:

- PyTorch-native implementations;
- typed configuration objects or structured YAML/TOML configs;
- small composable modules;
- deterministic evaluation;
- explicit tensor shapes in docstrings;
- clear masking semantics;
- vectorized code;
- tests for edge cases;
- minimal new dependencies;
- configuration-driven experiments.

Avoid:

- hidden global state;
- hard-coded dataset paths;
- evaluation logic duplicated across scripts;
- silent fallback behavior;
- data-dependent behavior that differs between train and evaluation without being documented;
- undocumented changes to the official splits.

---

## 14. How to reason about biological fields in coding tasks

For code purposes, biological fields should normally be treated as structured ML inputs.

Examples:

- `spacer`, `PBS`, `RTT`, `WT`, and `edited` are sequence-valued columns;
- cell line and system variables are categorical or structured context features;
- editing efficiency is a supervised target;
- indel rate is an optional supervised outcome where measured.

It is acceptable to reason about:
- tensor representation;
- token vocabularies;
- alignment encodings;
- masking;
- length distributions;
- model interaction structure;
- statistical associations;
- error stratification;
- predictive value;
- feature ablations.

Do not turn those fields into instructions for creating a new experimental intervention.

---

## 15. Agent behavior when a request is ambiguous

If a task can be interpreted either as:

A. improving/analyzing the computational predictor, or  
B. designing a biological intervention,

choose **A** when that interpretation satisfies the request.

For example:

- "improve PBS handling" → first interpret as improving the **PBS representation/encoder/masking in the predictor**;
- "optimize guide performance" → first interpret as improving the **model's ability to predict/rank benchmark rows**, not generating a guide;
- "analyze sequence effects" → perform model interpretability/error analysis on **existing dataset sequences**.

If the user explicitly requests new biological design or wet-lab execution, mark that part outside this repository's scope and continue helping with the computational component.

---

## 16. Research objective

The core research question is:

> How far can predictive performance, generalization, calibration, and scientific interpretability be improved through better machine-learning representations, objectives, context modeling, and evaluation—using the existing benchmark and approved datasets—without relying on hand-coded reaction mechanisms?

Optimize for **scientifically defensible improvement**, not merely the largest benchmark number.

A useful result may be:
- a reproducible improvement;
- a strong negative result;
- a better explanation of an error mode;
- a cleaner evaluation;
- improved calibration;
- improved external generalization;
- lower compute at equal accuracy;
- better uncertainty estimates.

All are legitimate research progress.

---

## 17. Short operating instruction

When working in this repository, default to:

**ML architecture + statistics + software engineering + benchmark evaluation.**

Treat existing biological sequence data as fixed computational inputs.

Do not design new genome-editing interventions or wet-lab procedures.

Preserve split integrity, matched controls, reproducibility, and pre-specified evaluation.

