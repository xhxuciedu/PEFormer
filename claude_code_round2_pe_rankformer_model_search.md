# Claude Code Research Prompt — Round 2 Model Optimization for Prime-Editing Efficiency Prediction

## Role

Act as a senior machine-learning researcher and computational biologist.

You are continuing an existing PE-RankFormer project for **prime-editing efficiency prediction**. The first pilot established that a minimally mechanistic relational Transformer can match or exceed OptiPrime depending on evaluation scope. The purpose of this second round is **not interpretation**. The purpose is to **search aggressively but systematically for the best predictive model**.

The primary goal is:

\[
\boxed{\text{maximize held-out prime-editing efficiency prediction performance}}
\]

with special emphasis on **Spearman correlation** and robust cross-context generalization.

Do not spend time trying to explain biological mechanisms unless needed for debugging. We will study interpretation later.

---

# 1. Current verified baseline

Treat the following as the current starting point.

The exact OptiPrime training corpus is now available:

\[
N_{\rm train}=297,962.
\]

The official held-out test set contains:

\[
N_{\rm test}=20,509,
\]

consisting of:

- Liu/Hsu held-out: \(n=9,175\)
- Kim/DeepPrime held-out: \(n=11,334\)

Current matched-protocol held-out performance:

| Scope | PE-RankFormer | OptiPrime | Difference |
|---|---:|---:|---:|
| Full held-out \(n=20,509\) | Spearman \(0.8865\) | \(0.8690\) | \(+0.0175\) |
| Liu only \(n=9,175\) | \(0.8349\) | \(0.8365\) | essentially tied |
| Kim only \(n=11,334\) | \(0.7751\) | \(0.7320\) | PE-RankFormer ahead |

For the full held-out set:

\[
\Delta\rho=+0.0175,
\]

with clustered bootstrap CI approximately:

\[
[+0.007,+0.028].
\]

The current lead is encouraging but still modest.

The **goal of round 2** is to find a substantially stronger model, ideally:

\[
\boxed{\rho_{\rm full}\ge 0.90}
\]

while maintaining or improving performance on both Liu and Kim.

---

# 2. Critical experimental rule: do not optimize on the 20,509 held-out test set

This rule is absolute.

The official 20,509-row held-out test set has already been used for final round-1 evaluation.

From now on:

- do not inspect test metrics during architecture search;
- do not use the test set for hyperparameter selection;
- do not choose ensembles based on test results;
- do not calibrate on the test set;
- do not tune losses based on test results.

All round-2 model development must use only:

\[
\boxed{\text{the 297,962 training rows and their official CV splits}}
\]

or a newly defined development split internal to those rows.

The 20,509-row test set should be touched only after the round-2 model and ensemble are frozen.

If code currently makes test evaluation easy to call accidentally, modify the pipeline so that test evaluation requires an explicit flag such as:

```bash
--allow-heldout-evaluation
```

and log every held-out evaluation.

---

# 3. Inspect the current repository before modifying anything

Start with:

```bash
git status
git log --oneline -n 20
```

Inspect:

- current PE-RankFormer architecture;
- exact training data;
- official OptiPrime fold assignments;
- configs;
- previous checkpoints;
- evaluation code;
- baseline reproduction code;
- reports from the first pilot.

Create:

```text
reports/round2_initial_inventory.md
```

Summarize:

- current architecture;
- current parameter count;
- current losses;
- context representation;
- simplex head implementation;
- current train/validation/CV protocol;
- current best validation performance;
- known round-1 failure modes;
- current held-out results;
- compute profile.

Do not rewrite stable components unnecessarily.

---

# 4. Preserve the round-1 model as an immutable baseline

The current best PE-RankFormer must remain reproducible.

Tag or record the exact commit.

Create a frozen config:

```text
configs/round2/baseline_round1.yaml
```

and verify it reproduces the expected validation/CV performance.

Do not change the implementation used by this baseline.

All round-2 models should inherit from the baseline code where possible.

---

# 5. Main research philosophy

Do not conduct an unstructured architecture search.

The round-2 strategy is:

\[
\boxed{
\text{one strong baseline}
\rightarrow
\text{high-probability modifications}
\rightarrow
\text{cheap validation screening}
\rightarrow
\text{full CV only for winners}
\rightarrow
\text{heterogeneous ensemble}
}
\]

Focus on modifications most likely to improve performance:

1. stronger context conditioning;
2. context-aware specialization / mixture of experts;
3. feature-augmented learning;
4. improved supervision and losses;
5. context-balanced training;
6. moderate model scaling;
7. training improvements;
8. lightweight self-supervised pretraining;
9. optimized heterogeneous ensemble.

Do not prioritize interpretability.

---

# 6. Use a two-stage search protocol

## Stage A — fast architecture screening

Use one fixed development fold or train/validation arrangement drawn entirely from the 297,962 training rows.

For every candidate architecture:

- same data;
- same split;
- same seed;
- same maximum epoch budget;
- same evaluation code.

Use this phase to rank variants.

Do not run 5-fold CV for every idea.

## Stage B — full 5-fold confirmation

Only the top approximately 3 model families from Stage A should receive:

- all 5 official CV folds;
- multiple seeds if justified;
- ensemble construction.

This preserves compute and makes the search interpretable.

---

# 7. Primary model family A: Layerwise Context PE-RankFormer

This is the highest-priority architectural experiment.

The current model uses context conditioning relatively late in the network.

Modify the model so that experimental context influences sequence representation **throughout the Transformer**.

Let:

\[
c =
E_{\rm cell}
+
E_{\rm PE}
+
E_{\rm Cas9}
+
E_{\rm PAM}
+
E_{\rm scaffold}
+
E_{\rm motif}
+
E_{\rm source}.
\]

At each Transformer block, apply context-conditioned modulation such as:

\[
h'_\ell
=
(1+\gamma_\ell(c))\odot h_\ell+\beta_\ell(c).
\]

Then:

\[
h_{\ell+1}
=
\operatorname{TransformerBlock}_\ell(h'_\ell).
\]

Apply this to:

- edit encoder;
- pegRNA encoder;
- cross-attention blocks.

Possible implementations:

- FiLM;
- adaptive LayerNorm;
- conditional RMSNorm;
- gated residual conditioning.

Start with one clean implementation, preferably **adaptive LayerNorm / FiLM per block**.

Do not add several variants at once.

Internal name:

```text
PE-RankFormer-LC
```

Hypothesis:

> Context-dependent sequence interpretation is especially important for heterogeneous Kim conditions.

Primary readout:

- overall validation Spearman;
- macro context-level Spearman;
- Kim-like condition performance.

---

# 8. Model family B: Context-gated Mixture of Experts

If layerwise context helps, extend it with a small mixture-of-experts component.

Do not create one expert per experimental condition.

Use:

\[
K\in\{4,8\}.
\]

Use the shared sequence representation:

\[
h=F_\theta(x,c)
\]

and a context gate:

\[
g(c,h)=\operatorname{softmax}(W[c;h]).
\]

Then:

\[
h_{\rm out}
=
\sum_{k=1}^{K}g_k f_k(h).
\]

Start with MoE only in:

- the final FFN block; or
- the final two FFN blocks.

Keep the majority of the model shared.

Use load-balancing regularization if expert collapse occurs.

Track:

- expert utilization;
- entropy of gating;
- expert-by-context preferences.

But do not spend time interpreting them biologically.

Internal names:

```text
PE-RankFormer-MoE4
PE-RankFormer-MoE8
```

---

# 9. Model family C: Feature-Augmented Transformer

The goal is now best predictive performance, not sequence purity.

Add a parallel numerical-feature branch containing high-value predictors that are reproducibly available across the exact training and test corpus.

Potential features:

```text
PBS length
RTT length
edit length
edit type
edit position
distance from nick to edit
PBS GC
RTT GC
extension GC
melting-temperature features
RNA MFE features
RuleSet3 / DeepSpCas9 activity
```

Important:

Do not include a feature unless it can be computed consistently for both training and official held-out rows.

Implement:

\[
z_f = \operatorname{MLP}(x_{\rm features})
\]

and concatenate/gate:

\[
z =
[z_{\rm sequence};z_f].
\]

Use normalization based only on training data.

Track missingness.

Do not hand-code mechanistic relationships between features.

The feature branch is simply additional predictive input.

Internal name:

```text
PE-RankFormer-Feat
```

Highest-priority engineered feature:

```text
RuleSet3 / DeepSpCas9
```

because activity of the primary nicking system is known to be an important upstream determinant.

Make sure its computation is identical across sources.

---

# 10. Model family D: Layerwise Context + Feature Branch

If A and C both help independently, combine them.

Internal name:

```text
PE-RankFormer-LC-Feat
```

This is likely to become one of the strongest candidates.

Do not combine all ideas automatically.

Only combine components that individually improve validation performance.

---

# 11. Improve the simplex supervision

Keep the simplex head:

\[
(p_U,p_E,p_I)
=
\operatorname{softmax}(z).
\]

Test whether direct optimization of the quantity of interest improves performance.

Use:

\[
L
=
L_{\rm simplex}
+
\alpha L_{\rm edited}
\]

where:

\[
L_{\rm simplex}
=
-\sum_k y_k\log p_k
\]

and:

\[
L_{\rm edited}
=
\operatorname{Huber}(p_E,y_E).
\]

Test only:

```text
alpha = 0
alpha = 0.25
alpha = 0.5
```

Do not run a large sweep.

---

# 12. Test multitask supervision versus true simplex modeling

Compare under identical encoders:

### Head 1 — scalar
Predict edited fraction only.

### Head 2 — independent multitask
Predict edited and indel independently:

\[
\hat y_E=\sigma(z_E),\qquad
\hat y_I=\sigma(z_I).
\]

### Head 3 — simplex
Predict:

\[
(p_U,p_E,p_I)=\operatorname{softmax}(z).
\]

Use the best-performing head for round-2 optimization.

---

# 13. Replace the old target-local RankNet objective

The previous within-target RankNet loss improved some selection metrics but substantially hurt global Spearman.

Do not use \(\lambda_{\rm rank}=0.25\) in the main model.

Instead test a weak global correlation-aware objective.

Option A:

\[
L_{\rm corr}
=
1-\operatorname{PearsonCorr}(\hat y,y)
\]

over sufficiently large batches.

Option B:

Use a differentiable rank approximation such as:

```text
SoftSort
NeuralSort
torchsort
```

to approximate:

\[
1-\rho_{\rm Spearman}.
\]

Use only small weights:

```text
beta = 0
beta = 0.01
beta = 0.025
beta = 0.05
```

Do not use the held-out set to choose beta.

If rank approximation is unstable or expensive, prioritize the simple correlation loss.

---

# 14. Context-balanced training

Implement context-aware weighting:

\[
w_c\propto N_c^{-\alpha}.
\]

Try:

```text
alpha = 0
alpha = 0.25
alpha = 0.5
```

Apply through weighted loss or context-aware sampling.

Track:

- global Spearman;
- macro context Spearman;
- low-resource context performance.

Do not improve small contexts at the cost of a large drop in overall validation performance.

---

# 15. Two-stage context adaptation

If context-balanced training helps, implement a second-stage adapter approach.

Stage 1: train globally on all 297,962 rows.

Stage 2: freeze most shared weights and train small context-specific adapters for a few epochs.

Possible form:

\[
h'=h+A_cB_ch.
\]

Keep per-context adapter parameters below roughly 1% of the shared model.

Avoid separate full models per context.

---

# 16. Moderate model scaling

Test exactly two larger scales.

## Medium

```text
d_model = 512
heads = 8
edit layers = 8
pegRNA layers = 6
cross-attention blocks = 2
FFN = 2048
```

Target approximately 45–60M parameters.

## Large

```text
d_model = 640
heads = 10
edit layers = 10
pegRNA layers = 6
cross-attention blocks = 3
FFN = 2560
```

Target approximately 75–100M parameters.

Use:

```text
dropout = 0.10 or 0.15
stochastic depth <= 0.05
```

Do not exceed ~100M parameters this round unless scaling clearly helps.

---

# 17. Training improvements

Test low-risk changes:

## EMA

```text
decay = 0.999
```

Evaluate EMA weights on validation.

## Checkpoint averaging

Average the best 3–5 validation checkpoints.

## Longer training

Compare:

```text
30 epochs
50 epochs
```

## Learning rate

For larger models compare:

```text
3e-4
1.5e-4
```

Continue using AdamW unless evidence strongly supports another optimizer.

---

# 18. Self-supervised pretraining

This is secondary to the higher-priority experiments.

Use only PE corpus sequences.

### Edit encoder
Masked paired-token reconstruction.

### pegRNA encoder
Masked nucleotide/span reconstruction over spacer/PBS/RTT.

Pretrain:

```text
5–10 epochs
```

Then fine-tune on supervised efficiency prediction.

Compare against identical architecture initialized from scratch.

---

# 19. Generic genomic foundation models are optional late-stage experiments

Only if custom PE-RankFormer variants plateau, consider:

```text
DNABERT-2
Nucleotide Transformer
Caduceus
```

Do not let foundation-model integration delay the higher-probability experiments.

---

# 20. Build a heterogeneous ensemble

Prefer architectural diversity over many seeds of one model.

Potential members:

```text
baseline simplex
layerwise-context
feature-augmented
layerwise-context + feature
MoE
medium-scale
large-scale
self-supervised model
```

Use CV out-of-fold predictions to choose the ensemble.

Try:

### Mean
\[
\hat y=K^{-1}\sum_k\hat y_k
\]

### Weighted mean
\[
\hat y=\sum_kw_k\hat y_k,
\quad w_k\ge0,\quad\sum_kw_k=1
\]

### Rank average
\[
R=K^{-1}\sum_k\operatorname{rank}(\hat y_k)
\]

Optimize ensemble composition/weights on CV predictions only.

---

# 21. Development metrics

Primary:

```text
validation Spearman
```

Secondary:

```text
Pearson
MAE
RMSE
macro context Spearman
Kim-like-context Spearman
high-efficiency performance
```

Also define:

\[
S_{\rm context}
=
\frac{1}{C}\sum_c \rho_c.
\]

Use it as a secondary generalization metric.

---

# 22. Search schedule

Approximate order:

## Phase 0
Reproduce baseline — 1 run.

## Phase 1: context
- layerwise context
- MoE4
- MoE8

## Phase 2: features
- feature branch
- layerwise context + feature

## Phase 3: loss
- simplex + edited Huber, alpha 0.25
- simplex + edited Huber, alpha 0.5
- weak correlation-aware loss

## Phase 4: balancing
- alpha_context 0.25
- alpha_context 0.5

## Phase 5: scaling
- medium
- large

## Phase 6: training
- EMA
- checkpoint averaging
- 50 epochs

## Phase 7
- self-supervised initialization

Target approximately:

```text
14–18 serious runs
```

---

# 23. Use successive halving

For Stage A:

1. train each new candidate ~10 epochs;
2. stop clearly inferior models;
3. continue promising models to 30 epochs;
4. only the strongest proceed to 50 epochs.

Use consistent criteria.

---

# 24. Compute budget

Current model trains in ~25 minutes for 30 epochs.

Approximate expected costs:

```text
current model: 25–30 min
medium:        35–50 min
large:         50–75 min
```

Measure actual throughput, peak VRAM, and runtime for every run.

The full round should remain feasible on one RTX PRO 6000 Blackwell.

---

# 25. Full 5-fold CV only for finalists

After Stage A, select roughly the top 3 model families using development/CV metrics only.

Then train all five official folds.

Save:

```text
fold checkpoints
OOF predictions
OOF metrics
per-context metrics
```

Use:

```text
results/round2/cv/
```

---

# 26. Multiple seeds only for finalists

For the final one or two architectures, run ~3 seeds if compute permits.

Estimate mean and SD of OOF Spearman.

---

# 27. Freeze the final model before test evaluation

Before touching the 20,509 held-out rows, freeze:

```text
architecture
feature set
loss
context strategy
context weighting
training schedule
model scale
CV checkpoints
ensemble members
ensemble weights
calibration
```

Write:

```text
reports/round2_final_model_spec.md
```

Commit the repository and record the hash.

---

# 28. Final held-out evaluation

Evaluate once on:

```text
full test: 20,509
Liu:       9,175
Kim:      11,334
```

Report:

```text
Pearson
Spearman
MAE
RMSE
per-condition Spearman
number of Kim conditions won
```

Use exactly the corrected OptiPrime baseline protocol.

---

# 29. Statistical comparison

Use paired protospacer-clustered bootstrap.

Use at least:

```text
2000 resamples
```

Prefer 5000 for the final result if cheap.

Report:

```text
observed Δ Spearman
bootstrap mean Δ
95% CI
fraction PE-RankFormer wins
empirical p-value
```

Also bootstrap MAE difference.

---

# 30. Success criteria

## Excellent
\[
\rho_{\rm full}\ge0.90
\]
with significant improvement over OptiPrime.

## Strong
\[
\Delta\rho\ge0.015
\]
with CI excluding zero and gains not confined to only one narrow context.

## Useful
Full-set gain remains significant and Kim performance improves substantially even if Liu remains tied.

## Negative
If no modification beats round-1 PE-RankFormer, preserve the original model and document the failed hypotheses. Do not tune against the test set.

---

# 31. Experiment tracking

For every run save:

```text
run_id
config
git commit
seed
parameter count
training rows
validation rows
context method
feature branch
head type
loss
context balance
epochs
best epoch
val Spearman
val Pearson
val MAE
macro context Spearman
Kim-like Spearman
runtime
peak VRAM
checkpoint
```

Create:

```text
results/round2/model_search.csv
```

---

# 32. Required figures

Using development/CV data only during search, create:

1. validation Spearman by model;
2. macro-context Spearman by model;
3. performance vs parameter count;
4. performance vs runtime;
5. per-context heatmap for finalists;
6. training curves;
7. ensemble member prediction-correlation matrix.

Final held-out plots are produced only after model freeze.

---

# 33. Error analysis for model search

For top development models, stratify errors by:

```text
cell type
PE system
source
edit type
edit length
PBS length
RTT length
efficiency bin
```

Use this only to decide between models on training/CV data.

---

# 34. Do not prematurely simplify

This is a performance-search round.

If a more complex architecture performs better, keep it.

Interpretability, mechanistic explanation, and manuscript simplification come later.

The current objective is:

\[
\boxed{\text{find the highest-performing model}}
\]

---

# 35. Do not conduct an unguided hyperparameter sweep

Avoid massive random/Bayesian search.

Prioritize model ideas.

Use narrow, hypothesis-driven hyperparameter choices.

Once the winning family is known, modest tuning is acceptable.

---

# 36. Protect the held-out test set programmatically

Training code must reject official test data.

Evaluation on heldout should require:

```bash
--allow-heldout-evaluation
```

and create a timestamped log.

Do not bypass this protection.

---

# 37. Maintain a round-2 research log

Create:

```text
reports/round2_research_log.md
```

For every experiment record:

```text
hypothesis
exact change
development result
whether hypothesis supported
decision
next experiment
```

---

# 38. Final round-2 report

Write:

```text
reports/round2_results.md
```

with:

```text
1. Objective
2. Baseline
3. Search protocol
4. Context-conditioning experiments
5. MoE experiments
6. Feature experiments
7. Loss experiments
8. Context-balancing experiments
9. Scaling experiments
10. Training/pretraining experiments
11. Finalist selection
12. Full 5-fold CV
13. Ensemble optimization
14. Frozen final model
15. Official held-out evaluation
16. Comparison with OptiPrime
17. Statistical significance
18. Liu/Kim/per-context results
19. Negative results
20. Compute cost
21. Recommendation for next round
```

---

# 39. Final principle

This round is an **exploratory model search focused on predictive performance**.

The research question is:

\[
\boxed{
\text{How far can a modern data-driven model exceed OptiPrime when both use the identical 297,962 training experiments?}
}
\]

Prioritize:

```text
predictive performance
fair comparison
strict test discipline
clean engineering
reproducibility
```

Do not prioritize mechanistic interpretation yet.

Start now:

1. inspect and freeze the round-1 baseline;
2. lock the 20,509-row held-out test set;
3. reproduce the baseline on development/CV data;
4. implement layerwise context conditioning;
5. run the first controlled screening experiment;
6. continue systematically through the search plan.
