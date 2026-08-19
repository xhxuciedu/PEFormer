# Claude Code Research Prompt — Round 4 PE-RankFormer / Ensemble Optimization

## Role

Act as a senior machine-learning researcher and computational biologist.

You are continuing the PE-RankFormer project for **prime-editing efficiency prediction**. Round 3 established that the strongest performance gain came from **ensemble diversity**, not from a single architecture improvement.

The purpose of Round 4 is to push predictive performance **clearly beyond OptiPrime and beyond the current Round-3 ensemble**, ideally crossing:

\[
\boxed{\rho_{\rm full} \ge 0.90}
\]

on the official held-out Liu+Kim benchmark.

The current focus is **predictive performance**, not biological interpretation.

---

# 0. Important research freedom

The experiments below are a **research plan, not a rigid checklist**.

You are explicitly encouraged to:

- challenge the proposed plan if the evidence suggests a better direction;
- pursue promising independent leads discovered during the experiments;
- introduce new architectures, objectives, preprocessing ideas, ensembling strategies, or training schemes when justified;
- stop low-value branches early;
- revisit earlier assumptions if new evidence contradicts them;
- combine ideas creatively when there is a strong empirical reason.

Do **not** follow the specification mechanically if a better scientific path emerges.

However, preserve these non-negotiable principles:

1. No held-out test leakage.
2. All model selection must use development/OOF data only.
3. Every new direction must be documented with a hypothesis and evidence.
4. Do not claim improvement unless it is supported by fair comparison.
5. Keep code clean and reproducible.

Think like an independent researcher, not a script executor.

---

# 1. Current verified state

Use the existing repository and Round-3 artifacts as the starting point.

Current official held-out benchmark:

\[
N_{\rm heldout}=20,509
\]

consisting of:

- Liu/Hsu: 9,175 rows
- Kim/DeepPrime: 11,334 rows

Current key performance:

| Model | Full Spearman |
|---|---:|
| OptiPrime | 0.8690 |
| Round 1 | 0.8865 |
| Round 2 | 0.8831 |
| **Round 3** | **0.8933** |

Round-3 partition scores:

```text
Liu: 0.8462
Kim: 0.7836
```

Round-3 vs OptiPrime:

\[
\Delta\rho_{\rm full}=+0.0243
\]

with clustered-bootstrap 95% CI approximately:

\[
[+0.0142,+0.0345].
\]

The Round-3 ensemble is:

```text
Family C
+ DAPT
+ Family A
```

combined by **equal-weight rank averaging**.

---

# 2. Central lesson from Round 3

The dominant gain came from **error diversity**.

Matched-dev Spearman:

```text
Round-1 single           0.8798
Family C                 0.8816
DAPT                     0.8820
Family A                 0.8845
Family C + DAPT          0.8919
Family C + DAPT + A      0.8982
```

The key pattern:

```text
Round-1 vs DAPT prediction-rank correlation ~0.997
Family A vs other members ~0.945-0.947
```

Thus:

\[
\boxed{
\text{a slightly weaker but decorrelated model may be more valuable than a slightly stronger redundant model}
}
\]

Round 4 should therefore optimize not only standalone performance, but also **incremental ensemble value**.

---

# 3. Non-negotiable held-out discipline

Do not use the official 20,509-row held-out set during model search.

Do not:

- inspect held-out metrics;
- fit ensemble weights on held-out;
- fit calibrators on held-out;
- choose architectures based on held-out;
- select checkpoints based on held-out;
- tune source-specific models using held-out labels.

All model development must use:

```text
matched Liu+Kim development folds
and/or
official OOF predictions
```

Require explicit access flag:

```bash
--allow-heldout-evaluation
```

and log every held-out evaluation.

---

# 4. Add a Round-4 internal lockbox

Because the official held-out benchmark has already been examined after Rounds 1–3, create an additional internal **Round-4 lockbox** from the 297,962 training rows.

Requirements:

- Liu+Kim only;
- protospacer-disjoint;
- representative Liu:Kim ratio;
- never used for Stage-A model development;
- evaluated only after shortlisting final candidates.

Suggested structure:

```text
train/dev pool
round4_internal_lockbox
official heldout
```

Use matched dev folds for exploration.

Use the Round-4 internal lockbox once for pre-final confirmation.

Only then freeze the final system and evaluate the official held-out benchmark.

Document this hierarchy clearly.

---

# 5. Round-4 search objective

For every candidate model M, measure:

## Standalone quality

\[
S_1(M)=\rho(y,\hat y_M)
\]

## Prediction diversity

\[
S_2(M)=\rho_{\rm rank}(\hat y_M,\hat y_{\rm current\ ensemble})
\]

## Residual-error correlation

Compute residuals on OOF/dev data:

\[
r_M = y-\hat y_M
\]

and compare:

\[
\rho(r_M,r_E)
\]

where E is the current ensemble.

## Incremental ensemble gain

Most important:

\[
S_3(M)
=
\rho(\operatorname{Ensemble}(E,M),y)
-
\rho(E,y).
\]

This should become the main search criterion.

A candidate should not advance merely because its standalone Spearman is slightly higher.

---

# 6. Promotion rule for new candidates

Prefer candidates with:

```text
standalone performance reasonably competitive
+
meaningful decorrelation
+
positive ensemble gain on all or most matched dev folds
```

Suggested promotion threshold:

```text
mean ensemble gain >= +0.003
```

or a clearly positive clustered-bootstrap signal.

Do not spend full 5-fold compute on candidates that are:

```text
prediction correlation > 0.99 with current ensemble
and
ensemble gain < 0.001
```

unless there is another strong reason.

---

# 7. Highest-priority experiment A — Medium-scale layerwise-context model

Train a larger but still practical PE-RankFormer.

Suggested architecture:

```text
d_model = 512
heads = 8
edit layers = 8
pegRNA layers = 6
cross-attention blocks = 2
FFN = 2048
```

Target:

```text
45-60M parameters
```

Use:

```text
layerwise context conditioning
adaptive LayerNorm or FiLM
simplex head
no ranking loss
no correlation loss
```

Train under the same matched Liu+Kim development protocol.

Measure:

```text
standalone Spearman
correlation with Family A
correlation with Family C
correlation with DAPT
residual correlation
incremental ensemble gain
```

Important:

If this model is only slightly stronger but nearly identical to the current ensemble, do not promote it.

Internal name:

```text
R4-Medium-AdaLN
```

---

# 8. Highest-priority experiment B — A genuinely different sequence architecture

Train a model whose inductive bias differs materially from the Transformer backbone.

Strong candidates include:

```text
bidirectional state-space model
Mamba-style encoder
Caduceus-style bidirectional DNA encoder
SSM + small cross-attention hybrid
convolutional/SSM hybrid
```

Suggested design:

### Edit stream
Use a bidirectional SSM over paired WT/edited tokens.

### pegRNA stream
Use a separate bidirectional SSM over spacer/PBS/RTT.

### Interaction
Use either:

```text
small cross-attention module
cross-gating
bilinear interaction
```

Do not simply recreate the current Transformer with different names.

Keep:

```text
context conditioning
simplex supervision
same training labels
same matched validation
```

Internal name:

```text
PE-SSM
```

The main goal is **decorrelated errors** while maintaining competitive accuracy.

---

# 9. Highest-priority experiment C — Context-gated ensemble

The current ensemble uses equal global weights.

Do not repeat unrestricted global fitted weights; Round 3 showed they were unstable and inferior.

Instead test **low-capacity context-dependent gating**.

Input to the gate:

```text
source
cell type
PE system
Cas9/PAM type
other known experimental context
```

Do not initially feed raw sequence.

Predict nonnegative ensemble weights:

\[
w_k(c)\ge0,
\qquad
\sum_k w_k(c)=1.
\]

Regularize toward equal weights:

\[
L_{\rm gate}
=
L_{\rm prediction}
+
\lambda
\sum_k
(w_k-1/K)^2.
\]

Use OOF predictions only.

Try:

```text
linear softmax gate
small 1-hidden-layer MLP gate
```

Do not use a large network.

Compare against:

```text
equal rank average
equal score average
source-specific fixed weights
context-gated weights
```

Use nested dev evaluation.

---

# 10. Experiment D — Nonlinear stacking

Use OOF predictions from current and new models as meta-features.

Start simple.

Inputs:

```text
rank predictions
raw predictions
source
cell type
PE system
```

Try:

```text
ridge regression
elastic net
small MLP
gradient-boosted trees
```

Use nested CV so the stacker never sees the fold it predicts.

Do not use official held-out rows.

Primary metric:

```text
matched Liu+Kim Spearman
```

The stacker must outperform simple rank averaging consistently before promotion.

---

# 11. Experiment E — Residual-learning model

Train a model specifically on what the current ensemble misses.

First construct OOF predictions of current ensemble E.

Define residual:

\[
r_i = y_i-\hat y_{E,i}.
\]

Train a residual predictor:

\[
\hat r_i = g(x_i,c_i).
\]

Possible inputs:

```text
sequence representation
context
design metadata
engineered features
current ensemble score
```

Final prediction:

\[
\hat y_i
=
\hat y_{E,i}
+
\eta\hat r_i.
\]

Tune eta on dev data only.

Possible residual models:

```text
small Transformer head
MLP on frozen embeddings
LightGBM/XGBoost on engineered features
small SSM
```

This experiment explicitly targets complementary information.

---

# 12. Experiment F — PE-specific relational pretraining

Do not start with generic genomic foundation models.

Pretrain using task-relevant relational objectives.

## Objective 1 — masked paired-token reconstruction
Mask paired WT/edit tokens and predict original pair state.

## Objective 2 — masked pegRNA reconstruction
Mask spans in spacer/PBS/RTT.

## Objective 3 — target–pegRNA contrastive learning

Given a target/edit representation, identify the matching pegRNA among hard negatives.

Use InfoNCE:

\[
L_{\rm contrast}
=
-\log
\frac{
\exp(\operatorname{sim}(z_t,z_p^+)/\tau)
}{
\sum_j
\exp(\operatorname{sim}(z_t,z_{p,j})/\tau)
}.
\]

Use hard negatives, not only random negatives.

Potential hard negatives:

```text
same target with different pegRNA
similar spacer sequence
same edit type
similar PBS/RTT lengths
same context
```

Pretrain for a modest number of epochs.

Then supervised train and evaluate under matched dev folds.

The goal is both better standalone accuracy and different learned representation.

---

# 13. Experiment G — Source-specialized models

Train global model first.

Then fine-tune separately:

```text
Liu specialist
Kim specialist
```

Use matched OOF source-specific evaluation.

At inference:

```text
Liu rows -> Liu specialist
Kim rows -> Kim specialist
```

Test whether specialization improves:

```text
rho_Liu
rho_Kim
full Liu+Kim rho after source calibration/rank combination
```

Kim is particularly important because it remains the larger opportunity.

Target:

\[
\rho_{\rm Kim}\ge0.80.
\]

Do not sacrifice Liu materially.

---

# 14. Experiment H — Source/context-specific adapters

If full specialists are redundant or unstable, use adapters.

Shared backbone:

\[
h=F(x)
\]

with:

\[
h'=h+A_cB_ch.
\]

Try:

```text
source-specific adapters
cell-specific adapters
PE-system adapters
```

Use small ranks:

```text
8
16
32
```

Only proceed if adapters improve ensemble complementarity or domain performance.

---

# 15. Experiment I — Moderate source balancing / replay

Round 3 showed DAPT helps only slightly and is highly redundant.

Therefore do not spend many runs here.

Only test source-balancing if a new architecture shows domain imbalance.

Possible mixtures:

```text
Liu 40 / Kim 40 / Schwank 20
Liu 45 / Kim 45 / Schwank 10
```

or late-stage replay:

```text
90% Liu+Kim / 10% Schwank
75% Liu+Kim / 25% Schwank
```

Treat these as refinements, not primary Round-4 directions.

---

# 16. Do not revisit clear dead ends without new evidence

Avoid re-running:

```text
old RankNet loss
batch correlation loss
original MoE
global unrestricted fitted ensemble weights
adding round-1 baseline to the current 3-way ensemble
```

unless a new experiment gives a specific reason to reopen them.

---

# 17. Ensemble-diversity diagnostics

For every serious candidate, save:

```text
prediction correlation matrix
rank correlation matrix
residual correlation matrix
error overlap
per-context performance
ensemble gain
```

Create:

```text
results/round4/diversity/
```

Generate a table:

| Candidate | Standalone rho | Corr w/ ensemble | Residual corr | Ensemble delta rho |
|---|---:|---:|---:|---:|

This table should drive promotion decisions.

---

# 18. Diversity-first candidate screening

Use a two-dimensional decision rule.

A candidate can advance if either:

### Path A — strong standalone

\[
\Delta\rho_{\rm standalone}\ge0.005
\]

or

### Path B — strong complementarity

Standalone may be similar, but:

```text
prediction correlation <= ~0.97
and
ensemble delta rho >= ~0.003
```

Use clustered bootstrap where useful.

---

# 19. Use successive halving

Do not fully train every idea.

Suggested:

```text
5-10 epoch screen
30 epoch confirmation
full 5-fold only for finalists
```

For new architectures, first train on one matched dev fold.

If standalone poor and ensemble gain negligible, stop early.

---

# 20. Round-4 lockbox gate

Before full 5-fold training, shortlist approximately 3 candidates.

Evaluate them once on the Round-4 internal lockbox.

Promotion criteria:

```text
positive lockbox ensemble gain
no major Liu collapse
no major Kim collapse
```

Only candidates that survive the lockbox should receive full 5-fold training.

---

# 21. Full 5-fold training

For promoted candidates:

- train all 5 official folds;
- save checkpoints;
- create OOF predictions;
- compute standalone metrics;
- compute diversity metrics;
- compute ensemble increments.

Do not touch official held-out.

---

# 22. Final ensemble search

Start from the frozen Round-3 ensemble:

```text
Family C
DAPT
Family A
```

Then consider adding:

```text
medium AdaLN
PE-SSM
pretrained relational model
source specialist
residual model
```

Use OOF and lockbox results.

Do not assume more members are always better.

Use greedy forward selection:

1. current ensemble;
2. add candidate with best positive incremental gain;
3. use only simple combination rules;
4. stop when no candidate improves robustly.

Try:

```text
equal rank average
context-gated rank combination
simple stacker
```

Avoid unstable weight fitting.

---

# 23. Final model freeze

Before official held-out evaluation, write:

```text
reports/round4_final_model_spec.md
```

Include:

```text
ensemble members
member architectures
training data
CV folds
pretraining
adaptation
combination rule
gate/stacker details
random seeds
checkpoint paths
git commit
```

Commit repository.

Do not change anything afterward.

---

# 24. Final official held-out evaluation

Evaluate once on:

```text
Full: 20,509
Liu:   9,175
Kim:  11,334
```

Primary:

```text
Spearman
```

Also report:

```text
Pearson if meaningful
MAE if calibrated
RMSE if calibrated
per-context Spearman
Kim conditions won
```

If final output is rank-based and not calibrated, say so clearly.

---

# 25. Statistical comparison

Use paired protospacer-clustered bootstrap.

Use:

```text
5000 resamples
```

Report:

\[
\Delta\rho_{\rm R4-OptiPrime}
\]

and:

\[
\Delta\rho_{\rm R4-R3}.
\]

Report:

```text
observed difference
bootstrap mean
95% CI
fraction wins
empirical p-value
```

---

# 26. Round-4 success criteria

## Primary target

\[
\boxed{\rho_{\rm full}\ge0.90}
\]

## Secondary targets

\[
\rho_{\rm Kim}\ge0.80
\]

and:

\[
\rho_{\rm Liu}\ge0.845.
\]

## Strong benchmark lead

Aim for:

\[
\Delta\rho_{\rm OptiPrime}\ge0.03.
\]

## Strong ensemble discovery

A new member producing:

\[
\Delta\rho_{\rm ensemble}\ge0.005
\]

on matched OOF/dev data is highly valuable even if its standalone score is not the best.

---

# 27. Recommended experiment order

Suggested priority:

## Phase 0
1. Build Round-4 internal lockbox.
2. Reproduce Round-3 ensemble OOF metrics and diversity matrix.

## Phase 1
3. Medium-scale AdaLN model.
4. PE-SSM / SSM-hybrid model.
5. Context-gated ensemble on existing members.

## Phase 2
6. Nonlinear stacking.
7. Residual learner.

## Phase 3
8. PE-specific relational pretraining.
9. Liu specialist.
10. Kim specialist.

## Phase 4
11. Adapters if specialists help.
12. Source balancing/replay if needed.

## Phase 5
13. Internal lockbox screening.
14. Full 5-fold training of finalists.
15. Greedy ensemble construction.
16. Freeze.
17. Official held-out evaluation.

This order is advisory, not mandatory.

If a new high-value lead emerges, pursue it.

---

# 28. Experiment tracking

Create:

```text
results/round4/model_search.csv
```

Columns:

```text
run_id
architecture
params
seed
dev_fold
standalone_spearman
liu_spearman
kim_spearman
macro_context_spearman
corr_with_r3_ensemble
residual_corr_with_r3
ensemble_delta_spearman
runtime_min
peak_vram_gb
checkpoint
decision
notes
```

---

# 29. Research log

Maintain:

```text
reports/round4_research_log.md
```

For each experiment:

```text
hypothesis
why this model may be complementary
implementation
standalone result
diversity result
ensemble result
decision
independent observations
next step
```

Also document any **independent lead** pursued outside the original specification.

---

# 30. Required plots

Create:

```text
results/round4/figures/
```

At minimum:

1. standalone Spearman vs ensemble gain;
2. prediction-correlation heatmap;
3. residual-correlation heatmap;
4. per-context performance heatmap;
5. Liu vs Kim performance scatter;
6. ensemble performance as members are added;
7. diversity vs ensemble gain;
8. scaling curve if applicable;
9. stacking/gating comparison.

---

# 31. Final Round-4 report

Write:

```text
reports/round4_results.md
```

Suggested structure:

```text
1. Objective
2. Round-3 baseline
3. Round-4 lockbox design
4. Ensemble-diversity analysis
5. Medium-scale model
6. Alternative architecture / SSM
7. Context-gated ensemble
8. Stacking
9. Residual learning
10. Relational pretraining
11. Source specialists
12. Independent exploratory leads
13. Finalists
14. Lockbox results
15. Full 5-fold results
16. Final ensemble construction
17. Frozen final system
18. Official held-out evaluation
19. Comparison with OptiPrime
20. Statistical significance
21. Negative results
22. Compute cost
23. Recommendation for Round 5
```

---

# 32. Final principle

The core Round-4 question is no longer:

> Which single architecture is best?

It is:

\[
\boxed{
\text{Which additional predictor contributes the most new information beyond the current ensemble?}
}
\]

Search for:

```text
strong predictors
+
decorrelated errors
+
robust ensemble gains
```

rather than only marginal standalone improvements.

Be systematic, but be creative.

If the experiments reveal a stronger direction than the one specified here, pursue it.

Do not optimize for elegance yet.

Optimize for **predictive performance, complementarity, and fair evaluation**.
