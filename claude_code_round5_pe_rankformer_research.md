# Claude Code Research Prompt — Round 5 PE-RankFormer / Ordinal-SSM Research

## Role

Act as a senior machine-learning researcher and computational biologist.

You are continuing a multi-round research program on **prime-editing efficiency prediction**.

Round 4 established a strong new result:

\[
\rho_{\rm heldout}=0.9079
\]

on the official 20,509-row Liu+Kim held-out benchmark, compared with:

\[
\rho_{\rm OptiPrime}=0.8690.
\]

The improvement is:

\[
\Delta \rho = +0.0389,
\]

with a paired protospacer-clustered bootstrap 95% CI approximately:

\[
[+0.0286,+0.0498].
\]

Round 4 also showed that the strongest new single model was an **ordinal-head + bidirectional SSM model**, and that the main successful mechanism was not simply more capacity or more ensembling, but the combination of:

\[
\boxed{
\text{strong standalone accuracy}
+
\text{different supervision geometry}
+
\text{different sequence representation}
+
\text{complementary errors}
}
\]

The goal of Round 5 is to push performance further, but in a more focused and hypothesis-driven way than Round 4.

The current emphasis remains:

\[
\boxed{\text{predictive performance first}}
\]

Interpretability and mechanistic explanation remain secondary.

---

# 0. Research freedom — important

This specification is a **research roadmap, not a rigid checklist**.

You are explicitly authorized to:

- pursue independent leads;
- abandon planned experiments when evidence makes them low-value;
- introduce new architectures, objectives, regularizers, augmentation schemes, calibration methods, or ensemble strategies;
- revisit earlier negative results if a genuinely new mechanism changes the context;
- combine ideas creatively;
- run small diagnostic experiments not listed here;
- reformulate the problem if the data strongly suggest a better objective.

Do not mechanically execute all items.

Think like an independent research scientist.

However, preserve these non-negotiable rules:

1. **No held-out leakage.**
2. **No model selection using the official 20,509-row held-out labels.**
3. **All development choices must be justified by OOF/dev/lockbox evidence.**
4. **Every independent lead must be documented.**
5. **Do not claim improvement without a fair comparison.**
6. **Keep all code and results reproducible.**

---

# 1. Current verified state

Use the existing repository and Round-4 artifacts as the starting point.

## Official held-out benchmark

```text
Full: 20,509
Liu:   9,175
Kim:  11,334
```

## Current held-out performance

| Model | Full | Liu | Kim |
|---|---:|---:|---:|
| OptiPrime | 0.8690 | 0.8365 | 0.7320 |
| Round 3 | 0.8933 | 0.8462 | 0.7836 |
| **Round 4** | **0.9079** | **0.8585** | **0.8124** |

Round 4 improves over Round 3 by:

\[
+0.0146
\]

and over OptiPrime by:

\[
+0.0389.
\]

Round-4 vs OptiPrime bootstrap:

```text
95% CI ≈ [+0.0286, +0.0498]
wins = 100% of 5000 resamples
p < 0.0002
```

Important caveat:

The point estimate exceeds 0.90, but the bootstrap CI for the absolute \(\rho\) includes values slightly below 0.90.

Therefore do not frame the Round-5 objective as merely “break 0.90.”

The more important target is to improve robustly over the frozen Round-4 system.

---

# 2. Central lessons from Round 4

Round 4 produced several strong empirical conclusions.

## 2.1 Ordinal supervision was a breakthrough

The CORAL-style ordinal head predicts cumulative events:

\[
P(y>t_k)
\]

at thresholds \(t_k\) derived from target quantiles.

The output score is approximately:

\[
s(x)
=
\frac{1}{K-1}
\sum_{k=1}^{K-1}
P(y>t_k \mid x).
\]

This aligns naturally with Spearman correlation.

Ordinal-head models were substantially more decorrelated from the existing simplex models.

## 2.2 SSM architecture was also independently useful

The bidirectional state-space sequence mixer produced a strong new representation.

The strongest single development model was:

```text
ordinal + SSM
```

with OOF development Spearman:

\[
0.9082,
\]

higher than the entire Round-3 ensemble:

\[
0.8982.
\]

## 2.3 Diversity alone is not enough

A weak pairwise-ranking model was highly decorrelated but hurt the ensemble.

Therefore:

\[
\boxed{
\text{new members must be both competent and complementary}
}
\]

## 2.4 Scaling, seeds, bagging, and post-hoc combination gave little

Round 4 found:

```text
medium AdaLN scaling: small gain only
new random seed: small gain only
bagging: negligible or negative
context-gated ensemble: null
nonlinear stacking: tiny
engineered-feature residual learner: essentially exhausted
```

Therefore Round 5 should focus on **new learning mechanisms**, not brute-force capacity.

---

# 3. New baseline for Round 5

Treat **Ordinal-SSM** as the primary new backbone.

Do not default back to the original PE-RankFormer Transformer unless needed as an ensemble diversity baseline.

The Round-5 search should use:

```text
Ordinal-SSM
```

as the main reference model.

Also preserve all Round-4 ensemble members and predictions for comparison.

Create:

```text
configs/round5/baseline_ordinal_ssm.yaml
reports/round5_initial_inventory.md
```

Record:

```text
architecture
parameter count
training objective
ordinal thresholds
training folds
OOF metrics
correlation to current ensemble
held-out metrics if already known from Round 4
checkpoint paths
git commit
```

---

# 4. Test discipline for Round 5

The official held-out set has already been used in multiple prior rounds.

Therefore Round 5 must be stricter.

Use:

```text
matched Liu+Kim dev folds
+
Round-4 internal lockbox
+
new Round-5 internal lockbox if practical
```

Do not touch official held-out during search.

Recommended hierarchy:

```text
Stage A:
matched dev folds

Stage B:
Round-5 internal lockbox

Stage C:
full official 5-fold OOF

Stage D:
freeze

Stage E:
single official held-out evaluation
```

If an independent external dataset is available and can be processed fairly, treat it as an additional confirmatory benchmark.

Do not use it for iterative hyperparameter tuning.

---

# 5. Main Round-5 search objective

For every candidate \(M\), record:

## Standalone accuracy

\[
S_1(M)
=
\rho(y,\hat y_M)
\]

## Correlation with Ordinal-SSM

\[
S_2(M)
=
\rho_{\rm rank}
(
\hat y_M,
\hat y_{\rm OrdSSM}
)
\]

## Correlation with Round-4 ensemble

\[
S_3(M)
=
\rho_{\rm rank}
(
\hat y_M,
\hat y_{\rm R4}
)
\]

## Residual correlation

\[
S_4(M)
=
\rho(
y-\hat y_M,
y-\hat y_{\rm R4}
)
\]

## Ensemble gain

\[
S_5(M)
=
\rho(
\operatorname{Ensemble}(\hat y_{\rm R4},\hat y_M),
y
)
-
\rho(
\hat y_{\rm R4},y
).
\]

Primary promotion criterion:

```text
robust positive ensemble gain
```

Secondary:

```text
strong standalone accuracy
```

Do not promote candidates based only on novelty or decorrelation.

---

# 6. Highest-priority experiment A — Dual-head Ordinal + Simplex

Use a shared Ordinal-SSM backbone.

Add:

## Head 1 — ordinal

Predict:

\[
P(y>t_k)
\]

for \(k=1,\ldots,K-1\).

## Head 2 — simplex

Predict:

\[
(p_U,p_E,p_I)
=
\operatorname{softmax}(z).
\]

Train:

\[
L
=
L_{\rm ordinal}
+
\lambda_{\rm simplex}
L_{\rm simplex}.
\]

Use the ordinal head as the primary ranking score.

Try only:

```text
lambda_simplex = 0.10
lambda_simplex = 0.25
lambda_simplex = 0.50
```

also keep:

```text
lambda_simplex = 0
```

as the baseline.

Hypothesis:

> Ordinal supervision optimizes ordering, while simplex supervision provides additional outcome information and regularization.

Promotion requires improvement on matched dev or meaningful ensemble complementarity.

Internal name:

```text
R5-OrdSSM-DualHead
```

---

# 7. Experiment B — Multi-resolution ordinal learning

Instead of choosing one threshold resolution, predict multiple resolutions jointly.

Example heads:

```text
K = 7
K = 18
K = 43
```

Train:

\[
L
=
w_7L_7
+
w_{18}L_{18}
+
w_{43}L_{43}.
\]

Start with equal weights.

At inference compare:

```text
average percentile score
learned monotone combination
single best-resolution head
```

Use OOF only for selecting the combination.

Hypothesis:

> Coarse thresholds stabilize large efficiency differences while fine thresholds preserve local ranking resolution.

Internal name:

```text
R5-OrdSSM-MultiRes
```

---

# 8. Experiment C — Global + within-context ordinal multitask learning

Use two ordinal tasks.

## Global ordinal target

Convert \(y\) to global training-distribution quantiles:

\[
q_{\rm global}(y).
\]

## Context-normalized ordinal target

For context \(c\):

\[
q_c(y)
=
F_c(y).
\]

Context may include:

```text
source
cell line
PE system
editor condition
```

Train:

\[
L
=
L_{\rm global}
+
\lambda_cL_{\rm context}.
\]

Use only the global head for the main final score.

Try:

```text
lambda_c = 0.10
lambda_c = 0.25
lambda_c = 0.50
```

Hypothesis:

> The auxiliary context-relative ranking task may improve representation learning in heterogeneous Kim conditions without sacrificing global ranking.

Track especially:

```text
Kim Spearman
macro-context Spearman
Liu Spearman
full Spearman
```

Internal name:

```text
R5-OrdSSM-ContextOrdinal
```

---

# 9. Experiment D — Reverse-complement augmentation / consistency

Test whether orientation invariance can improve generalization.

Important:

Do not naively reverse sequences.

Implement a biologically and geometrically consistent transformation for:

```text
WT sequence
edited sequence
spacer
PBS
RTT
PAM orientation
edit coordinates
nick-relative positions
all orientation-sensitive features
```

First write unit tests confirming that a transformed example represents the same underlying edit in reverse-complement coordinates.

Then test:

## Augmentation

Train on:

```text
original
+
reverse-complement transformed
```

## Consistency regularization

Require:

\[
f(x)\approx f(RC(x)).
\]

For ordinal outputs:

\[
L_{\rm RC}
=
D(
p_{\rm ord}(x),
p_{\rm ord}(RC(x))
).
\]

Possible \(D\):

```text
MSE on cumulative probabilities
KL divergence
logit-space L2
```

Do not force exact invariance if there is evidence that orientation conventions encode legitimate information.

Run a small diagnostic first.

Internal name:

```text
R5-OrdSSM-RC
```

---

# 10. Experiment E — Relational contrastive pretraining

Use Ordinal-SSM as the downstream architecture.

Pretrain with three objectives.

## 10.1 Paired WT/edit masked modeling

Mask paired WT/edit tokens.

Predict original token pair.

## 10.2 pegRNA masked span prediction

Mask spans in:

```text
spacer
PBS
RTT
```

Predict the original sequence.

## 10.3 Target–pegRNA contrastive learning

Represent:

```text
target/edit
pegRNA
```

separately.

Positive pair:

```text
correct target/edit + its pegRNA
```

Hard negatives should be challenging.

Preferred negative hierarchy:

1. same target, different pegRNA;
2. same edit type, similar edit position;
3. similar spacer;
4. similar PBS/RTT lengths;
5. same experimental context.

Use InfoNCE:

\[
L_{\rm InfoNCE}
=
-\log
\frac{
\exp(\operatorname{sim}(z_t,z_p^+)/\tau)
}{
\sum_j
\exp(\operatorname{sim}(z_t,z_{p,j})/\tau)
}.
\]

Pretraining loss:

\[
L_{\rm pre}
=
L_{\rm edit-mask}
+
L_{\rm peg-mask}
+
\lambda_{\rm con}
L_{\rm InfoNCE}.
\]

Try only a small number of \(\lambda_{\rm con}\) values.

Then:

```text
pretrain
-> supervised Ordinal-SSM
-> optional Liu+Kim adaptation
```

Compare against identical architecture trained from scratch.

Internal name:

```text
R5-OrdSSM-RelPretrain
```

---

# 11. Experiment F — Objective × architecture factorial study

Construct a clean 2×2 matrix.

| Architecture | Simplex | Ordinal |
|---|---|---|
| Transformer | existing | ordinal Transformer |
| SSM | simplex SSM | ordinal SSM |

For each model compute:

```text
standalone OOF Spearman
Liu Spearman
Kim Spearman
prediction correlation
residual correlation
ensemble marginal gain
```

Then quantify:

```text
architecture effect
objective effect
interaction effect
```

Do not overinterpret causally, but use the matrix to guide model selection.

---

# 12. Experiment G — Hybrid Transformer + SSM

If the factorial study suggests architecture complementarity, try a genuine hybrid.

Possible designs:

## Alternating blocks

```text
SSM
Transformer
SSM
Transformer
```

## Parallel mixer

\[
h_{\rm SSM}=F_{\rm SSM}(x)
\]

\[
h_{\rm Attn}=F_{\rm Attn}(x)
\]

then:

\[
h
=
g\odot h_{\rm SSM}
+
(1-g)\odot h_{\rm Attn}.
\]

## SSM local/global decomposition

Use:

```text
SSM for sequence mixing
attention only for target-pegRNA interaction
```

Do not create a broad architecture sweep.

Try one or two principled designs.

Internal name:

```text
R5-HybridMixer
```

---

# 13. Experiment H — Quantile distribution head

The ordinal breakthrough suggests that distributional prediction may be valuable.

Test direct quantile regression.

Predict:

```text
q10
q25
q50
q75
q90
```

using pinball loss.

For ranking, use:

```text
predicted median
or
mean of predicted quantiles
```

Potentially combine with ordinal supervision.

Hypothesis:

> Modeling the conditional outcome distribution may produce a more robust ordering under heteroscedastic assay noise.

This is exploratory and should be screened cheaply.

---

# 14. Experiment I — Weak pairwise consistency, not old RankNet

Do not reuse the old high-weight ranking loss.

Optionally test a very weak consistency regularizer only on pairs with clear observed separation:

\[
|y_i-y_j|>\delta.
\]

Encourage:

\[
\operatorname{sign}(s_i-s_j)
=
\operatorname{sign}(y_i-y_j).
\]

Use:

```text
lambda_pair <= 0.01
```

Treat this as optional.

Stop immediately if global Spearman declines.

---

# 15. Calibration — restore absolute efficiency prediction

The Round-4 rank-averaged ensemble is excellent for Spearman but is not a calibrated editing-efficiency estimate.

Round 5 should restore a meaningful absolute prediction without sacrificing rank quality.

Use OOF data only.

Fit a monotone map:

\[
g(s)
\rightarrow
\hat y.
\]

Candidate calibrators:

```text
isotonic regression
monotone cubic spline
monotone neural spline
```

A monotone transformation should preserve Spearman up to ties/numerical effects.

Then report:

```text
Spearman
Pearson
MAE
RMSE
calibration curve
```

---

# 16. Ensemble design for Round 5

Preserve the frozen Round-4 ensemble as the baseline.

Candidate additions may include:

```text
DualHead Ordinal-SSM
MultiRes Ordinal-SSM
ContextOrdinal Ordinal-SSM
RC-augmented Ordinal-SSM
RelPretrain Ordinal-SSM
HybridMixer
quantile model
```

Do not assume all strong models should be included.

Use greedy forward selection based on OOF ensemble gain.

For every candidate require:

```text
competent standalone score
+
positive marginal contribution
```

Prefer equal rank averaging unless another combination method gives a robust OOF gain.

Do not reopen unrestricted global weight fitting unless evidence justifies it.

---

# 17. Search protocol

Use successive halving.

## Stage A — one matched dev fold

Train each candidate for:

```text
5-10 epochs
```

or equivalent budget.

Measure:

```text
standalone rho
corr with Ordinal-SSM
corr with R4 ensemble
residual corr
ensemble delta rho
```

Stop clearly weak candidates.

## Stage B — three matched dev folds

Promising candidates must improve on at least 2 of 3 folds.

## Stage C — internal lockbox

Evaluate shortlisted candidates once.

Do not use lockbox to optimize fine details.

## Stage D — full 5-fold training

Only finalists receive full official-fold training.

## Stage E — freeze

Freeze architecture and ensemble.

## Stage F — official held-out

Single evaluation only.

---

# 18. Candidate promotion thresholds

A candidate can advance via either path.

## Path A — strong standalone improvement

\[
\Delta\rho_{\rm standalone}\ge0.005
\]

relative to Ordinal-SSM.

## Path B — strong ensemble contribution

\[
\Delta\rho_{\rm ensemble}\ge0.003
\]

with competitive standalone accuracy.

Do not promote weak standalone models even if highly decorrelated.

Round 4 already showed that this fails.

---

# 19. Independent external validation

If a suitable external PE dataset is available and can be processed fairly:

- identify it;
- document provenance;
- verify no overlap with training;
- freeze the candidate before evaluation;
- run PE-RankFormer and OptiPrime under matched preprocessing;
- report results separately from the official benchmark.

Do not tune on the external test set.

If no suitable independent dataset is available, document that explicitly.

---

# 20. Negative-result discipline

Do not hide failed ideas.

For every serious experiment, record:

```text
hypothesis
result
effect size
correlation/diversity behavior
why it was stopped
```

Distinguish:

```text
weak standalone
redundant
unstable
non-transferable
implementation-invalid
```

---

# 21. Experiment tracking

Create:

```text
results/round5/model_search.csv
```

Columns:

```text
run_id
architecture
objective
params
seed
dev_fold
standalone_spearman
liu_spearman
kim_spearman
macro_context_spearman
corr_with_ordssm
corr_with_r4_ensemble
residual_corr
ensemble_delta_spearman
pearson_if_calibrated
mae_if_calibrated
rmse_if_calibrated
runtime_min
peak_vram_gb
checkpoint
decision
notes
```

---

# 22. Research log

Maintain:

```text
reports/round5_research_log.md
```

For every experiment record:

```text
hypothesis
why it may improve accuracy
why it may improve complementarity
exact implementation
training protocol
dev result
lockbox result if used
decision
independent observations
next step
```

Also document every independent lead pursued outside this specification.

---

# 23. Required plots

Create:

```text
results/round5/figures/
```

At minimum:

1. standalone Spearman vs ensemble gain;
2. objective × architecture factorial plot;
3. prediction-correlation heatmap;
4. residual-correlation heatmap;
5. Liu vs Kim performance scatter;
6. ordinal threshold-resolution comparison;
7. calibration curve;
8. ensemble performance as members are added;
9. performance vs compute;
10. external-validation comparison if available.

---

# 24. Round-5 report

Write:

```text
reports/round5_results.md
```

Suggested structure:

```text
1. Objective
2. Round-4 baseline
3. Round-5 evaluation discipline
4. Ordinal-SSM baseline
5. Dual-head ordinal + simplex
6. Multi-resolution ordinal learning
7. Global + context ordinal multitask
8. Reverse-complement augmentation/consistency
9. Relational contrastive pretraining
10. Objective × architecture factorial study
11. Hybrid sequence mixer
12. Quantile/distributional experiments
13. Independent exploratory leads
14. Calibration
15. Ensemble search
16. Internal lockbox
17. Full 5-fold results
18. Frozen final system
19. Official held-out evaluation
20. Comparison with OptiPrime
21. Statistical significance
22. External validation
23. Negative results
24. Compute cost
25. Recommendation for Round 6 / manuscript phase
```

---

# 25. Statistical testing

For final comparisons use paired protospacer-clustered bootstrap.

Use:

```text
5000 resamples
```

Report:

\[
\Delta\rho_{\rm R5-R4}
\]

and:

\[
\Delta\rho_{\rm R5-OptiPrime}.
\]

Report:

```text
observed delta
bootstrap mean
95% CI
fraction wins
empirical p-value
```

If final output is calibrated, also bootstrap:

```text
MAE difference
RMSE difference
```

---

# 26. Round-5 success criteria

## Major success

\[
\rho_{\rm full}\ge0.915
\]

or:

\[
\Delta\rho_{\rm R5-R4}\ge0.005
\]

with CI excluding zero.

## Strong Kim improvement

\[
\rho_{\rm Kim}\ge0.82
\]

without meaningful Liu degradation.

## Strong single-model result

A single model that clearly exceeds the Round-4 ensemble OOF score is especially valuable.

## Strong methodological result

A new supervision strategy or representation that improves both:

```text
standalone accuracy
and
ensemble complementarity
```

is more important than a tiny isolated metric gain.

## No improvement

If Round 5 does not materially improve over Round 4, do not over-tune.

Preserve Round 4 and conclude that the present benchmark may be nearing saturation.

---

# 27. Recommended priority order

Suggested order:

## Phase 0
1. Freeze and reproduce Ordinal-SSM baseline.
2. Reproduce Round-4 ensemble OOF metrics.
3. Establish a new internal lockbox.

## Phase 1 — supervision geometry
4. Dual-head ordinal + simplex.
5. Multi-resolution ordinal.
6. Global + context ordinal multitask.

## Phase 2 — representation diversity
7. Reverse-complement augmentation/consistency.
8. Relational contrastive pretraining.
9. Objective × architecture factorial study.
10. Hybrid SSM/attention mixer.

## Phase 3 — exploratory distributional objectives
11. Quantile head.
12. Optional weak pairwise consistency.

## Phase 4 — calibration and ensemble
13. Monotone calibration.
14. Greedy OOF ensemble search.
15. Internal lockbox screening.
16. Full 5-fold finalists.
17. Freeze.
18. Single official held-out evaluation.

This order is advisory.

If a new idea produces a convincing signal, pursue it.

---

# 28. Final principle

Round 4 showed that the biggest gains came from **new learning mechanisms**, not from:

```text
more parameters
new seeds
bagging
post-hoc weighting
generic residual fitting
```

Therefore the central Round-5 question is:

\[
\boxed{
\text{Can new supervision geometry and genuinely different representations produce another strong, competent, complementary predictor?}
}
\]

Treat Ordinal-SSM as the new core.

Search aggressively but intelligently around:

```text
ordinal supervision
multi-task outcome structure
context-relative ranking
representation invariance
relational pretraining
architecture diversity
calibrated output
```

and remain open to independent directions suggested by the results.

Be rigorous about evaluation.

Be creative about modeling.

Optimize for **real predictive improvement**, not for completing a checklist.
