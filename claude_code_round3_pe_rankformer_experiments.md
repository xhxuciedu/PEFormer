# Claude Code Research Prompt — Round 3 PE-RankFormer Optimization

## Role

Act as a senior machine-learning researcher and computational biologist.

You are continuing the PE-RankFormer project for **prime-editing efficiency prediction**. Round 2 showed that the main limitation was not model capacity but **model-selection mismatch**: the CV/development distribution was dominated by Schwank data, while the official held-out benchmark contains only Liu and Kim data.

The purpose of Round 3 is to improve predictive performance against **OptiPrime** by correcting that mismatch and testing a focused set of high-probability improvements.

Interpretability and biological explanation are **not** the current objective.

The primary goal is:

\[
\boxed{\text{maximize predictive performance on the official Liu+Kim held-out benchmark}}
\]

while maintaining strict holdout discipline.

## 1. Current verified baseline

Treat the following as fixed starting facts.

- Exact OptiPrime training corpus: 297,962 rows.
- Official held-out benchmark: 20,509 rows.
- Liu/Hsu held-out: 9,175.
- Kim/DeepPrime held-out: 11,334.
- Round-1 PE-RankFormer full held-out Spearman: 0.8865.
- OptiPrime full held-out Spearman: 0.8690.
- Round-1 Liu: 0.8349 vs OptiPrime 0.8365.
- Round-1 Kim: 0.7751 vs OptiPrime 0.7320.
- Round-2 Family C improved CV validation from 0.9192 to 0.9220, but worsened official held-out performance from 0.8865 to 0.8831.
- Approximately 58% of the training/CV pool is Schwank, but the official held-out benchmark contains zero Schwank rows.

Therefore the central Round-3 lesson is:

\[
\boxed{\text{optimize against a Liu+Kim-matched development signal, not a Schwank-dominated CV metric.}}
\]

## 2. Absolute held-out test rule

Do not use the official 20,509-row held-out set for model search.

Do not:
- inspect its metrics during architecture development;
- select hyperparameters using it;
- choose ensemble members using it;
- fit calibration using it;
- tune adapters or source weights using it.

All model selection must happen inside the 297,962 training rows.

Require an explicit flag for held-out evaluation:

```bash
--allow-heldout-evaluation
```

and create a timestamped log every time this flag is used.

## 3. Inspect and freeze the current repository

Start with:

```bash
git status
git log --oneline -n 20
```

Identify:
- round-1 baseline commit;
- round-2 final commit;
- exact training corpus and hashes;
- official CV split definitions;
- current architecture;
- simplex head;
- context features;
- training/evaluation scripts;
- existing checkpoints;
- OptiPrime reproduction code.

Create:

```text
reports/round3_initial_inventory.md
```

Freeze the round-1 model as the main baseline and create:

```text
configs/round3/baseline_round1.yaml
```

Do not change its implementation.

## 4. Round-3 strategy

Use this order:

\[
\boxed{
\text{fix validation}
\rightarrow
\text{domain adaptation}
\rightarrow
\text{source specialization}
\rightarrow
\text{context improvements}
\rightarrow
\text{moderate scaling}
\rightarrow
\text{pretraining}
\rightarrow
\text{ensemble}
}
\]

Do not begin with another broad architecture search.

## 5. Stage 0 — Build Liu+Kim-matched development folds

The official held-out composition is approximately 44.7% Liu and 55.3% Kim.

Create at least 3 independent internal development folds from the 297,962 training rows.

For each fold:

```text
train = Liu + Kim + Schwank
validation = Liu + Kim only
```

Requirements:
- protospacer-disjoint;
- use stricter target grouping if available;
- validation ratio approximately 45% Liu / 55% Kim;
- enough rows per context for stable metrics.

Save:

```text
data/processed/round3_dev_assignments.parquet
```

Internal names:

```text
round3_dev_fold_0
round3_dev_fold_1
round3_dev_fold_2
```

## 6. Stage 0 sanity check — Re-score old models

Before training anything new, evaluate existing models on the new matched development folds where possible:

```text
round-1 baseline
round-2 Family A (layerwise context)
round-2 Family C (feature branch)
round-2 Family D (layerwise context + features)
```

Create:

```text
reports/round3_validation_recalibration.md
```

Include:
- old CV Spearman;
- new Liu+Kim dev Spearman;
- known held-out Spearman;
- rank ordering under old CV;
- rank ordering under new matched dev.

The new matched validation should better reflect the known held-out ordering. If it does not, diagnose before proceeding.

## 7. Primary Stage-A metric

Use:

\[
S_{\rm target}
=
\rho(y_{\rm Liu\cup Kim},\hat y_{\rm Liu\cup Kim}).
\]

Also report:

```text
rho_Liu
rho_Kim
macro_context_spearman
Pearson
MAE
RMSE
```

Primary selection metric:

```text
matched Liu+Kim validation Spearman
```

Secondary tie-breaker:

```text
macro context Spearman
```

Do not use Schwank validation performance as the primary criterion.

## 8. Experiment family 1 — All-data training followed by Liu+Kim fine-tuning

This is the highest-priority experiment.

Stage 1:
Train the round-1 baseline architecture on all 297,962 rows.

Stage 2:
Fine-tune on Liu+Kim rows only.

Test small learning rates:

```text
1e-5
3e-5
```

Fine-tune for:

```text
5 epochs
10 epochs
```

Use matched Liu+Kim validation for checkpoint selection.

Internal names:

```text
R3-DAPT-LK-1e5
R3-DAPT-LK-3e5
```

## 9. Experiment family 2 — Liu+Kim fine-tuning with Schwank replay

Test whether some Schwank replay prevents over-specialization.

Compare:

```text
100% Liu+Kim
90% Liu+Kim + 10% Schwank
75% Liu+Kim + 25% Schwank
```

Use source-aware sampling and the same global pretrained checkpoint.

Hypothesis:
Schwank is useful for generic representation learning but should not dominate late-stage optimization.

## 10. Experiment family 3 — Source-balanced training from scratch

Test source-aware batch mixtures:

```text
Liu 40% / Kim 40% / Schwank 20%
Liu 45% / Kim 45% / Schwank 10%
Liu 45% / Kim 55% / Schwank 0%
```

Compare against natural-frequency training.

Keep architecture fixed to isolate the effect of source distribution.

## 11. Experiment family 4 — Layerwise context conditioning revisited

Round 2 showed layerwise context conditioning improved old validation.

Re-evaluate under the corrected Liu+Kim-matched benchmark.

Compare:

```text
baseline late FiLM
layerwise FiLM
adaptive LayerNorm conditioning
```

Inject context into:
- edit encoder blocks;
- pegRNA encoder blocks;
- cross-attention blocks.

Do not use MoE.

Only retain a context strategy if it improves at least 2 of the 3 matched development folds.

## 12. Experiment family 5 — Source-specific adapters

Use deterministic source-specific adapters instead of MoE.

Shared representation:

\[
h=F_\theta(x).
\]

Adapter:

\[
h' = h + A_cB_ch.
\]

Use source:

```text
c in {Liu, Kim, Schwank}
```

Test low ranks:

```text
r = 8
r = 16
r = 32
```

Start by adapting only the final 1–2 Transformer blocks or pooled representation.

Do not train separate full models yet.

## 13. Experiment family 6 — Liu-specialized and Kim-specialized models

Train a global model first.

Then fine-tune separately:

```text
global -> Liu-specialized
global -> Kim-specialized
```

Use LR:

```text
1e-5
3e-5
```

for 5–10 epochs.

At inference, if source identity is a legitimate known input:

\[
\hat y(x)
=
\begin{cases}
F_{\rm Liu}(x), & x\text{ is Liu}\
F_{\rm Kim}(x), & x\text{ is Kim}
\end{cases}
\]

Do not use official held-out rows during specialization.

## 14. Cross-source calibration

Specialized models may have different output scales.

Fit calibration using matched CV/OOF predictions only.

Try:

### Affine logit calibration

\[
\operatorname{logit}(\tilde y)
=
a_c+b_c\operatorname{logit}(\hat y)
\]

for source \(c\).

### Isotonic regression

Fit separately for Liu and Kim.

### Monotone piecewise-linear calibration

Only if needed.

Evaluate on matched OOF data:

```text
full Liu+Kim Spearman
Liu Spearman
Kim Spearman
MAE
RMSE
```

Never fit calibration on the official test set.

## 15. Experiment family 7 — Moderate model scaling

Only after the domain-adaptation experiments are working.

### Medium model

```text
d_model = 512
heads = 8
edit layers = 8
pegRNA layers = 6
cross-attention blocks = 2
FFN = 2048
```

Target: 45–60M parameters.

Train using the best adaptation strategy.

### Large model

Only if medium clearly improves matched dev performance.

```text
d_model = 640
heads = 10
edit layers = 10
pegRNA layers = 6
cross-attention blocks = 3
FFN = 2560
```

Target: 75–100M parameters.

Do not go larger unless scaling clearly helps.

## 16. Experiment family 8 — PE-specific self-supervised pretraining

Only after adaptation and scaling are characterized.

### Edit encoder objective
Masked paired-token prediction over WT/edited paired tokens.

### pegRNA encoder objective
Masked nucleotide/span reconstruction over spacer/PBS/RTT.

Pretrain for 5–10 epochs.

Then use:

```text
PE self-supervised pretraining
-> all-source supervised training
-> Liu+Kim domain adaptation
```

Compare against the identical architecture trained from scratch.

Do not begin with generic DNA foundation models.

## 17. Policy on the round-2 feature branch

Do not make the feature branch the main Round-3 model.

It may be:
- retained as an ensemble member;
- revisited only after matched validation is fixed.

If revisited, use a gated residual feature branch:

\[
h' = h + \alpha(x)h_{\rm feat}
\]

with the gate initialized near zero and feature dropout enabled.

The model must be able to ignore non-transferable engineered features.

## 18. Do not revisit MoE or ranking/correlation losses

Round 2 showed:
- MoE negative;
- correlation-aware loss negative.

Round 1 showed ranking-loss tradeoffs.

Exclude these axes from Round 3 unless new evidence provides a specific reason to reopen them.

## 19. Training protocol

Use the proven setup:

```text
AdamW
BF16
gradient clipping = 1.0
warmup = 5%
cosine decay
```

Use the proven batch size unless scaling requires adjustment.

For adaptation:
- small LR;
- short fine-tune;
- matched Liu+Kim validation checkpoint selection.

Use early stopping on matched Liu+Kim Spearman.

## 20. EMA and checkpoint averaging

For strong finalists test:

```text
EMA decay = 0.999
checkpoint averaging = best 3-5 epochs
```

Evaluate using matched OOF predictions only.

## 21. Experiment scheduling

Use this order:

### Phase 0 — Validation repair
1. Build 3 Liu+Kim-matched dev folds.
2. Re-score old models.
3. Verify matched validation better predicts known held-out ordering.

### Phase 1 — Domain adaptation
4. All-data -> Liu+Kim fine-tune.
5. +10% Schwank replay.
6. +25% Schwank replay.

### Phase 2 — Source-balanced training
7. 40/40/20.
8. 45/45/10.
9. 45/55/0.

### Phase 3 — Context conditioning
10. Layerwise FiLM.
11. Adaptive LayerNorm.

### Phase 4 — Source adapters
12. Adapter r=8.
13. Adapter r=16.
14. Adapter r=32 only if useful.

### Phase 5 — Source-specialized models
15. Liu-specialized.
16. Kim-specialized.
17. Source-calibrated combined prediction.

### Phase 6 — Scaling
18. Medium model.
19. Large model only if medium wins.

### Phase 7 — Pretraining
20. PE-specific self-supervised pretraining.

Target about 15–20 serious experiments, with conditional branching rather than blindly running everything.

## 22. Successive halving

For exploratory models:

```text
5-10 epochs initial screen
```

Then:
- stop clearly inferior models;
- continue promising models to 30 epochs;
- use 50 epochs only for finalists.

Never stop based on official test metrics.

## 23. Finalist selection

A model qualifies as a finalist only if it improves matched Liu+Kim performance across at least 2 of 3 dev folds.

Prefer models that:
- improve Kim;
- do not materially degrade Liu;
- improve macro-context Spearman.

Select about 3 finalists.

## 24. Full official 5-fold training for finalists

For each finalist:
- train all 5 official folds;
- save checkpoints;
- generate OOF predictions;
- compute OOF metrics;
- compute per-source/context metrics.

Use:

```text
results/round3/cv/
```

## 25. Build a heterogeneous source-aware ensemble

Potential members:

```text
round-1 baseline
best domain-adapted model
best layerwise-context model
best adapter model
medium-scale model
Kim-specialized model
round-2 feature model
```

Require positive or complementary OOF behavior.

Try:

### Global mean
\[
\hat y=K^{-1}\sum_k\hat y_k.
\]

### Nonnegative OOF-optimized weights
\[
\hat y=\sum_kw_k\hat y_k,
\quad
w_k\ge0,
\quad
\sum_kw_k=1.
\]

### Source-specific ensemble weights
\[
w_k^{\rm Liu},\qquad w_k^{\rm Kim}.
\]

Learn all weights from matched OOF data only.

Also test rank averaging if useful.

## 26. Freeze before official held-out evaluation

Before touching the 20,509 official held-out rows, freeze:

```text
architecture
training strategy
adaptation strategy
feature set
model size
calibration
ensemble members
ensemble weights
random seeds
```

Write:

```text
reports/round3_final_model_spec.md
```

Commit the repository and record the commit hash.

## 27. Final official held-out evaluation

Evaluate once on:

```text
Full: 20,509
Liu:   9,175
Kim:  11,334
```

Report:

```text
Spearman
Pearson
MAE
RMSE
per-condition Spearman
number of Kim conditions won
```

Compare against:
- OptiPrime;
- round-1 PE-RankFormer;
- round-2 PE-RankFormer.

## 28. Statistical testing

Use paired protospacer-clustered bootstrap.

Use 5000 resamples if cheap.

Report:

\[
\Delta\rho_{\rm R3-OptiPrime}
\]

and:

\[
\Delta\rho_{\rm R3-R1}.
\]

Report:
- observed difference;
- bootstrap mean;
- 95% CI;
- fraction of bootstrap wins;
- empirical p-value.

Bootstrap MAE differences too.

## 29. Success criteria

### Excellent
\[
\rho_{\rm full}\ge0.90
\]
and significant improvement over OptiPrime.

### Strong
\[
\Delta\rho_{\rm OptiPrime}\ge0.025
\]
with CI excluding zero.

### Very strong Kim result
\[
\rho_{\rm Kim}\ge0.80
\]
while Liu remains approximately tied with or above OptiPrime.

### Useful
Clear improvement over round 1 even if full rho remains below 0.90.

### Negative
If no model beats round 1, keep round 1 and document the negative results. Do not tune against the official test set.

## 30. Experiment tracking

For every run save:

```text
run_id
git commit
config
seed
model parameters
data mixture
adaptation stage
source weighting
context strategy
adapter rank
model scale
epochs
best epoch
matched-dev Spearman
Liu-dev Spearman
Kim-dev Spearman
macro-context Spearman
Pearson
MAE
runtime
peak VRAM
checkpoint path
```

Create:

```text
results/round3/model_search.csv
```

## 31. Required plots

Using development/OOF data during search:

1. matched-dev Spearman by model;
2. Liu vs Kim Spearman scatter;
3. performance vs source-mixture strategy;
4. effect of Schwank replay fraction;
5. per-context heatmap;
6. scaling curve;
7. ensemble-member correlation matrix;
8. calibration curves by source.

Only after model freeze:
9. official held-out comparison plots.

## 32. Research log

Maintain:

```text
reports/round3_research_log.md
```

For every experiment record:

```text
hypothesis
exact change
matched-dev result
Liu result
Kim result
whether hypothesis supported
decision
next experiment
```

## 33. Final report

Write:

```text
reports/round3_results.md
```

Structure:

```text
1. Objective
2. Why round-2 validation failed
3. New Liu+Kim-matched validation design
4. Re-evaluation of round-1/round-2 models
5. Domain-adaptive fine-tuning
6. Schwank replay
7. Source-balanced training
8. Layerwise context conditioning
9. Source-specific adapters
10. Liu/Kim specialized models
11. Calibration
12. Model scaling
13. Self-supervised pretraining
14. Finalists
15. Full 5-fold CV
16. Ensemble construction
17. Frozen final model
18. Official held-out evaluation
19. Comparison with OptiPrime
20. Statistical significance
21. Negative results
22. Compute cost
23. Recommendation for Round 4
```

## 34. Final principle

Round 2 showed:

\[
\boxed{\text{a better model on the wrong validation distribution is not necessarily a better model on the target benchmark.}}
\]

Round 3 should prioritize:

\[
\boxed{
\text{target-matched validation}
+
\text{all-source representation learning}
+
\text{Liu/Kim domain adaptation}
+
\text{source/context specialization}
}
\]

before more exotic architectures.

The main research question is:

\[
\boxed{
\text{Can better domain matching and adaptation push PE-RankFormer clearly beyond OptiPrime and beyond the round-1 baseline?}
}
\]

Start now with:

1. freeze the round-1 baseline;
2. build Liu+Kim-matched development folds;
3. re-score the existing models on the new benchmark;
4. verify the corrected validation ranking better predicts known held-out behavior;
5. run all-data -> Liu+Kim domain-adaptive fine-tuning;
6. continue systematically through the Round-3 plan.
