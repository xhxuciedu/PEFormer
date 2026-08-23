# Claude Code Research Prompt — Round 6: Learning Context-Dependent Reordering in Prime Editing

## Role

Act as a senior machine-learning researcher and computational biologist.

You are continuing a multi-round research program on **prime-editing efficiency prediction**. The project has already produced a strong benchmark result against OptiPrime. The purpose of Round 6 is **not** another broad architecture sweep. It is to attack one specific, well-supported remaining failure mode:

\[
\boxed{
\text{the model predicts a nearly universal pegRNA ranking across contexts,}
\\
\text{while the real data show substantial context-dependent reordering.}
}
\]

The current best single backbone is **Ordinal-SSM**. Treat it as the default backbone unless the data strongly support a different choice.

Round 6 should focus on learning the explicit **design × experimental-context interaction**.

Interpretability is still secondary. Predictive performance is the primary objective.

---

# 0. Research freedom — important

This specification is a **research program, not a rigid checklist**.

You are explicitly encouraged to:

- pursue independent leads if experiments reveal a better mechanism;
- stop planned branches early when evidence makes them low-value;
- introduce alternative interaction parameterisations or objectives;
- combine ideas when there is strong empirical support;
- run small diagnostics not listed here;
- reformulate the interaction target if the data expose a cleaner construction.

Do not mechanically execute every experiment.

However, preserve these non-negotiable rules:

1. **No official held-out leakage.**
2. **No model selection using official held-out labels.**
3. **Use matched mechanism-free controls for every subtle effect.**
4. **Require replication across folds before trusting effects below ~0.005 Spearman.**
5. **Document every independent lead and every negative result.**
6. **Keep the pipeline reproducible and auditable.**
7. **Do not exploit Spearman tie behavior or other benchmark artifacts.**

Think like an independent researcher, not a script executor.

---

# 1. Current verified state

Use the current repository and report artifacts as the starting point.

## Current official held-out results

```text
Full held-out: 20,509 rows
Liu:           9,175 rows
Kim:          11,334 rows
```

Current performance:

| Model | Full | Liu | Kim |
|---|---:|---:|---:|
| OptiPrime | 0.8690 | 0.8365 | 0.7320 |
| PE-RankFormer final ensemble | 0.9079 | 0.8585 | 0.8124 |
| Ordinal-SSM single model | 0.9082 | 0.8576 | 0.8151 |

The single Ordinal-SSM model is effectively indistinguishable from the ensemble on held-out and is operationally preferable.

Therefore:

\[
\boxed{\text{Ordinal-SSM is the Round-6 core model.}}
\]

---

# 2. What Round 5 established

Approximately fifty post-freeze experiments failed to materially improve the current system.

Important negative results include:

```text
dual ordinal + simplex head             +0.0008
multi-resolution ordinal                +0.0008
context-relative ordinal auxiliary      +0.0016
quantile head                           -0.0140
true CORAL parameterization             +0.0005
monotonicity penalty                    +0.0007
hurdle / zero-inflation head            -0.0022
hybrid SSM+attention                     negative
larger SSM state dimension              near-null
wider/deeper SSM                        near-null / non-replicating
MoE                                     non-replicating / negative
layerwise context on SSM                +0.0001
source reweighting                      non-replicating / negative
bagging                                 near-null
new seed                                +0.0022
```

Therefore Round 6 should **not** repeat generic searches over:

```text
capacity
depth
width
seed
bagging
standard MoE
generic context FiLM
ordinary ranking losses
quantile heads
zero-inflation heads
global ensemble weighting
```

unless a genuinely new mechanism changes the reason for testing them.

---

# 3. The key remaining failure mode

For designs measured in two experimental conditions, the report found:

\[
\text{mean true cross-condition rank correlation}
=
0.683
\]

but:

\[
\text{mean predicted cross-condition rank correlation}
=
0.835.
\]

Thus the model preserves design ordering across contexts far too strongly.

Interpretation:

\[
s(d,c)
\approx
g(d)+a(c)
\]

where:

- \(d\): pegRNA/design;
- \(c\): experimental context;
- \(g(d)\): nearly universal design quality;
- \(a(c)\): context-level scale/shift.

But the true data require:

\[
s(d,c)
=
g(d)+a(c)+h(d,c)
\]

where:

\[
\boxed{h(d,c)}
\]

is a **design × context interaction that changes ranking**.

Round 6 should explicitly target \(h(d,c)\).

---

# 4. Evidence that the interaction is learnable

The current report contains two important observations.

## 4.1 Replicate-based headroom

On Kim development data:

```text
model correlation ≈ 0.7869
empirical repeatability ceiling estimate ≈ 0.9026
gap ≈ +0.1157
```

Even after excluding exact zeros:

```text
model ≈ 0.8465
ceiling ≈ 0.9489
gap ≈ +0.1024
```

Therefore the current model is probably not noise-limited.

Treat the ceiling as an **empirical repeatability-based ceiling estimate**, not a mathematically exact ceiling.

## 4.2 Rank-shift predictability

Cross-condition rank shift can already be predicted from only ~16 engineered design features at approximately:

\[
\rho \approx 0.275.
\]

Thus cross-context reordering contains learnable signal.

This strongly motivates direct interaction modeling.

---

# 5. Why previous context conditioning failed

Layerwise FiLM on the SSM backbone was effectively null on prediction:

```text
delta ≈ +0.0001
```

and moved cross-condition prediction correlation in the wrong direction:

```text
control ≈ 0.828
layerwise FiLM ≈ 0.875
```

So more FiLM made the model **more context-invariant**.

Interpretation:

FiLM is mainly a per-channel scale/shift mechanism. It makes it easier to learn:

```text
context mean
context scale
```

but does not force context-specific reordering.

Therefore Round 6 must use **interaction-capable scoring mechanisms and interaction-specific losses**, not simply “more context conditioning.”

---

# 6. Core modeling decomposition

Use the conceptual decomposition:

\[
s(d,c)
=
g(d)
+
a(c)
+
h(d,c).
\]

The goal is not necessarily to implement these exact terms literally, but every proposed mechanism should be evaluated according to whether it improves learning of \(h(d,c)\).

Recommended interpretation:

- \(g(d)\): universal design representation from Ordinal-SSM;
- \(a(c)\): context-level intercept/scale;
- \(h(d,c)\): context-dependent ranking correction.

A model that only improves \(g(d)\) or \(a(c)\) is not solving the diagnosed failure.

---

# 7. Stage 0 — Build the interaction analysis dataset

Before training any new model, create a clean interaction dataset from the training corpus.

Identify repeated designs measured in multiple contexts.

Define a design identity key using all sequence/design fields required to ensure the same physical pegRNA/edit design is being compared.

Do not merge:

```text
plain pegRNA
epegRNA
different motif
different linker
different scaffold
different edit geometry
```

unless they are truly the same design.

For each repeated design \(d\), collect all contexts:

\[
\{c_1,\ldots,c_m\}.
\]

Save:

```text
data/processed/round6_repeated_designs.parquet
data/processed/round6_context_pairs.parquet
```

For each design-context observation retain:

```text
source
cell line
PE system
Cas9/PAM
scaffold
motif
all design features
observed efficiency
context-specific percentile/rank
global percentile/rank
official fold
dev fold
lockbox status
```

Verify:

```text
zero cross-fold leakage for any pair used in validation
```

---

# 8. Define context-relative rank carefully

For each context \(c\), compute the empirical percentile:

\[
r(d,c)=F_c(y_{d,c})
\]

using **training-only statistics within each fold**.

Do not compute validation ranks using validation labels in a way that leaks information into training.

For training targets, context-specific empirical ranks are allowed because they are labels.

For validation scoring, evaluate model predictions against observed validation ranks, but do not use validation rank distributions to transform model inputs.

Handle ties carefully.

Record both:

```text
average rank
dense rank
fractional rank
```

for diagnostics.

Do not exploit zero-mass tie conventions to artificially improve Spearman.

Use a fixed rank convention consistently.

---

# 9. Primary experiment A — Same-design cross-context difference loss

This is the highest-priority Round-6 experiment.

For the same design \(d\) measured in contexts \(c_1,c_2\), define observed context shift:

\[
\Delta r_d(c_1,c_2)
=
r(d,c_1)-r(d,c_2).
\]

The model predicts:

\[
\widehat{\Delta r}_d(c_1,c_2)
=
s(d,c_1)-s(d,c_2).
\]

Train:

\[
L
=
L_{\rm ordinal}
+
\lambda_{\rm shift}L_{\rm shift}.
\]

Use:

\[
L_{\rm shift}
=
\operatorname{Huber}
\left(
\widehat{\Delta r}
-
\Delta r
\right).
\]

Why this is important:

If:

\[
s(d,c)=g(d)+a(c)+h(d,c),
\]

then taking the same design across contexts cancels:

\[
g(d).
\]

This directly reduces the universal-design shortcut.

Test only a narrow set:

```text
lambda_shift = 0
lambda_shift = 0.05
lambda_shift = 0.10
lambda_shift = 0.25
```

Use a matched no-mechanism rerun for each experiment batch.

Internal name:

```text
R6-OrdSSM-ShiftLoss
```

---

# 10. Shift-target alternatives

If raw percentile difference is noisy, test one or two alternatives.

## A. Signed shift classification

Predict:

```text
context 1 rank higher
tie / negligible shift
context 2 rank higher
```

using thresholds:

\[
|\Delta r|<\epsilon.
\]

## B. Quantized shift bins

For example:

```text
large down
small down
no change
small up
large up
```

## C. Pairwise context preference

Predict:

\[
P(r(d,c_1)>r(d,c_2)).
\]

Do not run all variants unless the primary continuous loss appears unstable.

---

# 11. Primary experiment B — Difference-in-differences interaction loss

Construct \(2\times2\) rectangles:

```text
design d1 in context c1
design d1 in context c2
design d2 in context c1
design d2 in context c2
```

Define observed interaction:

\[
I
=
[r(d_1,c_1)-r(d_2,c_1)]
-
[r(d_1,c_2)-r(d_2,c_2)].
\]

Predicted interaction:

\[
\hat I
=
[s(d_1,c_1)-s(d_2,c_1)]
-
[s(d_1,c_2)-s(d_2,c_2)].
\]

Train:

\[
L
=
L_{\rm ordinal}
+
\lambda_{\rm DID}
L_{\rm DID}.
\]

Possible:

\[
L_{\rm DID}
=
\operatorname{Huber}(\hat I-I).
\]

This objective removes:

- much of the universal design effect;
- context-level mean shift;
- context-level scale shortcut.

It directly supervises:

\[
\boxed{\text{context-induced reordering}}
\]

which is the diagnosed missing signal.

Before training:

- count the number of valid \(2\times2\) rectangles;
- quantify their context coverage;
- quantify their edit/design diversity;
- ensure no validation leakage.

If rectangles are sparse, use stochastic pair construction from repeated-design groups.

Internal name:

```text
R6-OrdSSM-DID
```

---

# 12. Difference-in-differences classification variant

For robustness, optionally convert \(I\) into:

```text
interaction positive
interaction negligible
interaction negative
```

or predict only:

\[
\operatorname{sign}(I).
\]

This may be more stable under noisy efficiency labels.

Use only if continuous \(I\) regression is unstable.

---

# 13. Primary experiment C — Bilinear context-specific ordinal scoring

Replace the universal final scoring direction with a context-dependent direction.

Let Ordinal-SSM produce design embedding:

\[
z_d\in\mathbb R^p.
\]

Let context embedding be:

\[
e_c\in\mathbb R^q.
\]

Define:

\[
s(d,c)
=
w_0^\top z_d
+
z_d^\top Ue_c
+
b_c.
\]

The crucial term is:

\[
\boxed{z_d^\top Ue_c}.
\]

This gives context-specific reordering capacity.

Use low-rank factorization:

\[
U=AB^\top
\]

with:

```text
rank = 8
rank = 16
rank = 32
```

Start with rank 8 and 16.

Use the ordinal threshold head on top of context-specific scores.

Do not introduce large context-specific parameter counts initially.

Internal name:

```text
R6-BilinearOrdSSM
```

---

# 14. Bilinear head variants

If useful, test one alternative:

## Context-conditioned threshold logits

For ordinal threshold \(k\):

\[
z_k(d,c)
=
w_k^\top z_d
+
z_d^\top U_ke_c
+
b_{k,c}.
\]

Use shared low-rank factors across \(k\) to limit capacity.

Do not give every threshold and context a completely independent full matrix.

---

# 15. Primary experiment D — Hierarchical context-specific ordinal heads

Alternative to the bilinear head.

Use:

\[
w_c
=
w_{\rm shared}
+
\Delta w_c.
\]

Regularize:

\[
L_{\rm reg}
=
\lambda_{\rm ctx}
\sum_c
\|\Delta w_c\|^2.
\]

Or use low-rank updates:

\[
W_c
=
W_0+A_cB_c.
\]

Test:

```text
rank 8
rank 16
rank 32
```

Start with context-specific adaptation only at:

```text
final pooled embedding
final ordinal head
```

Do not modify every SSM block initially.

Goal:

> allow context-dependent ranking while strongly shrinking low-data contexts toward the shared model.

Internal name:

```text
R6-HierarchicalHeads
```

---

# 16. Context hierarchy

Experimental context is structured.

Possible hierarchy:

```text
source
cell line
PE system
Cas9/PAM
scaffold/epegRNA state
```

Consider hierarchical decomposition:

\[
e_c
=
e_{\rm source}
+
e_{\rm cell}
+
e_{\rm PE}
+
e_{\rm interactions}.
\]

For example:

\[
w_c
=
w_0
+
\Delta w_{\rm cell}
+
\Delta w_{\rm PE}
+
\Delta w_{\rm cell\times PE}.
\]

Use only low-rank or strongly regularized interactions.

Do not explode into a separate unrestricted head for every fine-grained context unless data support it.

---

# 17. Primary experiment E — Interaction-aware paired batch sampling

Ordinary minibatches may rarely expose same-design cross-context comparisons.

Implement an interaction-aware sampler.

Suggested training mix:

```text
50-75% ordinary examples
25-50% repeated-design / multi-context examples
```

For interaction batches:

- sample a design appearing in >=2 contexts;
- sample 2-4 contexts for that design;
- optionally sample another matched design to form a DID rectangle.

Compute:

```text
ordinary ordinal loss
+
shift loss
+
DID loss where available
```

Log per-batch:

```text
number repeated-design pairs
number DID rectangles
context diversity
effective interaction sample count
```

Verify the sampler does not distort global source composition excessively.

---

# 18. Primary experiment F — Dedicated rank-shift interaction branch

The report already shows that ~16 engineered design features predict context-induced rank shift at approximately:

\[
\rho \approx 0.275.
\]

Exploit that signal directly.

Let:

\[
s_{\rm base}(d,c)
\]

be Ordinal-SSM's score.

Add:

\[
h_{\rm shift}(d,c)
=
f(
x_{\rm design},
e_c
).
\]

Final score:

\[
s(d,c)
=
s_{\rm base}(d,c)
+
\alpha h_{\rm shift}(d,c).
\]

Train \(h_{\rm shift}\) specifically using:

```text
same-design context-shift labels
DID labels
```

rather than total residuals.

Possible interaction inputs:

```text
PBS length
RTT length
edit type
edit position
nick distance
PBS GC
RTT GC
melting temperatures
MFE
RuleSet3 / DeepSpCas9
motif/epegRNA state
other existing engineered features
```

Use a small MLP or low-capacity interaction network.

Do not call this generic residual learning.

Its target is specifically:

\[
\boxed{\text{context-dependent rank shift}.}
\]

Internal name:

```text
R6-ShiftBranch
```

---

# 19. Diagnostic feature analysis for rank shift

Before training the branch, quantify which engineered features predict:

\[
\Delta r(d,c_1,c_2).
\]

Use only development/OOF data.

Possible analyses:

```text
univariate Spearman
permutation importance
small gradient-boosted model
regularized linear model
```

Do not interpret biologically yet.

The goal is simply to identify whether rank shift is concentrated in a small subset of features.

Save:

```text
results/round6/rank_shift_feature_analysis.csv
```

---

# 20. Combine objective and architecture only after isolated tests

Do not start with an overcomplicated model.

Recommended sequence:

1. Ordinal-SSM control.
2. + shift loss only.
3. + DID loss only.
4. bilinear head only.
5. hierarchical head only.
6. interaction-aware sampler only.
7. shift branch only.

Only after isolated mechanisms show replicated value should you combine:

```text
best interaction objective
+
best context-dependent head
+
interaction-aware sampler
```

Avoid combining multiple null effects.

---

# 21. Interaction-specific evaluation metrics

For every Round-6 candidate, report:

## A. Pooled predictive performance

```text
Spearman
Pearson if calibrated
MAE if calibrated
```

## B. Liu Spearman

## C. Kim Spearman

## D. Macro context Spearman

## E. Cross-condition predicted rank similarity

For each repeated design measured across contexts, compute the model's cross-context ranking similarity.

Aggregate across context pairs:

\[
\rho_{\rm cross}^{\rm pred}.
\]

Compare to empirical truth:

\[
\rho_{\rm cross}^{\rm true}\approx0.683.
\]

Current model baseline:

\[
\rho_{\rm cross}^{\rm pred}\approx0.835.
\]

## F. Rank-shift prediction

\[
\rho(
\widehat{\Delta r},
\Delta r
).
\]

## G. DID interaction prediction

\[
\rho(\hat I,I).
\]

## H. Context-specific regret / selection metrics if available

The mechanism-specific metrics are essential.

A model that improves pooled Spearman while becoming even more context-invariant should be treated skeptically.

---

# 22. Mechanism-specific promotion rule

A candidate should be considered especially promising if it does all three:

1. improves or preserves pooled validation Spearman;
2. improves Kim or macro-context performance;
3. moves predicted cross-context rank similarity **toward** empirical truth.

For example:

```text
pooled rho: +0.003
Kim rho:    +0.006
cross-context predicted rho: 0.835 -> 0.77
```

is more meaningful than:

```text
pooled rho: +0.003
Kim rho:    +0.002
cross-context predicted rho: 0.835 -> 0.87
```

even if pooled metrics are similar.

Do not select solely by mechanism metric, however.

Prediction remains primary.

---

# 23. Mechanism-free control — mandatory

Every subtle experiment must have a matched no-mechanism control.

Examples:

```text
new sampler + shift loss
vs
new sampler + lambda_shift=0

bilinear head
vs
same parameter count with context term disabled

hierarchical head
vs
shared head with equal parameter budget

interaction branch
vs
same branch receiving shuffled context labels
```

A plain seed/batch-size rerun should also be included periodically.

Reason:

Previous rounds showed that +0.0015 to +0.0020 apparent gains can arise from ordinary run variation.

Do not attribute a small positive delta to the mechanism without the matched control.

---

# 24. Replication rule

Effects below approximately:

\[
+0.005
\]

should not be trusted from one fold.

Require:

```text
same-sign effect on >= 3 folds
```

and preferably:

```text
positive clustered-bootstrap difference
```

before promotion.

If a candidate is:

```text
+0.004
+0.003
-0.002
```

do not call it successful.

If:

```text
+0.004
+0.005
+0.003
```

then continue.

---

# 25. Development protocol

Use:

```text
matched Liu+Kim development folds
```

with protospacer-disjoint evaluation.

Do not use the official held-out set.

Do not use consumed lockbox labels for iterative selection.

If no fresh Kim lockbox remains, rely on:

```text
official OOF
matched dev folds
cross-fold replication
```

and document the limitation.

---

# 26. Round-6 internal evaluation hierarchy

Suggested:

## Stage A
One matched dev fold for debugging and coarse screening.

## Stage B
Three matched dev folds for replication.

## Stage C
Full official 5-fold OOF for finalists.

## Stage D
Freeze.

## Stage E
Official held-out evaluation only if there is a compelling reason.

Because the official held-out set has already informed multiple rounds, avoid another final query for trivial improvements.

Prefer external validation if available.

---

# 27. Replicate-free held-out audit

This is a post-hoc audit, not a model-selection surface.

The current report notes that some Kim replicate groups cross the official train/held-out split.

Build a subset:

\[
\boxed{
\text{held-out rows with no exact design-and-condition replicate in training}
}
\]

Evaluate:

```text
Ordinal-SSM
final PE-RankFormer system
OptiPrime
```

Report:

```text
n
number of protospacer clusters
Spearman
bootstrap CI
delta vs OptiPrime
```

Label this explicitly:

```text
post-hoc robustness audit
```

Do not use it to choose Round-6 models.

---

# 28. Verify the repeated 0.9082 number

The current report gives:

```text
Ordinal-SSM OOF development rho = 0.9082
Ordinal-SSM held-out rho        = 0.9082
```

This may be coincidence.

Mechanically regenerate both numbers from their underlying metric artifacts.

Verify:

```text
different prediction files
different row counts
correct fold aggregation
correct held-out aggregation
```

Document:

```text
reports/round6_ordssm_metric_audit.md
```

This is a reporting audit, not a model experiment.

---

# 29. Focus diagnostics on weak contexts

Current weak large conditions include roughly:

```text
A549 / PE2
MDA-MB-231 / PE2
A549 / PE4
```

while some DLD1 conditions are much stronger.

For every promising interaction model, report per-context deltas.

Ask:

> Does the model improve where context-specific reordering is strongest or where baseline cross-context invariance is most wrong?

Do not tune directly to A549 or MDA-MB-231.

Use them diagnostically.

---

# 30. Context-pair stratification

For each context pair, compute:

```text
true cross-context rho
predicted cross-context rho
excess invariance
rank-shift prediction rho
number of shared designs
```

Sort context pairs by:

\[
\text{excess invariance}
=
\rho_{\rm pred}-\rho_{\rm true}.
\]

Promising interaction models should reduce excess invariance particularly on highly mis-modeled context pairs.

Save:

```text
results/round6/context_pair_diagnostics.csv
```

---

# 31. Context identifiability audit

Before building complex context heads, quantify whether context embeddings are sufficiently supported.

For each context:

```text
number of training rows
number of unique protospacers
number of repeated designs
number of cross-context pairs
```

Low-data contexts may need hierarchical shrinkage rather than independent heads.

Use this audit to choose between:

```text
bilinear shared interaction
hierarchical head
context-specific adapter
```

---

# 32. Optional advanced lead — context-conditioned hypernetwork

Only if bilinear heads work and more capacity appears useful.

Use a small hypernetwork:

\[
\theta_c
=
H(e_c)
\]

to produce low-rank scoring parameters.

Do not generate the full SSM weights.

Generate only:

```text
final scoring vectors
low-rank adapters
ordinal-head offsets
```

Strongly regularize.

This should be considered only if simpler bilinear/hierarchical mechanisms show positive evidence.

---

# 33. Optional advanced lead — interaction contrastive objective

For the same design measured in contexts \(c_1,c_2\), construct representation:

\[
z(d,c).
\]

Do not force all contexts together.

Instead encourage:

\[
z(d,c_1)-z(d,c_2)
\]

to predict observed rank shift.

Possible auxiliary objective:

\[
\operatorname{MLP}
(
z(d,c_1)-z(d,c_2)
)
\rightarrow
\Delta r.
\]

This is representation-level interaction learning.

Only test if output-level shift losses appear promising.

---

# 34. Optional advanced lead — low-rank tensor interaction

Treat:

```text
design representation
cell embedding
PE-system embedding
```

as factors in a low-rank tensor model.

Example:

\[
h(d,c)
=
\sum_{r=1}^R
(a_r^\top z_d)
(b_r^\top e_{\rm cell})
(c_r^\top e_{\rm PE}).
\]

This can model cell × editor × design interactions while remaining low-rank.

Only test if simple bilinear context interaction is insufficient.

---

# 35. Do not exploit metric artifacts

The report found that collapsing low scores into a tied zero-like block can artificially improve pooled Spearman.

Do not use:

```text
score clipping
hard zero thresholding
tie injection
rank collapsing
```

as performance improvements.

If any postprocessing creates ties, report:

```text
before
after
number of ties
effect on Spearman
```

and reject the method if the gain is primarily due to benchmark tie behavior.

---

# 36. Calibration

Keep the existing monotone calibration pipeline for absolute efficiency estimates.

Do not use calibration as a model-selection lever.

For final promising models, report:

```text
Spearman
Pearson
MAE
RMSE
```

with calibration fit strictly on OOF development predictions.

The core Round-6 research metric remains rank prediction.

---

# 37. External-validation track — strongly recommended

Run in parallel with the performance track.

Identify an independent prime-editing dataset satisfying:

```text
not in OptiPrime training corpus
not in current held-out benchmark
no target/protospacer overlap if possible
sufficiently documented context
compatible sequence reconstruction
```

Strongest cases:

```text
new cell line
new PE system
new laboratory
endogenous validation
arrayed validation
```

For the external benchmark:

1. document provenance;
2. freeze the current Ordinal-SSM and OptiPrime pipelines;
3. do not adapt on external labels;
4. run both fairly;
5. report standalone results separately.

If a Round-6 interaction model is developed later, evaluate it externally only after freezing it.

Do not use external validation for iterative tuning.

---

# 38. External benchmark search artifact

Create:

```text
reports/round6_external_dataset_inventory.md
```

For each candidate dataset record:

```text
publication
accession
sample size
cell line
PE system
design fields available
efficiency outcome
overlap risk
sequence reconstruction feasibility
whether OptiPrime can run natively
whether labels are public
```

If no suitable dataset exists, document that.

---

# 39. Experiment order

Recommended priority:

## Phase 0 — audits and dataset construction

1. Reproduce Ordinal-SSM baseline.
2. Verify the duplicated 0.9082 reporting number.
3. Build repeated-design multi-context dataset.
4. Build context-pair diagnostics.
5. Build context identifiability summary.
6. Build replicate-free held-out audit subset.

## Phase 1 — interaction objectives

7. Same-design shift loss.
8. DID loss.
9. Interaction-aware sampler.

## Phase 2 — interaction-capable scoring

10. Bilinear context-dependent ordinal head.
11. Hierarchical context-specific heads.
12. Shift-feature interaction branch.

## Phase 3 — combinations

13. Best interaction loss + best head.
14. Best loss + paired sampler.
15. Best head + paired sampler.
16. Best three-way combination only if isolated effects replicate.

## Phase 4 — advanced ideas if justified

17. Hypernetwork low-rank head.
18. Tensor interaction.
19. Representation-level shift objective.

## Parallel track

20. Independent external dataset search.
21. Frozen external evaluation if feasible.

This order is advisory.

Pursue independent high-value leads when justified.

---

# 40. Promotion criteria

A candidate can advance if it satisfies one of:

## Path A — clear predictive gain

\[
\Delta\rho_{\rm pooled}\ge0.005
\]

with replicated sign.

## Path B — strong Kim/context gain

For example:

```text
Kim +0.008 or more
with pooled performance preserved
```

## Path C — mechanism-success candidate

Even if pooled gain is modest:

```text
cross-context predicted rho moves substantially toward truth
rank-shift prediction improves
Kim/context macro improves
no pooled degradation
```

Such a candidate deserves further combination experiments.

Do not advance models that only improve context mean/scale.

---

# 41. Success criteria for Round 6

## Major success

A new model that improves Ordinal-SSM by:

\[
\Delta\rho_{\rm pooled}\ge0.005
\]

with replicated OOF evidence.

## Strong interaction success

A model that:

```text
reduces cross-context excess invariance substantially
+
improves Kim
+
preserves pooled rho
```

even if pooled gain is <0.005.

## External-validation success

A frozen model clearly outperforming OptiPrime on a truly independent dataset.

This may be scientifically more important than another small gain on the official benchmark.

## Negative result

If explicit interaction modeling does not help:

- document it clearly;
- do not revert to broad random search;
- reconsider whether missing experimental covariates, not model structure, limit context reordering.

---

# 42. Experiment tracking

Create:

```text
results/round6/model_search.csv
```

Columns:

```text
run_id
architecture
interaction_mechanism
loss_shift
loss_did
sampler
context_head
params
seed
fold
pooled_spearman
liu_spearman
kim_spearman
macro_context_spearman
cross_context_pred_rho
cross_context_true_rho
excess_invariance
rank_shift_rho
did_rho
mechanism_control_delta
runtime_min
peak_vram_gb
checkpoint
decision
notes
```

---

# 43. Research log

Maintain:

```text
reports/round6_research_log.md
```

For each experiment record:

```text
hypothesis
specific shortcut being targeted
exact implementation
matched control
pooled result
Kim result
context result
cross-condition diagnostic
replication status
decision
independent observations
next step
```

Include all negative results.

---

# 44. Required figures

Create:

```text
results/round6/figures/
```

At minimum:

1. true vs predicted cross-context correlation by context pair;
2. excess invariance by context pair;
3. rank-shift prediction scatter;
4. DID prediction scatter;
5. pooled Spearman vs excess invariance;
6. Kim Spearman vs excess invariance;
7. per-context delta heatmap;
8. mechanism-control comparison;
9. repeated-design coverage plot;
10. replicate-free held-out comparison;
11. external-validation comparison if available.

---

# 45. Round-6 report

Write:

```text
reports/round6_results.md
```

Suggested structure:

```text
1. Objective
2. Current Ordinal-SSM baseline
3. Why generic architecture search was stopped
4. Evidence for remaining headroom
5. Cross-context reordering diagnosis
6. Repeated-design interaction dataset
7. Same-design shift loss
8. Difference-in-differences loss
9. Interaction-aware sampling
10. Bilinear context-dependent head
11. Hierarchical context-specific heads
12. Rank-shift feature branch
13. Combined interaction models
14. Mechanism-specific diagnostics
15. Per-context results
16. Replicate-free held-out audit
17. External-validation search
18. External-validation results if available
19. Negative results
20. Compute cost
21. Recommendation for next phase
```

---

# 46. Final principle

The central Round-6 hypothesis is:

\[
\boxed{
\text{the remaining predictive gap is dominated by underlearned design × context interaction.}
}
\]

The current model already learns:

```text
general design quality
context-level efficiency shifts
global ranking
```

very well.

The next frontier is:

\[
\boxed{
\text{learning how context changes which design is best.}
}
\]

Therefore do not spend Round 6 trying to make the universal ranking slightly better.

Build objectives and parameterisations that make **context-specific reordering unavoidable**.

Be rigorous.

Use matched controls.

Require replication.

Be creative when the data reveal a better interaction mechanism.

Optimize for **real predictive improvement and real context sensitivity**, not for completing a checklist.
