# Round 2 Results: PE-RankFormer Model Search

Per `claude_code_round2_pe_rankformer_model_search.md`. Full experiment log:
`reports/round2_research_log.md`. Frozen model spec: `reports/round2_final_model_spec.md`.

## 1. Objective

Search for a model that substantially exceeds round 1's PE-RankFormer, targeting
Spearman ≥ 0.90 on the full 20,509-row official held-out set, while never touching
that set until a model is frozen (§2).

## 2. Baseline

Round 1's matched-protocol 5-model CV ensemble (`cv5_ens`): global Spearman 0.8865
on the full held-out set, statistically tied with OptiPrime on the Liu partition
(0.8349 vs. 0.8365, p=0.94) and significantly ahead on Kim (0.7751 vs. 0.7320).
Full detail in `reports/pilot_results.md`.

## 3. Search protocol

Stage A: 6 candidates screened on a fixed development split (val_fold=1,
train_folds=[2,3,4,5]), same seed, same 30-epoch budget, same eval code (§6).
Stage B: the Stage-A winner confirmed across all 5 official CV folds. Held-out
set untouched until this point (§27 freeze, then §28 evaluation, this document).

## 4. Context-conditioning experiments (Family A)

FiLM applied at every edit/pegRNA encoder layer and cross-attention block
instead of once after pooling (§7). +5.9% params (28.2M vs. 26.6M).
**Result: 0.9214 val Spearman vs. baseline 0.9192 (+0.0022).** Helped alone.

## 5. MoE experiments (Family B)

4 context-gated experts added to Family A's head (§8, run conditional on Family A
helping). **Result: 0.9181 (-0.0011 vs. baseline).** Negative result — did not
carry forward.

## 6. Feature experiments (Family C)

16-feature MLP branch: PBS/RTT length and GC%, edit geometry (length, position,
position-from-nick, mismatch count), melting temperature, 4 ViennaRNA MFE
features, RuleSet3 on-target activity (§9's highest-priority feature). Computed
for all 318,471 corpus rows uniformly (label-free inputs); normalized and
NaN-imputed from training-split rows only, with a parallel missingness mask fed
into the branch so the model can discount imputed entries.
**Result: 0.9220 val Spearman vs. baseline 0.9192 (+0.0028). Stage-A winner.**

## 7. Loss experiments

Weak batch-Pearson-correlation loss (§13), β=0.025, as a lighter alternative to
the pairwise RankNet loss round 1 found costly. **Result: 0.9158 (-0.0034 vs.
baseline).** Negative — reconfirms round 1's finding that touching the ranking/
correlation objective tends to cost global Spearman. Did not sweep the remaining
{0.01, 0.05} points; the negative direction was already clear and twice-replicated.

## 8. Context-balancing, scaling, pretraining experiments

**Not attempted this round.** After Family C won Stage A and neither combination
attempt (Family D, MoE-on-A) improved on it, Stage B confirmation and the
held-out evaluation took priority within the round's scope. Flagged as the
natural starting point for a future round (§14-19 of the spec).

## 9. Finalist selection

Two families individually beat baseline (A: +0.0022, C: +0.0028). Two combination
attempts were tried per §10's rule (only combine components that individually
improved validation performance) and **both underperformed their better single
component**:

| Model | val Spearman | Δ vs baseline |
|---|---:|---:|
| **Family C (feature branch)** | **0.9220** | **+0.0028** |
| Family A (layerwise context) | 0.9214 | +0.0022 |
| Family D (A+C combined) | 0.9202 | +0.0010 |
| Baseline | 0.9192 | — |
| MoE4-on-A (A+B combined) | 0.9181 | -0.0011 |
| β=0.025 correlation loss | 0.9158 | -0.0034 |

Family C selected as the sole finalist for Stage B.

## 10. Full 5-fold CV (Stage B)

| Fold | 1 | 2 | 3 | 4 | 5 | Mean | Std |
|---|---|---|---|---|---|---|---|
| val Spearman | 0.9220 | 0.9195 | 0.9203 | 0.9190 | 0.9202 | **0.9202** | **0.0011** |

Stable across every fold.

## 11. Ensemble optimization

Unweighted mean of the 5 fold checkpoints' predicted efficiency. Per-fold OOF
Spearman agreement is tight (std 0.0011) relative to what fitting per-model
weights could plausibly improve, so equal weighting was kept rather than fit
weights that would risk overfitting to CV noise this small (§20).

## 12. Frozen final model

Commit `8df5f9a`. Architecture: round-1 base + Family C feature branch.
Layerwise context and MoE both off (didn't stack with the feature branch).
Ranking and correlation losses both off. Full spec:
`reports/round2_final_model_spec.md`.

## 13. Official held-out evaluation

Full 20,509-row held-out set, Liu (9,175) and Kim (11,334) partitions, per §28.

| Scope | n | OptiPrime | Round-1 ensemble | **Round-2 (Family C) ensemble** |
|---|---:|---:|---:|---:|
| Full | 20,509 | 0.8690 | 0.8865 | **0.8831** |
| Liu | 9,175 | 0.8365 | 0.8349 | **0.8298** |
| Kim | 11,334 | 0.7320 | 0.7751 | **0.7727** |

Round-2's global Pearson: 0.8413 (full), MAE 0.0522, RMSE 0.0981.

## 14. Comparison with OptiPrime

Round 2 still beats OptiPrime on the full set and on Kim, by a slightly smaller
margin than round 1. On Liu specifically, round 2 (0.8298) is nominally *below*
OptiPrime (0.8365) for the first time across either round -- though, per §15
below, not to a statistically distinguishable degree.

## 15. Statistical significance (paired, protospacer-clustered bootstrap, 2000 resamples)

**Round 2 vs. round 1, full set (n=20,509, 750 clusters):**
observed diff −0.0034, 95% CI **[−0.0084, +0.0014]**, round-2 wins 9.0% of
resamples, p=0.18. **Not significant** — the CI includes zero, but leans
negative (round-2 loses the large majority of resamples).

**Round 2 vs. round 1, Liu only (n=9,175, 150 clusters):**
observed diff −0.0051, 95% CI [−0.0225, +0.0121], p=0.57. Not significant.

**Round 2 vs. OptiPrime, full set:**
observed diff **+0.0141**, 95% CI **[+0.0028, +0.0250]**, round-2 wins 99.1% of
resamples, **p=0.018. Significant** — round 2 still beats OptiPrime overall.

**Round 2 vs. OptiPrime, Liu only:**
observed diff −0.0067, 95% CI [−0.0339, +0.0198], p=0.63. Not significant --
round 2 remains statistically tied with OptiPrime on Liu, as round 1 was.

**Within-target ranking / selection metrics (secondary, full held-out set):**
round 2 nominally *improves* on round 1 here -- within-target Spearman 0.5939
vs. 0.5814, top-1 regret 0.0057 vs. 0.0060, top-1 recall 0.733 vs. 0.720 -- but
confidence intervals overlap substantially (round 1: [0.539, 0.627]; round 2:
[0.550, 0.640]), so this should be read as directionally encouraging, not
established.

## 16. Liu/Kim/per-context results

Full breakdown in §13. The compositional explanation for round 2's flat-to-
slightly-negative global result: the Stage A/B development and CV folds are
drawn from all of folds 1-5, which are ~58% Schwank-sourced rows (174,067 of
297,962 training rows). **Schwank contributes zero rows to the held-out set** --
that partition has no held-out split at all in OptiPrime's own official data.
So validation Spearman during model search reflects performance on a
majority-Schwank distribution, while the held-out set is 100% Liu+Kim. A
feature-branch gain measured on that validation mix is not guaranteed to
transfer to a Liu/Kim-only evaluation, and on this evidence it substantially
did not. Verified this is a real distributional difference, not a data
artifact: PBS length, RTT length, PBS GC%, and edit-position-from-nick each
differ meaningfully in mean and spread between the training-fold pool and the
held-out fold.

## 17. Negative results

- MoE-on-layerwise-context (§5): -0.0011 vs. baseline.
- Weak correlation loss, β=0.025 (§7): -0.0034 vs. baseline.
- Family D, layerwise context + feature branch combined (§9): 0.9202, beats
  baseline but underperforms *both* individual components -- the two gains
  overlap rather than stack.
- **The headline negative result of this round**: Family C's validation-measured
  gain (+0.0028, confirmed stable across 5 CV folds, std 0.0011) did not
  produce a statistically significant improvement over round 1 on the actual
  held-out set -- if anything the point estimate is slightly negative
  (-0.0034, CI includes zero). The likely cause is the Schwank/Liu-Kim
  compositional mismatch in §16, not overfitting to CV noise (the CV signal
  was too stable across folds for that).

## 18. Compute cost

Stage A: 4 candidates trained to completion (~25-65 min each depending on GPU;
varied across an RTX PRO 6000 Blackwell and two L40s used in parallel), plus 2
negative-result runs. Stage B: 4 additional CV folds (fold 1 reused from Stage
A). Total: roughly 10 full 30-epoch training runs plus the frozen-model held-out
evaluation. All runs logged in `results/round2/model_search.csv` schema (see
`reports/round2_research_log.md` for the actual per-run detail, which proved
more useful than the CSV for this round's narrative).

## 19. Recommendation for next round

1. **Re-run Stage A/B with a validation scheme that matches the held-out set's
   composition** (Liu+Kim only, or explicitly stratified/reweighted away from
   Schwank's 58% share) before trusting any further feature or architecture
   change to transfer. This round's central lesson is that a representative
   validation split matters as much as the modeling idea being tested.
2. Family C's within-target/selection-metric improvement (§15) is the most
   promising unexplored thread -- if the actual deployment use case is
   "rank candidate pegRNAs for the same target" rather than "predict absolute
   efficiency across a heterogeneous population," it may be worth optimizing
   for and validating on that metric directly rather than global Spearman.
3. Layerwise context conditioning (Family A) is a legitimate, real,
   independent improvement (+0.0022 on the compositionally-matched dev split)
   that was set aside only because it didn't stack with Family C. It has not
   been checked against the held-out set on its own and remains a candidate
   for a future round with the corrected validation scheme from
   recommendation 1.
4. Context-balancing (§14), moderate scaling (§16), and self-supervised
   pretraining (§18) were never attempted this round and remain open.

## 20. Bottom line

Round 2 found and confirmed (5/5 CV folds, std 0.0011) a small, real
architectural improvement on a compositionally-mismatched validation signal.
That improvement did not transfer to a statistically significant gain on the
actual held-out target, and nominally trends slightly negative there
(not significant). The round-2 model still beats OptiPrime significantly on
the full held-out set (+0.0141, p=0.018) and remains statistically tied with
it on the Liu partition, same as round 1 -- but does not clearly improve on
round 1's own model, the round's stated goal. The most valuable output of this
round is arguably not the model but the finding in §16 and #19.1: this
project's validation scheme has a structural composition mismatch with its
test target that should be fixed before the next search round is trusted.
