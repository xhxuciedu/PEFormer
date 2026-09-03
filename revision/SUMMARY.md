# Phase 1 summary — every new number, and what it does to the abstract

All eight tasks are descriptive re-analyses of frozen predictions. Nothing was trained,
tuned, thresholded or selected on held-out data. Every interval is a protospacer-clustered
bootstrap (750 clusters) unless stated. Scripts, JSON and per-task markdown are in
`revision/`.

---

## The two results that change what the paper should claim

### 1. The design problem is much harder than 0.9079 suggests — and the margin there is much larger

Pooled Spearman mixes "which locus is easy" with "which pegRNA is best at this locus".
Only the second is what anyone deploys a predictor to answer.

| evaluation | OptiPrime | PE-RankFormer | Δρ | 95% CI |
|---|---:|---:|---:|---|
| Pooled, 20,509 rows | 0.8690 | **0.9079** | +0.0389 | [+0.0288, +0.0498] |
| Within condition, n-weighted (14) | 0.7605 | 0.8111 | +0.0506 | [+0.0345, +0.0670] |
| **Within target** (670 protospacers, ≥5 designs) | **0.5472** | **0.6356** | **+0.0884** | **[+0.0761, +0.1007]** |

Ahead on **78.4%** of 670 targets. Two things follow, and they point in opposite
directions:

- **Weakens the absolute claim.** Within-target ρ is 0.636, not 0.908. The headline
  number is heavily locus-driven and is not a statement about design-ranking ability.
  "0.90" should not be presented as what the model does for a user.
- **Strengthens the comparison.** The margin more than doubles once the locus effect is
  removed, and is comfortably clear of zero. Pooling makes the reported difference
  *conservative*.

### 2. The weighting asymmetry is confirmed, and it is confined to Kim

Matched runs on development fold 0 (`r9_ctrl` uniform vs `r9_opw` under OptiPrime's own
row weights; same seed, fold, recipe, code state). Positive Δ = uniform is better.

| partition | n | zero-mass | OptiPrime weights | uniform | Δρ | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| all | 35,649 | 28.4% | 0.8830 | 0.8978 | +0.0148 | [+0.0091, +0.0218] |
| **Kim** | 19,747 | 50.8% | 0.7389 | 0.7636 | **+0.0247** | **[+0.0144, +0.0355]** |
| **Liu** | 15,902 | 0.7% | 0.8492 | 0.8556 | +0.0063 | **[−0.0052, +0.0179]** |

Exactly what the mechanism predicts: adopting OptiPrime's weights costs us 0.0247 on the
study it down-weights 10×, and **nothing measurable on Liu**, whose effective share
actually rises slightly under those weights. So:

- roughly **a third of the +0.0803 Kim margin** is attributable to a training-weight
  choice rather than to architecture or objective;
- the **+0.0220 Liu margin is untouched by weighting** and is therefore the cleanest
  architecture/objective result in the paper — which matters, because Liu is also the
  surface OptiPrime's own paper reports on.

---

## Effect on each claim in the current abstract

| abstract claim | Phase 1 verdict | evidence |
|---|---|---|
| ρ = 0.9079 vs 0.8690, Δ = +0.0389 | **unchanged** | reproduced exactly (1.1, 1.4) |
| held-out absolute level | **weakened** — 0.636 within target | 1.1 |
| within-condition +0.0506 | **unchanged**, CI corrected to [+0.0345, +0.0670] | 1.1 |
| Kim +0.0803 | **weakened** — ~⅓ is the weighting choice | 1.8 |
| Liu +0.0220 | **strengthened** — weighting-independent, and τ-b agrees | 1.3, 1.8 |
| weighting asymmetry (2.0% / 55.3%) | **strengthened** — mechanism confirmed, Kim-specific | 1.8 |
| MAE 0.0478 vs 0.0590 | **strengthened** — floor is 0.1227, so this is real skill | 1.5 |
| calibration recovers absolute efficiency | **qualified** — over-predicts top 1% by +0.065 | 1.5 |
| ceiling ⇒ ~+0.10 headroom | **unchanged but imprecise** — CI [+0.053, +0.120] | 1.6 |
| tie caveat | **strengthened** — both predictors tie-free, τ-b confirms | 1.3 |
| leakage limitation (196 groups) | **downgraded to a non-issue** — 1.0% of rows, Δ moves 0.0004 | 1.4 |
| "no effect" in the negatives table | **must be reworded to "bounded"** | 1.7 |

---

## Task-by-task numbers

**1.1 per-target / per-condition** — 670 of 750 protospacers scorable. Within-target
distribution: ours median 0.667 vs OptiPrime 0.575. Table: `task_1_1_per_target_table.csv`.

**1.2 deployment utility** (670 targets; random = exact expectation over designs)

| metric | random | OptiPrime | ours | Δ vs OptiPrime | 95% CI |
|---|---:|---:|---:|---:|---|
| precision@1 | 0.051 | 0.224 | **0.367** | +0.143 | [+0.106, +0.180] |
| precision@5 | 0.255 | 0.504 | 0.573 | +0.069 | [+0.055, +0.082] |
| NDCG@5 | 0.300 | 0.677 | 0.765 | +0.088 | [+0.071, +0.104] |
| efficiency of top-1 pick | 0.082 | 0.220 | 0.235 | +0.015 | [+0.009, +0.022] |
| regret vs best design | 0.187 | 0.049 | **0.033** | −0.015 | [−0.022, −0.009] |
| top-1 achieves ≥5% | 0.290 | 0.529 | 0.540 | +0.011 | **[−0.003, +0.026]** |
| top-1 achieves ≥20% | 0.142 | 0.370 | 0.396 | +0.026 | [+0.010, +0.044] |

**Report this honestly.** Our model finds the single best design **64% more often**
(0.367 vs 0.224), but the *efficiency you actually obtain* from its top pick is only
+0.015 better, and "top pick works at all (≥5%)" is **not significant**. The reason is
that OptiPrime's first pick is usually already decent. The gain is real and it is largest
where it is easiest to measure; the bench-relevant gain is smaller than the ranking
metrics imply.

**1.3 tie-robust and decomposed** — both predictors are effectively tie-free
(20,509 and 20,504 distinct values), so the Spearman comparison is like-for-like.

| partition | Kendall τ-b | AUROC (edits at all) | Spearman ρ \| y>0 |
|---|---|---|---|
| all | +0.0547 [+0.0426, +0.0674] | +0.0292 [+0.0220, +0.0367] | +0.0469 [+0.0340, +0.0613] |
| Liu | +0.0244 [+0.0015, +0.0489] | +0.0236 **[−0.0109, +0.0496]** | +0.0217 [+0.0021, +0.0419] |
| Kim | +0.0903 [+0.0751, +0.1067] | +0.0516 [+0.0411, +0.0632] | +0.0874 [+0.0654, +0.1082] |

The advantage is in **both** detection and quantification, not just the zero block. Liu
is the consistently weak partition across all four metrics.

**1.4 leakage-free subset** — 196 of 20,509 held-out rows (1.0%) have an exact
design-and-condition twin in training. Excluding them: ours 0.9075 (from 0.9079),
Δ = +0.0385 [+0.0286, +0.0494]. The limitation is real and negligible.

**1.5 calibration floors** — best trivial predictor (constant at training median) has
MAE 0.1227; ours 0.0478, OptiPrime 0.0590. So 0.0478 is 61% below the floor: real skill,
not the target's scale. **But the isotonic map over-predicts the tail**: the model's top
5% predicts 0.599 against 0.573 observed (+0.026), and its top 1% predicts 0.799 against
0.733 (+0.065). A user told "80%" should expect ~73%.

**1.6 ceiling uncertainty** — over 649 groups, held-out ceiling 0.9164
[0.8653, 0.9327], gap +0.1040 **[+0.0529, +0.1204]**. Estimator noise is an order of
magnitude below group-resampling noise, so the uncertainty is the 649 groups.
"Not saturated" survives on every surface; a third decimal place does not.

**1.7 multiplicity and power** — measured null SD over 7 matched mechanism-free pairs is
**0.0008**, implying 0.0013 detectable on 3 folds. **This contradicts the project's own
experience** of interventions flipping sign at ±0.004, because a matched pair cancels
fold and seed and so measures the wrong variance. The measured SD is a lower bound; the
paper should keep quoting ≈0.005 and say it is calibrated from observed fold-to-fold sign
changes. **Phase 2.1's 5-seed factorial is the only thing that measures this properly.**

---

## Recommendations before Phase 2

1. **Add per-target metrics to the main results** and demote pooled Spearman, with an
   explicit statement of what it does and does not measure.
2. **Lead the utility table with precision@1** and report the ≥5% null alongside it.
3. **Split the Kim margin explicitly** into the weighting component (~0.025) and the
   remainder, and foreground Liu as the weighting-independent result.
4. **Downgrade the leakage limitation** to a measured non-issue.
5. **Qualify the calibration claim** with the tail over-prediction.
6. **Reword the negatives table** from "no effect" to "bounded below ≈0.005".
7. **Run Phase 2.1** — it is the only route to the variance estimate that three separate
   claims in this paper currently rest on without measurement.
