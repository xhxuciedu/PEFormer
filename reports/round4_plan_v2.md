# Round 4, revised plan — my own, written after the §9/§10/§11 results

The round-4 spec front-loaded post-hoc ensemble machinery (context gating, stacking,
residual learning). All three are now closed and all three failed. This document is
the plan I think the evidence actually supports, replacing the remainder of the spec's
ordering. It is written before seeing any new results so it can be judged honestly.

## Where we stand

| Quantity | Value |
|---|---:|
| Round-3 ensemble, official held-out | **0.8933** |
| OptiPrime, same rows | 0.8690 |
| Current margin | **+0.0243** (p < 0.0001) |
| Target | **≥ 0.90 held-out**, margin ≥ +0.03 |
| Gap to target | **+0.0067** |

## What the evidence says

**1. Post-hoc combination is exhausted.** Context gating +0.0000, stacking +0.0013,
residual learning +0.0006 *oracle ceiling*. Nothing recoverable remains in the current
members' outputs, nor in engineered features, nor in experimental context. I am not
spending more of this round on combination machinery.

**2. Error decorrelation is the only lever that has ever paid.** Every real gain came
from adding a member whose *errors* differ, not from a better single model:

| Change | Dev gain | Member correlation |
|---|---:|---:|
| + Family C (2nd member) | +0.0121 | 0.955 |
| + Family A (3rd member) | +0.0063 | 0.945 |
| DAPT (+0.0015 standalone) | ~0 | 0.997 -- worthless |

The DAPT row is the important one: a member that is *better standalone* but 0.997
correlated adds nothing. Standalone accuracy is close to irrelevant; **decorrelation is
the objective.**

**3. Diminishing returns set the budget.** +0.0121 then +0.0063 implies roughly +0.003,
+0.002, +0.0015 for members 4, 5, 6. Reaching +0.0067 therefore needs **three to four
genuinely decorrelated new members**, not one brilliant one. With unlimited GPU the
right move is to generate many *differently-wrong* models in parallel and let a greedy
selector keep the ones that actually decorrelate.

**4. The floor is ~0.005.** Single-model differences below that are unmeasurable on dev,
which is another reason to optimise the portfolio rather than any individual model.

## Strategy: maximise decorrelation per member, along independent axes

Rather than more variants of one architecture, I am varying the axes most likely to
produce *different failure modes*:

| Axis | Member | Rationale |
|---|---|---|
| Sequence mixing | **PE-SSM** | convolutional/recurrent bias vs attention |
| Capacity | medium AdaLN | different bias/variance operating point |
| **Objective** | **ordinal head** (new) | different loss geometry -- see below |
| Objective x mixing | **ordinal + SSM** | stacks the two most independent axes |
| Objective x features | ordinal + feature branch | different inputs *and* different loss |
| Seed | Family A seed 2 | cheapest reliable variance reduction |

### The ordinal head — the one genuinely new idea this round

Every member so far is trained on the simplex head: a 3-way cross-entropy over outcome
proportions (unedited, edited, indel). So every member shares a loss geometry, which
plausibly explains why their correlations bottom out around 0.945 and why nothing is
left in their residuals.

The ordinal head (CORAL-style) changes the learning problem rather than the
parameters. It predicts K-1 cumulative indicators `P(y > t_k)` at fixed quantiles of
the training efficiency distribution, trained with BCE, and scores a row by the mean of
those probabilities — an estimate of its **normalised rank**.

Two independent reasons to expect this to work:

- **It is metric-matched.** Spearman depends only on order. A head that estimates the
  target's quantile optimises the evaluated quantity directly, whereas the simplex head
  optimises proportions and is only incidentally monotone.
- **It is robust to the target's skew.** Prime-editing efficiency is heavily
  right-skewed with large mass at zero; a proportion/Huber objective spends most of its
  gradient on a few high-efficiency rows, while K-1 balanced binary decisions spread
  supervision across the whole distribution.

It also *gives up* the indel signal the simplex head uses. That is deliberate — it is a
different view of the same rows, and different views are what the ensemble rewards.
Implemented in `losses.py::ordinal_loss` and the `outcome_head="ordinal"` config path;
thresholds are computed from **training targets only** and stored in the checkpoint.

## Promotion rule (unchanged from spec §6/§18, applied honestly)

Advance a candidate if **S3 incremental ensemble gain ≥ +0.003 and positive on all
folds**, or **standalone gain ≥ +0.005**. Reject if **S2 correlation > 0.99 and S3 gain
< 0.001**. Standalone score alone never promotes.

## Sequence

1. **Screen** all candidates on dev fold 0 (7 runs in parallel), compute S1/S2/S2r/S3.
2. **Promote** survivors to full 5-fold training.
3. **Greedy ensemble construction** over the promoted pool on the three dev folds.
4. **Lockbox gate** (17,975 held-back rows) on the shortlist only.
5. **Freeze** `reports/round4_final_model_spec.md`.
6. **One** official held-out evaluation, 5000-resample protospacer-clustered bootstrap.

Step 6 happens exactly once. If it lands below 0.90 that is the reported result; I will
not re-open the search against the held-out set, because a number obtained by
re-selecting on the test set would not mean anything.

## Honest risks

- **0.90 may not be reachable.** The gap is +0.0067 and the per-member increment is
  decaying. Four good members might yield +0.005-0.008 — the target is plausible but
  genuinely uncertain, and I would rather say so now than explain it afterwards.
- **Dev→held-out transfer is unstable** (round 2's lesson: dev gains have reversed on
  held-out before). The lockbox exists to catch this, but it is a mitigation, not a
  guarantee.
- **The ordinal head may simply be worse standalone.** That is acceptable if it
  decorrelates; it is only a failure if it is both worse *and* correlated.
