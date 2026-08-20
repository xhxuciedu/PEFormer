# Round 4 results — PE-RankFormer vs OptiPrime

**Headline: 0.9079 Spearman on the official 20,509-row held-out set, +0.0389 over
OptiPrime (p < 0.0002, wins 100% of 5000 resamples).** Both round-4 goals are met:
the margin over OptiPrime exceeds the +0.03 target, and the 0.90 accuracy threshold
is cleared on the point estimate.

## Final held-out results

Single evaluation of the frozen model (`round4_final_model_spec.md`), never re-run.

| Model | Full (n=20,509) | Liu (n=9,175) | Kim (n=11,334) |
|---|---:|---:|---:|
| **PE-RankFormer (round 4)** | **0.9079** | **0.8585** | **0.8124** |
| PE-RankFormer (round 3) | 0.8933 | 0.8462 | 0.7836 |
| OptiPrime | 0.8690 | 0.8365 | 0.7320 |

Paired protospacer-clustered bootstrap, 5000 resamples over 750 clusters:

| Comparison | Δρ | 95% CI | wins | p |
|---|---:|---|---:|---:|
| **round 4 − OptiPrime** | **+0.0389** | [+0.0286, +0.0498] | 1.000 | < 0.0002 |
| round 4 − round 3 | +0.0146 | [+0.0113, +0.0184] | 1.000 | < 0.0002 |
| round 3 − OptiPrime | +0.0243 | [+0.0138, +0.0346] | 1.000 | < 0.0002 |

### The one caveat that belongs next to the headline

Round-4's absolute ρ has a 95% CI of **[0.8955, 0.9173]**, and **90.0%** of bootstrap
resamples exceed 0.90. So the 0.90 threshold is cleared by the point estimate on this
test set, but it is **not** established at 95% confidence — the interval's lower bound
sits just below. The *margin over OptiPrime* is a much stronger claim than the
absolute threshold: it is significant by any reading, winning every one of 5000
resamples.

The gain is far larger on Kim/DeepPrime (+0.0804 over OptiPrime) than on Liu
(+0.0220), consistent with every previous round.

## What produced the gain

Round 3 ended at 0.8933 and concluded that **error decorrelation**, not single-model
accuracy, was the only lever that had ever paid. Round 4 tested that conclusion to
destruction and then found the thing it was missing.

**Post-hoc combination is exhausted.** Three approaches, all on existing predictions:

| Approach | Result |
|---|---|
| Context-gated ensemble weights | +0.0000 (clean null) |
| Nonlinear stacking (GBM/ridge) | +0.0013 (below the +0.003 bar) |
| Residual learning on engineered features | **+0.0006 oracle ceiling** |

For the residual learner I bounded the method rather than reporting my tuning: even
choosing the blend weight *on the scoring fold itself* — an illegitimate procedure
used as a diagnostic ceiling — the maximum achievable was +0.0006. The ensemble's
residual is essentially unpredictable from engineered features and context. Together
these say nothing recoverable remained in the members' outputs, which is what
justified spending the round's compute on new *models* instead.

**The ordinal head was the breakthrough.** Every member through round 3 trained on the
same simplex 3-way cross-entropy, so they all shared a loss geometry — a plausible
reason their correlations bottomed out near 0.945. The new CORAL-style ordinal head
predicts K−1 cumulative indicators `P(y > t_k)` at fixed quantiles of the training
targets and scores a row by their mean, an estimate of its normalised rank. It is
metric-matched (Spearman depends only on order) and robust to the target's heavy
right skew.

The hypothesis was confirmed by the measurement designed to test it, not by a
convenient headline: the four lowest **residual** correlations in the screen
(0.686–0.708) were all ordinal-head models, while every simplex-head candidate sat at
0.76–0.78.

**Stacking the two new axes won.** `ordinal + SSM` combined the new objective with a
bidirectional state-space sequence mixer and achieved both the highest standalone
score and the lowest ensemble correlation — normally a trade-off. Out-of-fold on dev,
that single model scores **0.9082**, beating the entire round-3 three-member ensemble
(0.8982).

### Unbiased out-of-fold standalone scores (mean of 3 dev folds)

| Member | OOF dev ρ |
|---|---:|
| ordinal + SSM | **0.9082** |
| PE-SSM | 0.9011 |
| ordinal + features | 0.8924 |
| ordinal + layerwise | 0.8920 |
| ordinal (base) | 0.8911 |
| *round-3 ensemble (3 members)* | *0.8982* |
| Family A (best incumbent) | 0.8845 |

Every new member beats every incumbent.

## The refinement to round 3's conclusion

Round 3 concluded "decorrelation is the objective; standalone accuracy is close to
irrelevant." Round 4 shows that is **half right**, and the wave-2 ranking-loss model
is the clean counterexample:

| Candidate | S1 solo | S2 corr | S2r resid | S3 gain |
|---|---:|---:|---:|---:|
| simplex + ranking loss | 0.8244 | **0.9077** | **0.6654** | **−0.0037** |
| ordinal + SSM | 0.8972 | 0.9407 | 0.6863 | +0.0092 |

The ranking-loss model is the **most decorrelated candidate ever measured** on this
project — lowest prediction correlation *and* lowest residual correlation — and it
still **hurt** the ensemble, because at 0.8244 standalone it is too weak to carry its
weight in an equal-weight average.

So the correct statement is: **a member must be both decorrelated and competent.**
Decorrelation is what distinguishes a useful member from a redundant one *among
models of comparable strength* (which is why DAPT's 0.997 correlation made it
worthless despite being better standalone), but it cannot rescue a weak model. Round
4's gain came from candidates that were simultaneously stronger *and* differently
wrong — not from diversity alone.

## What did not work

| Idea | Result | Why it is informative |
|---|---|---|
| Protospacer bagging (×4) | S3 +0.0002 to −0.0022 | At ~38k training protospacers these models are not variance-limited; dropping 30% costs more than the diversity is worth |
| Medium AdaLN (59M params) | S3 +0.0022 | More capacity does not produce new errors |
| Family A, new seed | S3 +0.0022 | Neither do new seeds |
| Pairwise ranking loss | S3 −0.0037 | Most decorrelated candidate measured, still harmful — see above |
| Ordinal K sensitivity | S3 +0.0039 to +0.0043 across K ∈ {7, 18, 43} | Standalone varies with K (+0.0046 from 7→43) but ensemble contribution does not; K is not a sensitive knob |

Only new *mechanisms* — a different objective, a different sequence mixer — produced
new errors. Scale, seeds, and resampling did not.

## Method discipline

- **The selection protocol was pre-registered** before the ensemble search ran. When
  the dev winner (k=4) and lockbox winner (k=5) disagreed by 0.001011 — eleven
  millionths past the pre-set 0.001 near-tie threshold — the rule was followed as
  written rather than re-argued. The two candidates are statistically
  indistinguishable, so this was low-stakes, but re-deciding it after seeing the
  numbers is exactly what pre-registration exists to prevent.
- **The lockbox was used as a gate, never as a selection surface.** Its unconstrained
  best subset (0.9155) was *not* adopted, because selecting on it would have destroyed
  the only unworn evaluation data left.
- **The held-out set was evaluated once**, after the freeze, with the result committed
  in advance. The pre-recorded expectation was ~0.910 (plausible range 0.905–0.915);
  the actual 0.9079 landed inside it.
- **Phase-1 screening numbers were flagged as biased** in the log before they were
  quoted anywhere: candidates were best-epoch-selected on the rows they were scored
  on. Phase 2 retrained all five promoted members on the official 5-fold split to
  remove that bias, which is where the trustworthy numbers come from.

## Reproduction

```bash
# Phase 2: 5 members x 5 official folds
JOBS_FILE=... SCRATCH_DIR=... bash scripts/train/queue_runs.sh
# Phase 3: OOF dev predictions, ensemble search, lockbox gate
SCRATCH_DIR=... bash scripts/evaluate/run_round4_phase3.sh
PYTHONPATH=src .venv/bin/python scripts/evaluate/search_ensemble.py --members ...
# Final (requires the held-out guard flag; every run is audit-logged)
PYTHONPATH=src .venv/bin/python scripts/evaluate/evaluate_heterogeneous_heldout.py \
  --member "ordSSM:checkpoints/r4p2_ordSSM_cv*/best.pt" ... --allow-heldout-evaluation
PYTHONPATH=src .venv/bin/python scripts/evaluate/round4_final_bootstrap.py
```
