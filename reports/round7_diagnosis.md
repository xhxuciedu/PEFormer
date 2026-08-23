# Round 7 — diagnosis before intervention

Rounds 5 and 6 ran ~40 model/loss/training variations and every one returned what a
mechanism-free re-run returns. Round 6 inferred the data was the limit. **That
inference was wrong**, and this round establishes what the limit actually is.

## 1. Kim has real headroom (the round-6 conclusion was wrong)

Kim contains genuine technical replicates: 649 groups where all 16 design and
condition covariates *and* the target site are identical.

| estimate | replicate ρ | ceiling √R |
|---|---:|---:|
| raw replicate pairs | 0.9531 | 0.976 |
| **reweighted, empirical** | **0.936** | **0.9026** |

Against **0.9026**, Ordinal-SSM's Kim score of **0.7869** leaves a **+0.1157** gap.
Kim is not noise-limited.

Two corrections were needed to get this right, and either shortcut would have
produced the opposite conclusion:

- **Naive replicate keys give 0.7987** — almost exactly our model's score, which would
  have "confirmed" saturation. But grouping on (spacer, rtt, pbs, cell, PE, Cas9)
  merges epegRNA with plain-pegRNA constructs, which differ in `motif`, `epegRNA` and
  `linker`. Those are different designs, not repeats.
- **Treating observed zeros as noiseless inflates the ceiling.** 21.4% of rows measured
  at exactly 0 have a non-zero replicate (median 0.0015). A zero is *censored*, not
  certain. The Gaussian estimate that assumed otherwise gave 0.968; the empirical
  nearest-neighbour resampler gives 0.9026. The difference falls almost entirely on the
  zero-heavy conditions the model does worst on — i.e. it would have manufactured
  headroom exactly where it matters most.

**The gap is not merely the zero block.** Restricting to rows that actually edit
(n=9,713) still leaves model 0.8465 vs ceiling 0.9489, a **+0.1024** gap.

## 2. Where the gap sits: the model under-uses experimental context

Per-condition gap to ceiling tracks zero-fraction and low efficiency:

| condition | %zero | model | ceiling | gap |
|---|---:|---:|---:|---:|
| A549/PE2 | 63.9% | 0.6009 | 0.7805 | +0.1796 |
| A549/PE4 | 62.9% | 0.6792 | 0.8555 | +0.1763 |
| MDA-MB-231/PE2 | 59.5% | 0.6623 | 0.8131 | +0.1507 |
| … | | | | |
| HCT116/PE2 | 53.7% | 0.7841 | 0.8708 | +0.0867 |

## 3. The mechanism: context rescales, it does not reorder

For designs measured in two conditions, across 45 condition pairs:

| | mean cross-condition ρ |
|---|---:|
| **True** (what the biology does) | **0.683** |
| **Model** (what it predicts) | **0.835** |
| excess | **+0.152** |

The model applies a near-universal sequence ranking and uses context mainly for scale.
Reality reorders designs substantially between cell lines and PE systems.

**And that reordering is learnable.** Predicting the cross-condition rank shift from
just the 16 engineered design features reaches held-out ρ ≈ **0.275** (A549/PE2 vs
DLD1/PE4, honest 2-fold split by design). It is real, recoverable signal — not noise —
and the model is not capturing it.

Note the magnitudes line up: the excess cross-condition similarity (+0.152) is the same
order as the gap to ceiling (+0.116).

## 4. The intervention

Late FiLM conditions **once, on the pooled summary**, so context can mostly rescale.
Layerwise FiLM conditions **at every block**, letting context shape the representation
as it is built.

Layerwise was implemented in round 2 and became a member of the final round-4 ensemble
— but **only for the Transformer**. On the SSM backbone, which the round-5 factorial
showed is the stronger architecture (+0.0192 vs +0.0092 for the objective), it raised
`ValueError: not implemented`. So the best backbone has never been able to use the
conditioning mechanism the diagnosis points at.

Round 7 closes that gap and tests it on **all three dev folds with matched controls**,
because round 6 established that a fold-0-only result at this effect size is not
evidence.

**Falsifiable prediction:** if the mechanism is right, layerwise-SSM should *lower* the
model's cross-condition rank correlation toward the true 0.683 while raising Kim
Spearman. If Kim improves but cross-condition similarity does not fall, the gain came
from somewhere else and the explanation is wrong.

---

# Result: the prediction was falsified, informatively

Layerwise context on the SSM backbone, three folds with matched controls:

| fold | layerwise | control | delta |
|---|---:|---:|---:|
| 0 | 0.8963 | 0.8964 | −0.0001 |
| 1 | 0.8994 | 0.8993 | +0.0001 |
| 2 | 0.8960 | 0.8957 | +0.0003 |

**Mean +0.0001.** No effect.

The pre-registered prediction was that layerwise conditioning would *lower* the model's
cross-condition rank correlation toward the true 0.683. It did the opposite:

| model | true xcond ρ | model xcond ρ | excess |
|---|---:|---:|---:|
| late FiLM (control) | 0.683 | 0.828 | +0.145 |
| **layerwise FiLM** | 0.683 | **0.875** | **+0.192** |

**More conditioning made the model *more* condition-invariant.** That is the opposite
of the intent and it explains the null cleanly.

**Why.** FiLM is structurally a per-channel scale and shift: `h' = (1+γ(c))·h + β(c)`.
Applying it at every block does not create capacity to *reorder* designs; it creates a
better pathway for representing the cross-condition **mean shift**. Cross-condition
mean differences explain a large share of pooled variance and are far easier to fit
than reordering, so extra conditioning capacity gets spent there, and the pressure on
the sequence pathway to specialise per condition goes *down*.

The diagnosis stands — context is under-used for reordering, and the reordering is
learnable (ρ ≈ 0.275 from design features alone). The instrument was wrong.

# Next intervention: attack the objective, not the architecture

If the mean shift is the shortcut, remove it from the objective. **ctx-primary** trains
the ordinal head on **within-condition quantiles** `F_c(y)` as the *primary* target, so
predicting the condition's mean earns no credit and the only way to reduce loss is to
rank correctly *within* a condition.

Location is then restored exactly rather than learned approximately, by inverting
through each condition's empirical **training** CDF:

    yhat = F_c^{-1}( qhat(sequence, context) )

This is a different proposition from round 5's §8 experiment, which added
context-normalised supervision as a small *auxiliary* head (λ ≤ 0.5) alongside the
global objective and scored +0.0016. Keeping the global target dominant left the
shortcut fully available; making it primary removes it.

Two variants are running on all three folds: pure ctx-primary, and ctx-primary with a
global auxiliary head retaining some absolute-location signal.

**Falsifiable prediction, again recorded in advance:** ctx-primary should reduce the
cross-condition excess below the control's +0.145 and improve Kim's mean
within-condition Spearman above 0.765. If pooled Kim rises without either of those
moving, the mechanism story is wrong and I will drop it rather than rationalise it.
