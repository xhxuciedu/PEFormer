# Round 6 plan — single-model performance

**Scope:** model architecture, loss functions, training scheme. **Not ensembling** —
the ensemble machinery is built and can be re-applied at the end; the question here is
how good one model can be.

**Reference:** Ordinal-SSM, **0.9082** OOF dev (Liu 0.8444, Kim 0.7912). The frozen
round-4 five-member ensemble is 0.9156 OOF dev / 0.9079 held-out. A single model is
only **0.0074** below the whole ensemble, so the single-model track is not obviously
the harder path.

---

## Why the round-5 approach failed, and what that implies

Round 5 added five mechanisms *on top of* the Ordinal-SSM backbone (auxiliary simplex,
multi-resolution ordinal, context-ordinal, hybrid mixers, quantile head). Every one
returned what a **mechanism-free re-run** returned: +0.0018 vs +0.0020. Combined with
the factorial's sub-additive interaction (−0.0043), the lesson is:

> Bolting additional supervision onto a backbone that already fits the data well is
> exhausted. The remaining error is not from missing auxiliary signal.

So round 6 does not add mechanisms. Every proposal below instead **fixes something
demonstrably wrong** with how the current model is specified or trained. Each is
backed by a measurement, not by analogy to other domains.

---

## The four leads, in priority order

### Lead 1 — Training distribution is grossly mismatched to evaluation *(training scheme)*

Measured composition:

| source | % of training | % of held-out | zero-mass |
|---|---:|---:|---:|
| Schwank (`pridict_pridict2`) | **58.4%** | **0.0%** | 10.3% |
| Liu (`hsu2026`) | 22.0% | 44.7% | 0.8% |
| Kim (`deepprime`) | **19.6%** | **55.3%** | 49.9% |

**Well over half of every gradient step is spent on a study that is never evaluated**,
while Kim — the largest evaluation partition and by far the weakest (0.7912 dev vs
Liu's 0.8444) — receives under a fifth of the training signal.

This is the single largest structural inefficiency in the setup, and it is a *training
scheme* problem, not a modelling one.

**Proposed experiments**
- **1a. Per-source loss weighting.** Scale each row's loss by `w_source`, sweeping Kim
  up toward its evaluation share. Cheap; one flag.
- **1b. Distribution-matched resampling.** Resample each epoch so the batch
  composition matches the evaluation mix (≈45% Liu / 55% Kim), keeping Schwank at a
  low residual weight for its regularising value.
- **1c. Schwank as pretraining only.** Two-stage: pretrain on everything, then train
  on Liu+Kim alone with a fresh LR schedule — a stronger version of round 3's DAPT.

**Why revisit DAPT, which round 3 rejected?** Two things changed, and the round-5 spec
explicitly permits revisiting a negative when the context changes. First, DAPT was
rejected for being **0.997 correlated** with an existing ensemble member — an
*ensemble* criterion that is irrelevant now that we are optimising a single model; its
standalone effect was **positive** (+0.0015). Second, it was applied to the round-1
Transformer, not the Ordinal-SSM backbone, and the factorial showed architecture
interacts with everything else.

**Risk, stated up front.** Down-weighting Schwank discards 174k rows of signal. If the
model is data-limited rather than allocation-limited this will hurt. That is exactly
what 1a's sweep measures, and the honest outcome may be that Schwank's volume is worth
more than its distributional mismatch costs.

---

### Lead 2 — The ordinal head is not rank-consistent *(loss / head)*

The head that round 4 credited as its breakthrough is described in our own code as
"CORAL-style". **It is not CORAL.** Its final layer has one *independent* weight vector
per threshold (17 × 384), so nothing constrains the predicted cumulative probabilities
to be non-increasing. Measured on the trained `r4p2_ordSSM_cv1` checkpoint:

```
rows with >= 1 monotonicity violation : 100.0%
mean violated threshold pairs per row : 8.0 of 16
```

Every row produces an **incoherent CDF** — it asserts P(y > t₅) > P(y > t₄) for roughly
half its thresholds. The ranking score averages those incoherent probabilities. It
works empirically (0.9082), but it is estimating a quantity that is not a distribution.

**Proposed experiments**
- **2a. True CORAL.** One shared weight vector, K−1 learned biases constrained to be
  ordered. Monotonicity holds **by construction**; strictly fewer parameters.
- **2b. CORN.** Model conditional probabilities P(y > t_k | y > t_{k−1}) and chain
  them, which is rank-consistent without forcing a shared weight vector — more
  expressive than 2a, so it separates "consistency helps" from "capacity constraint
  hurts".
- **2c. Monotonicity penalty.** Keep independent weights, add
  `λ · Σ_k max(0, p_{k+1} − p_k)`. Isolates the *constraint's* value from the
  *parameterisation's*.

Running all three is what distinguishes the two competing explanations, rather than
assuming rank consistency is the operative factor.

**Honest counter-hypothesis.** The violations may be harmless: averaging an incoherent
CDF can still be a fine ranking statistic, and 2a's weight sharing removes capacity
that may be doing real work. This lead is cheap and principled, not certain.

---

### Lead 3 — Zero-inflation is unmodelled *(loss / head)*

28.4% of rows are **exactly** 0.0; in Kim it is **49.9%**. The current head treats the
target as continuous on a quantile grid, with no term for the point mass at zero.

Round 5's rejected tie-floor experiment is the evidence this matters: artificially
tying the low-scoring rows gained **+0.0127 to +0.0162 on Kim for every model tested**.
I rejected it because it exploits Spearman's tie handling and would be unfair against
tie-free OptiPrime. **But the size of that number bounds how much signal sits in the
zero block.** A model that predicts *which* rows are zero — rather than one that
refuses to rank them — captures the same structure legitimately.

**Proposed experiments**
- **3a. Hurdle / two-part head.** A binary head for P(y > 0) plus the ordinal head
  trained *only on non-zero rows*; rank by `P(y>0) · E[rank | y>0]`. This matches the
  data-generating process directly and is distinct from the simplex head, which
  decomposes *outcome type* (unedited/edited/indel), not *zero versus positive*.
- **3b. Zero-aware ordinal.** Keep one head but add an explicit threshold at exactly
  0⁺, weighted up, so the model is directly supervised on the zero/non-zero boundary.
- **3c. Focal weighting** on the low thresholds, where the class balance is extreme.

Because the zero-mass is overwhelmingly a Kim phenomenon, **this lead and Lead 1 both
target Kim**, which holds most of the headroom. Report Liu and Kim separately
throughout — a Kim gain that costs Liu is not a win.

---

### Lead 4 — Scale the architecture that actually matters *(model)*

The factorial says architecture is the larger lever (**+0.0192** vs +0.0092 for the
objective). Yet the only scaling experiment ever run — round 4's "medium AdaLN", 59M
parameters — scaled the **Transformer**, which the factorial subsequently showed to be
the *worse* backbone. It gained +0.0022. **The SSM has never been scaled.**

Round 4 also showed the epoch budget may be binding: of 25 Phase-2 runs, **9 (36%)
peaked in the final five epochs** (best-epoch median 21, max 28).

**Proposed experiments**
- **4a. SSM state dimension:** 64 → 128, 256. Governs how much sequence history each
  channel retains; the cheapest capacity axis, and specific to the SSM.
- **4b. SSM depth:** 6+4 → 10+6 layers.
- **4c. Longer schedule:** 30 → 60 epochs with cosine decay, run *before* concluding
  anything about capacity — extra parameters that merely relieve an epoch shortage
  would be misattributed as a capacity win.
- **4d. EMA / SWA** of weights. Cheap, no architecture change, and untested here.

**Confound to control.** 4a/4b and 4c both plausibly explain a gain. Running the
longer schedule at the *current* size first separates "needs more capacity" from
"needs more steps" — otherwise a bigger model trained longer proves nothing about
which change mattered.

---

## Sequencing

**Wave 1 (parallel, all 8 GPUs) — the cheap, high-information runs**

| GPU | Experiment |
|---|---|
| 2, 6, 7 | 1a source weighting sweep (3 settings) |
| 6 | 2a true CORAL, 2b CORN |
| 0, 1 | 4c longer schedule at current size; 4d EMA |
| 3, 4 | 3a hurdle head; 3b zero-aware ordinal |
| 5 | 4a `ssm_state_dim=128` |

Every run screens on dev fold 0 against the Ordinal-SSM baseline, **and every wave
includes a plain re-run of the baseline as a control** — round 5 showed that without
one, uniform small positives are indistinguishable from noise. That control is now
non-negotiable in this project.

**Wave 2** — combine only what independently cleared the bar in Wave 1, testing for
the same sub-additivity the factorial predicts.

**Wave 3** — 3 dev folds, then the round-4 lockbox, then 5-fold OOF, then freeze.

## Promotion criteria

Single-model standalone gain ≥ **+0.005** over Ordinal-SSM on dev, positive on **all
three** folds. Kim and Liu reported separately; a Kim-driven gain that degrades Liu by
more than 0.002 does not promote. Ensembling is deferred entirely — but every promoted
model keeps its OOF predictions so the round-4 ensemble machinery can be re-applied
later at no extra cost.

## Expected value, stated honestly

Leads 1 and 3 target Kim, which has ~0.05 of headroom against Liu and holds 55% of the
evaluation set — that is where a real gain would come from. Leads 2 and 4 are cheaper
and more likely to yield ~0.002–0.005 each.

**Most likely outcome:** Lead 1 is the largest single effect, because a 58%/0%
train-eval mismatch is a big, fixable inefficiency and nothing else on the list is
comparably mis-specified. **Least likely:** Lead 2, which fixes a real incoherence that
may nonetheless be empirically harmless.

Given round 5's null and the sub-additive interaction, I would treat a **+0.005 to
+0.010 single-model gain** as a good round-6 outcome and anything above +0.015 as a
surprise worth double-checking for leakage before believing.
