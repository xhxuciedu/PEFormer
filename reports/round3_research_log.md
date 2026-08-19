# Round-3 Research Log

Chronological experiment log per `claude_code_round3_pe_rankformer_experiments.md` §32.
Each entry: hypothesis, exact change, matched-dev result, Liu result, Kim result,
whether the hypothesis was supported, decision, next experiment.

---

## 2026-08-18 — Phase 0: inventory, baseline freeze, Liu+Kim-matched dev folds

**Hypothesis**: N/A (setup phase).

**Exact change**:
- `reports/round3_initial_inventory.md`: commits, corpus, architecture, existing
  checkpoints and which have known held-out scores (only round-1 baseline and
  round-2 Family C; Family A and D were Stage-A-only in round 2 and were never
  evaluated on held-out data).
- `configs/round3/baseline_round1.yaml`: frozen, byte-identical to round-1's
  actual winning config.
- `scripts/data/build_round3_dev_folds.py`: 3 Liu+Kim-matched, protospacer-
  disjoint development folds from the 297,962 training rows. Design: repeated
  random subsampling (not an exhaustive partition -- the 3 folds' validation
  sets may overlap each other, standard for repeated holdout), row-count-
  weighted (not protospacer-count-weighted -- Liu averages 65.8 rows/protospacer
  vs. Kim's 38.7, so a naive protospacer-count split skewed ~57% Liu instead of
  the target 44.7%; fixed by selecting protospacers via cumulative row count).
  Any protospacer also touching Schwank (62 Liu/Schwank + 241 Kim/Schwank
  collisions found) is pinned to always-train, since Schwank is otherwise
  unconditionally in every fold's training set and a shared protospacer sitting
  in both train (via its Schwank copy) and val (via its Liu/Kim copy) would be
  a real leak. Result: all 3 folds hit 44.6-44.7% Liu / 55.3-55.4% Kim (target:
  44.74%/55.26% exactly), verified protospacer-disjoint, zero Schwank in any
  validation set. 14 regression tests added (`tests/test_round3_dev_folds.py`).
- `scripts/train/train_pilot.py`: added `--dev-folds-file`/`--dev-fold-col` so
  training/evaluation can use the new dev folds instead of val_fold/train_folds.
  Smoke-tested (1 epoch, clean run).

**Development result**: N/A (infrastructure). 126/126 tests passing.

**Hypothesis supported**: N/A.

**Decision**: proceed to Stage 0's re-scoring sanity check (§6) -- evaluate the
4 existing checkpoints (round-1 baseline, round-2 Family A/C/D) on the new dev
folds and check whether the new ranking better predicts the 2 known held-out
outcomes (round-1 > Family C on held-out, despite Family C > round-1 on the
old Schwank-heavy validation).

**Next experiment**: Stage 0 sanity check (§6), then Phase 1 domain adaptation
(§8) if the sanity check passes.

---

## 2026-08-18 — Stage 0 sanity check (§6): INCONCLUSIVE, and it reframes the round

**Hypothesis** (§6): Liu+Kim-matched dev folds should rank round-1 above round-2
Family C, matching the known held-out ordering (0.8865 vs 0.8831), where the old
Schwank-heavy CV ranked them backwards (0.9192 vs 0.9202).

**Exact change**: `scripts/evaluate/evaluate_on_devfolds.py`, evaluating existing
checkpoints on the 3 matched dev folds.

**A methodological bug caught first**: evaluating the round-1 5-model ensemble on
the dev folds gave 0.9431 -- wildly above its known held-out 0.8865. Cause: each
ensemble member trains on 4 of the official folds 1-5, and the dev folds are drawn
from those same folds, so *every* dev row is ~80% in-sample. Added `--oof` mode
(score each row only with the checkpoint that held that row's official fold out).
OOF round-1 dev = 0.8798, now consistent with its 0.8865 held-out. Any dev-fold
number for an official-fold-trained checkpoint that isn't OOF is meaningless.

**Matched-dev result (OOF)**:

| Model | dev combined | dev Liu | dev Kim | held-out combined |
|---|---:|---:|---:|---:|
| round-1 baseline | 0.8798 | 0.8000 | 0.7517 | 0.8865 |
| round-2 Family C | 0.8816 | 0.8130 | 0.7534 | 0.8831 |

Matched dev says Family C is **+0.0018 better**; held-out says it is **-0.0034
worse**. The new validation reproduces the *same wrong ordering* as the old one.

**Diagnosis (paired protospacer-clustered bootstrap, per dev fold):**

| Dev fold | observed Δ (FamilyC − round1) | 95% CI | p |
|---|---:|---|---:|
| 0 | +0.0072 | [+0.0007, +0.0142] | **0.029** |
| 1 | +0.0014 | [−0.0044, +0.0073] | 0.664 |
| 2 | −0.0031 | [−0.0092, +0.0034] | 0.346 |
| *held-out (round 2)* | *−0.0034* | *[−0.0084, +0.0014]* | *0.18* |

The three folds **disagree with each other in sign**, and fold 0 produces a
nominally significant result in the direction opposite to held-out. This is not
a validation-design failure -- it is a **power** failure. The true round-1 vs
Family C difference is ~0.003 or smaller, and *no* evaluation set of this size
(dev or held-out, ~20-36k rows, ~750-800 protospacer clusters) can resolve it.
Note the held-out comparison was itself non-significant (p=0.18); round 2's
"regression" and this round's "improvement" are the same null result read twice.

**Hypothesis supported**: **No, and the check cannot be made to pass with this
model pair.** §6 says "if it does not, diagnose before proceeding" -- diagnosis
is that the check is underpowered by construction, not that the dev folds are
broken. The structural argument for matched dev folds (58% Schwank in training
vs 0% in test) is unchanged and still sound; they simply cannot be *validated*
against a model pair whose true difference is indistinguishable from zero.

**Decision -- this changes round-3 strategy.** The operative lesson from round 2
is not "the validation distribution was wrong" but "**a ±0.003 effect is not
worth chasing, because nothing in this project can measure it.**" Round 3 should
only pursue changes with expected effect sizes well above the ~0.005-0.01
resolution floor, and should require consistency across all 3 dev folds rather
than a mean. Concretely:
- Drop the low-yield end of the spec's plan (adapter-rank sweeps r=8/16/32,
  β-calibration variants, feature-branch revival) unless something large lands
  first.
- Prioritise the one intervention that plausibly moves 0.01+: **domain
  adaptation**, i.e. exploiting the fact that 58% of training is a source that
  is 0% of the target. That is a distributional intervention, not a
  ±0.003 architectural tweak.

**Next experiment**: Phase 1 domain adaptation (§8), designed so it can be
evaluated cleanly: train a global model *per dev fold* (so the dev fold's
validation rows are never in any training stage), then fine-tune on that fold's
Liu+Kim rows only. Requires new `--init-from` and `--train-sources` support in
the training script.

---

## 2026-08-18 — Ensemble DIVERSITY beats architectural improvement (unplanned, decisive)

**Hypothesis** (not in the round-3 spec's plan; came from the Stage-0 diagnosis):
if round-1 and round-2 Family C are statistically indistinguishable *individually*
(true Δ ~0.003, unresolvable), they may still make **different errors** -- in which
case blending them is worth far more than the difference between them. Round 2
treated them as competitors and picked one; that may have been the wrong frame.

**Exact change**: none to any model. Blended the already-computed OOF dev
predictions of the round-1 5-model ensemble and the Family C 5-model ensemble,
via simple mean and via rank-average.

**Matched-dev result (OOF, all 3 folds):**

| Dev fold | round-1 | Family C | mean blend | **rank-avg blend** |
|---|---:|---:|---:|---:|
| 0 | 0.8759 | 0.8831 | 0.8885 | **0.8899** |
| 1 | 0.8820 | 0.8834 | 0.8911 | **0.8926** |
| 2 | 0.8814 | 0.8783 | 0.8893 | **0.8903** |
| **mean** | 0.8798 | 0.8816 | 0.8896 | **0.8909** |

Rank-average blend gains **+0.0093 over the best single model**, and beats mean
blending on every fold. Paired protospacer-clustered bootstrap vs. the better
single model, per fold:

| Dev fold | observed Δ | 95% CI | bootstrap wins | p |
|---|---:|---|---:|---:|
| 0 | +0.0068 | [+0.0033, +0.0103] | 100% | <0.0001 |
| 1 | +0.0092 | [+0.0061, +0.0124] | 100% | <0.0001 |
| 2 | +0.0088 | [+0.0057, +0.0123] | 100% | <0.0001 |

**Hypothesis supported: emphatically yes.** Consistent in sign and magnitude
across all 3 folds, CI excludes zero everywhere, 100% bootstrap wins. Compare to
the round-1-vs-FamilyC difference this same machinery could *not* resolve
(p=0.03/0.66/0.35, signs disagreeing): the blend effect is ~3x larger and
qualitatively different in reliability. Inter-model prediction correlation is
0.909 -- correlated enough to be individually similar, decorrelated enough that
averaging cancels error.

**Why this matters more than the planned experiments**: round 2 spent an entire
round trying to find a *better single architecture* and moved the needle by an
unmeasurable ~0.003. Ten minutes of blending two *already-trained* models moved
it by a reliably-measurable ~0.009. The dominant axis of improvement here is
error decorrelation, not model quality.

**Decision -- reprioritise round 3.** Ensemble diversity becomes the primary
strategy (spec §25, promoted from last to first). Concretely:
1. Round-1 + Family C rank-average is already a validated finalist requiring
   zero new training -- both 5-checkpoint sets exist.
2. Add more *architecturally distinct* members trained across all 5 official
   folds, to extend the effect: Family A (layerwise context) currently has only
   its val_fold=1 checkpoint from round-2 Stage A; folds 2-5 are worth training.
3. Domain-adapted models (Phase 1, in flight) are valuable both on their own
   merits *and* as a further-decorrelated ensemble member -- a
   Liu+Kim-specialised model should make systematically different errors from a
   Schwank-heavy-trained one.

Extrapolating (with appropriate caution -- ensemble gains saturate): if a
2-architecture blend gives +0.009 over its best member, a 3-4 architecture blend
plausibly lands round-1's 0.8865 held-out near 0.895-0.90, which would clear the
spec's "Strong" bar vs OptiPrime (Δρ ≥ 0.025 vs the current +0.0175) and
approach "Excellent" (ρ_full ≥ 0.90). Not assumed -- to be measured on dev folds
before any held-out query.

**Next experiment**: Phase 1 domain adaptation (global models finishing now),
then Family A official folds 2-5 to add a third diverse ensemble member.

---

## 2026-08-18 — Phase 1 domain adaptation (§8): works, but is redundant with its parent

**Hypothesis** (§8): fine-tuning on Liu+Kim only should specialise the model to the
target distribution (58% of training is Schwank, 0% of the target).

**Exact change**: new `--init-from`, `--train-sources`, `--val-sources`, `--lr`.
Design chosen over the spec's literal instruction for a reason: rather than train
fresh global models per dev fold and fine-tune those, I fine-tuned the **existing
round-1 official-fold checkpoints**. Checkpoint k was trained on official folds
{1..5}\{k}; the fine-tune uses the same fold restriction, so fold k is unseen in
both stages and OOF dev evaluation stays clean -- and the 5 resulting checkpoints
are *directly usable* as a final ensemble member. The spec's version would have
required a separate retraining pass on official folds to produce the deliverable.
Validation restricted to Liu+Kim (`--val-sources`) so checkpoint selection targets
the benchmark.

**Result -- standalone (vs. each checkpoint's own pre-fine-tune control on the
Liu+Kim rows of its held-out fold):**

| Fold | pre | DAPT (lr 3e-5) | Δ |
|---|---:|---:|---:|
| 1 | 0.8798 | 0.8825 | +0.0027 |
| 2 | 0.8764 | 0.8775 | +0.0011 |
| 3 | 0.8801 | 0.8812 | +0.0011 |
| 4 | 0.8756 | 0.8778 | +0.0022 |
| 5 | 0.8835 | 0.8841 | +0.0006 |
| **mean** | | | **+0.0015** |

Positive on all 5 folds -- so a real effect, unlike the sign-flipping noise of the
Stage-0 comparison -- but small, right at the resolution floor. OOF dev: 0.8820
vs round-1's 0.8798.

**Hypothesis supported**: yes, weakly. Domain adaptation helps, but not enough to
matter on its own.

**The more useful finding -- why it doesn't help the ensemble.** Rank-prediction
correlations on dev fold 1:

| | round-1 | Family C | DAPT |
|---|---:|---:|---:|
| round-1 | 1.0000 | 0.9550 | **0.9970** |
| Family C | 0.9550 | 1.0000 | 0.9551 |
| DAPT | **0.9970** | 0.9551 | 1.0000 |

DAPT is **0.997-correlated with its parent** -- 10 epochs at lr 3e-5 barely moved
the function. So it is not a new ensemble member, it is a slightly better round-1.
Ensemble search confirms this exactly:

| Ensemble (rank-avg) | per-fold | mean |
|---|---|---:|
| Family C + DAPT | 0.8910 / 0.8936 / 0.8911 | **0.8919** |
| round-1 + Family C | 0.8899 / 0.8926 / 0.8903 | 0.8909 |
| round-1 + Family C + DAPT | 0.8887 / 0.8922 / 0.8905 | 0.8905 |
| round-1 + DAPT | 0.8781 / 0.8837 / 0.8830 | 0.8816 |

Adding DAPT *on top of* round-1 makes the 3-way blend **worse** than the best
2-way, and round-1+DAPT alone barely beats its members (fails the all-folds-win
test). Fitted weights independently confirm it: fitting on each fold drives
round-1's weight to 0.20, 0.00, 0.01 -- the optimiser discards round-1 whenever
DAPT is present. **DAPT is a replacement for round-1 in the ensemble, not an
addition.** Best blend so far: Family C + DAPT, 0.8919, better than
round-1+Family C on all 3 folds.

**Decision**: the operative variable is architectural decorrelation, and
fine-tuning does not produce it (0.997). To extend the ensemble further I need
genuinely different *architectures*, not further variations on round-1. Launched
Family A (layerwise context conditioning -- FiLM at every block rather than once
after pooling, trained from scratch) on official folds 2-5; fold 1 already exists
from round-2. Expect it to decorrelate like Family C (~0.955) rather than like
DAPT (~0.997).

Also killed the lr 1e-5 DAPT arm: at lr 3e-5 the model already stays 0.997-
correlated with its parent, so a smaller LR can only be more redundant. Kept the
5-fold lr 3e-5 set.

**Negative result recorded**: fitted ensemble weights do not beat equal weights
(0.8907 vs 0.8909 on the 2-member set; 0.8916 vs 0.8919 on the 3-member set), and
the fitted optimum is unstable across folds. Equal-weight rank-averaging is the
selection rule for this round -- one fewer thing to overfit.

**Next experiment**: Family A folds 2-5 (running), then a 3-way ensemble search
over {Family C, DAPT, Family A}.

---

## 2026-08-19 — Family A (layerwise context) re-evaluated under matched validation (§11)

**Hypothesis** (§11): round 2 found layerwise context conditioning helped its
(Schwank-heavy) validation but never took it to a held-out evaluation. Under the
corrected Liu+Kim-matched benchmark it should still help -- and, more importantly
given the decorrelation finding, it should be *architecturally* distinct enough to
extend the ensemble in a way domain adaptation could not.

**Exact change**: trained Family A (`--context-strategy layerwise`) on official
folds 2-5 with `--val-sources hsu2026 deepprime`, so checkpoint selection targets
the benchmark. Fold 1 already existed from round-2 Stage A.

**Result -- standalone, vs. each fold's round-1 control on the same Liu+Kim rows:**

| Fold | round-1 | Family A | Δ |
|---|---:|---:|---:|
| 2 | 0.8764 | 0.8812 | +0.0048 |
| 3 | 0.8801 | 0.8838 | +0.0037 |
| 4 | 0.8756 | 0.8839 | +0.0083 |
| 5 | 0.8835 | 0.8831 | −0.0004 |
| **mean** | | | **+0.0041** |

Positive on 3 of 4, mean +0.0041 -- nearly 3x domain adaptation's +0.0015 and the
largest single-model effect measured in rounds 2-3. Consistent with round 2's
finding that layerwise conditioning helps; the difference is that round 2 measured
it on the wrong distribution and then discarded it for failing to stack with
Family C.

**A consistency problem caught before it propagated**: the fold-1 Family A
checkpoint from round 2 was selected on Schwank-inclusive validation (its reported
0.9214 is on a different validation set entirely and is not comparable to the
0.881-0.884 above). Mixing it into a 5-fold ensemble would mean one member chosen
by a different, known-mismatched criterion. Retraining fold 1 with
`--val-sources` for consistency rather than reusing it.

**Hypothesis supported**: yes on the standalone question. The decorrelation
question is pending the fold-1 retrain, after which the full OOF ensemble search
over {Family C, DAPT, Family A} can run.

**Next experiment**: OOF-evaluate the consistent 5-fold Family A set, measure its
correlation with the other members, and search all subsets. Then freeze and run
the single held-out evaluation.

---

## 2026-08-19 — Final ensemble search: Family C + DAPT + Family A, round-1 excluded

**Full 4-member OOF evaluation** (round-1, Family C, DAPT, Family A) and exhaustive
subset search over {mean, rank-average} x all subsets, per §25.

**Individual members (mean over 3 dev folds):**

| Model | dev Spearman |
|---|---:|
| round-1 baseline | 0.8798 |
| Family C | 0.8816 |
| DAPT | 0.8820 |
| **Family A** | **0.8845** (new best single model) |

**Correlation structure** (rank-prediction Pearson r, dev fold 1) explains the
whole ensemble result:

| | round-1 | Family C | DAPT | Family A |
|---|---:|---:|---:|---:|
| round-1 | 1.000 | 0.955 | **0.997** | 0.946 |
| Family C | 0.955 | 1.000 | 0.955 | 0.945 |
| DAPT | 0.997 | 0.955 | 1.000 | 0.947 |
| **Family A** | 0.946 | 0.945 | 0.947 | 1.000 |

Family A is the most decorrelated member from *everything*, including Family C
-- explaining both why it is the strongest standalone model (§ previous entry)
and why it is the strongest ensemble addition.

**Best ensemble: Family C + DAPT + Family A (rank-average), round-1 excluded.**

| Dev fold | 3-way (best) | best single | Δ vs single |
|---|---:|---:|---:|
| 0 | 0.8979 | 0.8851 | +0.0128 |
| 1 | 0.8997 | 0.8868 | +0.0129 |
| 2 | 0.8969 | 0.8845 | +0.0124 |
| **mean** | **0.8982** | 0.8845 | **+0.0137** |

**Verified this is not multiple-comparisons cherry-picking.** Two direct paired,
protospacer-clustered bootstrap tests (2000 resamples each):

1. 3-way (FamilyC+DAPT+FamilyA) vs. the best 2-way (FamilyC+DAPT, 0.8919): all 3
   folds significant, CI excludes zero on every fold, **100% bootstrap wins,
   p<0.0001 on all three**. The 3rd member is adding real, reproducible signal.
2. 3-way vs. 4-way (adding round-1 back in): round-1 **significantly hurts** on
   all 3 folds (p<0.0001, p<0.0001, p=0.034). Confirms the correlation-matrix
   prediction directly -- round-1 and DAPT are 0.997-correlated, so including
   both just adds redundant weight to the same signal and dilutes the other two
   members' contribution.

**Decision**: freeze the ensemble as **Family C + DAPT + Family A**, equal
weights, rank-average combination, **round-1 excluded**. This is the frozen
round-3 model (spec §26).

Progression across this round's work: round-1 alone 0.8798 -> best 2-way blend
0.8919 (+0.0121) -> best 3-way blend 0.8982 (+0.0184 total). All measured on
matched Liu+Kim dev folds with consistent sign and significance across all 3
folds; held-out set still untouched.

**Next**: write reports/round3_final_model_spec.md, freeze and commit, then run
the single held-out evaluation via evaluate_heterogeneous_heldout.py.

---

## 2026-08-19 — Final held-out evaluation: significant win over round-1 AND OptiPrime

User authorized the held-out evaluation. Ran the frozen Family C + DAPT +
Family A ensemble (rank-average, round-1 excluded) via
`evaluate_heterogeneous_heldout.py` on all 20,509 held-out rows (see
`logs/heldout_evaluations.log`).

**Result: 0.8933 full-set Spearman**, vs. round-1's 0.8865, round-2's 0.8831,
OptiPrime's 0.8690. Improves on every prior round on every scope (full/Liu/Kim)
-- the first time across all three rounds a change hasn't traded one off
against another.

| Scope | OptiPrime | round-1 | round-2 | round-3 |
|---|---:|---:|---:|---:|
| Full | 0.8690 | 0.8865 | 0.8831 | **0.8933** |
| Liu | 0.8365 | 0.8349 | 0.8298 | **0.8462** |
| Kim | 0.7320 | 0.7751 | 0.7727 | **0.7836** |

**Statistical significance** (paired protospacer-clustered bootstrap, 5000
resamples): round-3 vs round-1 full +0.0068 [+0.0042,+0.0094] p<0.0001; vs
OptiPrime full +0.0243 [+0.0142,+0.0345] p<0.0001. Both partitions individually
significant vs round-1 too (Liu p=0.01, Kim p<0.0001) -- round 3 is the first
round to beat round-1 significantly on Liu specifically. Kim vs OptiPrime:
+0.0515, p<0.0001, and round-3 wins **10/10** individual Kim conditions
(n>=50 each).

Liu vs OptiPrime remains non-significant (+0.0098, p=0.33) -- statistically
tied, same qualitative conclusion as round 1, but no longer trending negative
the way round-2's single-architecture change did.

**Success criteria** (spec §29): "Useful" clearly met (+0.0068 vs round-1,
p<0.0001). "Strong" (Δρ_OptiPrime>=0.025) narrowly missed at 0.0243. "Excellent"
(ρ_full>=0.90) not met at 0.8933. Reported honestly rather than rounded up.

Wrote `reports/round3_results.md` (full spec-§33 writeup) documenting every
phase actually run, every phase skipped and why, and the central methodological
finding: at this project's measurement resolution (~750-800 protospacer
clusters), ensemble diversity is an order of magnitude more valuable than
further single-architecture search, and the two are easy to conflate because a
diversity-driving change (Family A, Family C) also happens to look like a small
standalone improvement -- but the standalone improvement is not where most of
the value comes from.
